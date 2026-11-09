from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    ShardedStateDictConfig,
    StateDictType,
)
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from verl.trainer.config import CheckpointConfig
from verl.utils.fsdp_utils import get_init_weight_context_manager
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _prompt_visual_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
    prompt_length = getattr(self, "_echoes_prompt_length", None)
    mask_ids = input_ids
    if prompt_length is not None and input_ids is not None:
        mask_ids = input_ids.clone()
        suffix = mask_ids[:, prompt_length:]
        visual = (suffix == self.config.image_token_id) | (suffix == self.config.video_token_id)
        # This copy controls visual scatter only; response embeddings remain unchanged.
        neutral_id = 0
        while neutral_id in {self.config.image_token_id, self.config.video_token_id}:
            neutral_id += 1
        suffix.masked_fill_(visual, neutral_id)
    return self._echoes_native_visual_mask(
        mask_ids, inputs_embeds, image_features=image_features, video_features=video_features
    )


class EchoesEngine(FSDPEngine):
    """verl FSDP training with native multimodal forwards and exact CTA logits."""

    def _build_module(self):
        context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings,
            mesh=self.device_mesh,
        )
        with context():
            module = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_config.local_path,
                config=self.model_config.hf_config,
                dtype=torch.float32,
                trust_remote_code=False,
            )
        module.model._echoes_native_visual_mask = module.model.get_placeholder_mask
        module.model.get_placeholder_mask = MethodType(_prompt_visual_mask, module.model)
        module.model._echoes_prompt_length = None
        if self.engine_config.forward_only:
            module.requires_grad_(False)
        elif self.model_config.enable_gradient_checkpointing:
            module.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return module

    def _build_fsdp_module(self, module):
        # Keep teacher and student shards colocated for round-boundary refreshes.
        original = self.engine_config
        if original.forward_only:
            self.engine_config = replace(original, forward_only=False)
        try:
            return super()._build_fsdp_module(module)
        finally:
            self.engine_config = original

    @property
    def hf_model(self):
        return self.module.module

    @property
    def processor(self):
        return self.model_config.processor

    @property
    def tokenizer(self):
        return self.model_config.tokenizer

    @property
    def device(self):
        return torch.device("cuda", torch.cuda.current_device())

    def _inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        result = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
            if key not in {"labels", "position_ids", "past_key_values", "cache_position"}
        }
        if result["input_ids"].ndim != 2 or result["input_ids"].shape[0] != 1:
            raise ValueError("Each multimodal microbatch must contain exactly one prompt.")
        if "attention_mask" not in result:
            result["attention_mask"] = torch.ones_like(result["input_ids"])
        if "image_grid_thw" in result and "mm_token_type_ids" not in result:
            result["mm_token_type_ids"] = (
                result["input_ids"] == self.model_config.hf_config.image_token_id
            ).long()
        return result

    def response_logits(self, inputs: Mapping[str, Any], response_ids: torch.Tensor):
        """Score the identical response tokens after this view's own prompt."""
        batch = self._inputs(inputs)
        response_ids = torch.as_tensor(response_ids, device=self.device, dtype=torch.long).reshape(1, -1)
        count = response_ids.shape[-1]
        if count == 0:
            raise ValueError("A response must contain at least one token.")
        prompt_length = batch["input_ids"].shape[-1]
        batch["input_ids"] = torch.cat((batch["input_ids"], response_ids), dim=-1)
        batch["attention_mask"] = torch.cat(
            (batch["attention_mask"], torch.ones_like(response_ids)), dim=-1
        )
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in batch:
                batch[key] = torch.cat((batch[key], torch.zeros_like(response_ids)), dim=-1)
        native = self.hf_model.model
        native.rope_deltas = None
        native._echoes_prompt_length = prompt_length
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.module(
                    **batch, use_cache=False, logits_to_keep=count + 1, return_dict=True
                )
        finally:
            native._echoes_prompt_length = None
            native.rope_deltas = None
        return output.logits[:, :-1, :]

    @torch.no_grad()
    def generate_ids(
        self, inputs: Mapping[str, Any], max_new_tokens: int, temperature: float
    ) -> torch.Tensor:
        """Sample through FSDP forwards while synchronizing rank termination."""
        if max_new_tokens < 1 or temperature < 0:
            raise ValueError("Generation requires a positive length and a nonnegative temperature.")
        batch = self._inputs(inputs)
        training = self.module.training
        self.module.eval()
        self.hf_model.model.rope_deltas = None
        eos_ids = self.hf_model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.model_config.tokenizer.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        if not eos_ids:
            raise ValueError("The model must define an end-of-sequence token.")
        stop_ids = torch.tensor(eos_ids, device=self.device)
        generated: list[torch.Tensor] = []
        finished = False
        attention_mask = batch["attention_mask"]
        try:
            for _ in range(max_new_tokens):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = self.module(
                        **batch, use_cache=True, logits_to_keep=1, return_dict=True
                    )
                logits = output.logits[:, -1, :].float()
                if temperature == 0:
                    token = logits.argmax(dim=-1, keepdim=True)
                else:
                    token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
                if not finished:
                    generated.append(token[0, 0])
                    finished = bool(torch.isin(token, stop_ids).any().item())
                else:
                    token.fill_(eos_ids[0])
                active = torch.tensor(0 if finished else 1, device=self.device)
                dist.all_reduce(active, op=dist.ReduceOp.SUM)
                if active.item() == 0:
                    break
                attention_mask = torch.cat((attention_mask, torch.ones_like(token)), dim=-1)
                position = attention_mask.long().cumsum(-1)[:, -1:] - 1
                if output.rope_deltas is not None:
                    position = position + output.rope_deltas.to(self.device)
                batch = {
                    "input_ids": token,
                    "attention_mask": attention_mask,
                    "past_key_values": output.past_key_values,
                    "position_ids": position.unsqueeze(0).expand(3, -1, -1),
                }
            return torch.stack(generated).long()
        finally:
            self.hf_model.model.rope_deltas = None
            self.module.train(training)

    @torch.no_grad()
    def refresh_from(self, student: "EchoesEngine") -> None:
        if not self.engine_config.forward_only:
            raise ValueError("Only a frozen teacher can receive a round refresh.")
        settings = ShardedStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(student.module, StateDictType.SHARDED_STATE_DICT, settings):
            state = student.module.state_dict()
        with FSDP.state_dict_type(self.module, StateDictType.SHARDED_STATE_DICT, settings):
            self.module.load_state_dict(state, strict=True)
        self.module.eval()
        self.hf_model.model.rope_deltas = None
        dist.barrier()

    def _get_hf_export_per_tensor_param(self):
        settings = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.module, StateDictType.FULL_STATE_DICT, settings):
            state = self.module.state_dict()
        return iter(state.items()), None

    def export_hf(self, output_path: str | Path) -> None:
        destination = Path(output_path)
        if self.rank == 0:
            destination.mkdir(parents=True, exist_ok=True)
            config = deepcopy(self.hf_model.config)
            config.dtype = torch.bfloat16
            config.text_config.dtype = torch.bfloat16
            config.save_pretrained(destination)
            self.hf_model.generation_config.save_pretrained(destination)
            self.model_config.get_processor().save_pretrained(destination)
        dist.barrier()
        self.checkpoint_manager._save_hf_weights_from_per_tensor_param(str(destination))
        dist.barrier()


def create_engine(config: Mapping[str, Any], total_steps: int, teacher: bool = False) -> EchoesEngine:
    if not dist.is_initialized():
        raise RuntimeError("Initialize the torchrun process group before constructing an engine.")
    model = config["model"]
    optimizer = config.get("optimizer", {})
    runtime = config.get("runtime", {})
    if total_steps < 1:
        raise ValueError("total_steps must be positive.")
    model_config = HFModelConfig(
        path=model["path"],
        trust_remote_code=False,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=bool(runtime.get("gradient_checkpointing", True)) and not teacher,
        override_config={"attn_implementation": runtime.get("attention_backend", "sdpa")},
    )
    if model_config.hf_config.model_type != "qwen3_vl":
        raise ValueError("ECHOES requires the Qwen3-VL architecture.")
    model_config.processor = AutoProcessor.from_pretrained(
        model_config.local_path,
        min_pixels=int(model.get("min_pixels", 65536)),
        max_pixels=int(model.get("max_pixels", 1048576)),
        trust_remote_code=False,
    )
    model_config.tokenizer = model_config.processor.tokenizer
    engine = EchoesEngine(
        model_config=model_config,
        engine_config=FSDPEngineConfig(
            strategy="fsdp",
            forward_only=teacher,
            model_dtype="fp32",
            use_remove_padding=False,
            use_fused_kernels=False,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1,
            infer_micro_batch_size_per_gpu=1,
            use_torch_compile=False,
            param_offload=False,
            optimizer_offload=False,
            seed=int(runtime.get("seed", 42)),
            mixed_precision={"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
        ),
        optimizer_config=FSDPOptimizerConfig(
            lr=float(optimizer.get("lr", 1e-6)),
            weight_decay=float(optimizer.get("weight_decay", 0.01)),
            lr_warmup_steps_ratio=float(optimizer.get("warmup_ratio", 0.03)),
            clip_grad=float(optimizer.get("clip_grad", 1.0)),
            total_training_steps=total_steps,
            lr_scheduler_type="cosine",
            min_lr_ratio=0.1,
        ),
        checkpoint_config=CheckpointConfig(
            save_contents=["model", "optimizer", "extra", "hf_model"],
            load_contents=["model", "optimizer", "extra"],
        ),
    )
    engine.initialize()
    if teacher:
        engine.module.eval()
    return engine
