"""Offline multi-GPU backend for the local 32B model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bian.models.dual_7b_backend import Dual7BBackend, GenerationConfig


class Sharded32BBackend(Dual7BBackend):
    """Reuse strict structured generation with an explicit multi-GPU map."""

    def __init__(
        self,
        model_path: Path,
        *,
        config: GenerationConfig,
        prompt_dir: Path,
        device_map: str | dict[str, Any] = "balanced",
        max_memory: dict[int, str] | None = None,
    ) -> None:
        super().__init__(
            model_path,
            config=config,
            prompt_dir=prompt_dir,
            response_prefix="{",
        )
        self.device_map = device_map
        self.max_memory = max_memory
        self.precision = "bfloat16"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._model is not None:
            return
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map=self.device_map,
            max_memory=self.max_memory,
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        devices = {
            str(device)
            for device in self._model.hf_device_map.values()
            if str(device) not in {"cpu", "disk", "meta"}
        }
        if len(devices) < 2:
            raise RuntimeError(
                f"32B BF16 did not shard across at least two GPUs: {devices}"
            )
        if any(str(device) in {"cpu", "disk", "meta"} for device in self._model.hf_device_map.values()):
            raise RuntimeError("unexpected 32B CPU/disk/meta offload")

    def model_manifest(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("model is not loaded")
        return {
            "model_path": str(self.model_path),
            "precision": self.precision,
            "local_files_only": True,
            "device_map_strategy": self.device_map,
            "max_memory": self.max_memory,
            "resolved_device_map": dict(self._model.hf_device_map),
        }
