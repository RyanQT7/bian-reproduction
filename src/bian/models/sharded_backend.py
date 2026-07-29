"""Offline multi-GPU backend for the local 32B model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

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
        precision: str = "bfloat16",
    ) -> None:
        super().__init__(
            model_path,
            config=config,
            prompt_dir=prompt_dir,
            response_prefix="{",
        )
        self.device_map = device_map
        self.max_memory = max_memory
        self.precision = precision

    def load(self) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if self._model is not None:
            return
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        quantization_config = None
        if self.precision == "int8":
            warnings.filterwarnings(
                "ignore", message="MatMul8bitLt: inputs will be cast"
            )
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.precision == "nf4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.precision != "bfloat16":
            raise ValueError(f"unsupported precision {self.precision!r}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map=self.device_map,
            max_memory=self.max_memory,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
        self._model.eval()
        resolved_map = getattr(self._model, "hf_device_map", self.device_map)
        if isinstance(resolved_map, str):
            resolved_map = {"": resolved_map}
        devices = {
            str(device)
            for device in resolved_map.values()
            if str(device) not in {"cpu", "disk", "meta"}
        }
        minimum_devices = 2 if self.precision == "bfloat16" else 1
        if len(devices) < minimum_devices:
            raise RuntimeError(
                f"32B {self.precision} used too few GPUs: {devices}"
            )
        if any(
            str(device) in {"cpu", "disk", "meta"}
            for device in resolved_map.values()
        ):
            raise RuntimeError("unexpected 32B CPU/disk/meta offload")
        self._resolved_device_map = resolved_map

    def model_manifest(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("model is not loaded")
        return {
            "model_path": str(self.model_path),
            "precision": self.precision,
            "local_files_only": True,
            "device_map_strategy": self.device_map,
            "max_memory": self.max_memory,
            "resolved_device_map": dict(self._resolved_device_map),
        }
