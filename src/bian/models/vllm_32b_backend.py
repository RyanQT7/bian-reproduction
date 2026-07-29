"""Offline vLLM multi-GPU BF16 backend with native structured outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable

from bian.data.validators import ValidationError
from bian.models.dual_7b_backend import GenerationConfig


@dataclass
class VLLMCall:
    request_id: str
    role: str
    prompt_version: str
    attempt: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    output_tokens_per_second: float
    finish_reason: str | None
    schema_constrained: bool
    raw_output: str
    reasoning: str
    error: str | None
    gpu_memory_mib: dict[str, float]


class VLLM32BBackend:
    """Keep one local 32B BF16 vLLM engine resident across all logical stages."""

    def __init__(
        self,
        model_path: Path,
        *,
        config: GenerationConfig,
        prompt_dir: Path,
        tensor_parallel_size: int,
        pipeline_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
        max_num_seqs: int = 16,
        enforce_eager: bool = True,
        raw_output_dir: Path | None = None,
        physical_gpu_ids: list[int] | None = None,
    ) -> None:
        self.model_path = model_path
        self.config = config
        self.prompt_dir = prompt_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.raw_output_dir = raw_output_dir
        self.physical_gpu_ids = physical_gpu_ids or []
        self.calls: list[VLLMCall] = []
        self._engine: Any = None
        self._tokenizer: Any = None
        self.load_seconds: float | None = None
        self._peak_gpu_memory_mib: dict[str, float] = {}

    def load(self) -> None:
        if self._engine is not None:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        from vllm import LLM

        started = time.perf_counter()
        self._engine = LLM(
            model=str(self.model_path),
            tokenizer=str(self.model_path),
            trust_remote_code=False,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            dtype="bfloat16",
            quantization=None,
            seed=self.config.seed,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            cpu_offload_gb=0,
            enforce_eager=self.enforce_eager,
            reasoning_parser="deepseek_r1",
            enable_prefix_caching=True,
            disable_log_stats=False,
        )
        self.load_seconds = time.perf_counter() - started
        self._tokenizer = self._engine.get_tokenizer()
        self._update_gpu_peak()

    def _prompt(
        self,
        name: str,
        payload: dict[str, Any],
        retry_error: str | None,
    ) -> str:
        template = (self.prompt_dir / f"{name}.txt").read_text(encoding="utf-8")
        suffix = ""
        if retry_error:
            suffix = (
                "\n\nPREVIOUS STRUCTURED OUTPUT FAILED VALIDATION: "
                + retry_error
                + "\nReturn one corrected JSON object that exactly matches the schema."
            )
        content = (
            template
            + "\n\nINPUT_JSON:\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + suffix
        )
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # DeepSeek-R1 Distill templates open a reasoning channel. The engine has
        # the matching parser configured; structured scoring deliberately starts
        # after an explicit close so only the final JSON enters the validator.
        return rendered + (
            "</think>\n"
            if rendered.rstrip().endswith("<think>")
            else "<think>\n</think>\n"
        )

    def _gpu_snapshot(self) -> dict[str, float]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            text = subprocess.check_output(command, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return {}
        available = {}
        for line in text.splitlines():
            index, memory = (part.strip() for part in line.split(",", 1))
            if not self.physical_gpu_ids or int(index) in self.physical_gpu_ids:
                available[index] = float(memory)
        return available

    def _update_gpu_peak(self) -> dict[str, float]:
        snapshot = self._gpu_snapshot()
        for key, value in snapshot.items():
            self._peak_gpu_memory_mib[key] = max(
                value, self._peak_gpu_memory_mib.get(key, 0.0)
            )
        return snapshot

    def _save_raw(self, call: VLLMCall) -> None:
        if self.raw_output_dir is None:
            return
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", call.request_id)
        path = self.raw_output_dir / f"{safe}_attempt{call.attempt}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite raw output {path}")
        path.write_text(
            json.dumps(asdict(call), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def generate_json_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        max_new_tokens: int,
    ) -> list[dict[str, Any]]:
        self.load()
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        results: list[dict[str, Any] | None] = [None] * len(requests)
        errors: dict[int, str] = {}
        pending = list(range(len(requests)))
        for attempt in range(1, self.config.retries + 2):
            prompts = []
            params = []
            input_tokens = []
            for index in pending:
                request = requests[index]
                prompt = self._prompt(
                    request["prompt_name"], request["payload"], errors.get(index)
                )
                count = len(self._tokenizer.encode(prompt))
                if count > self.config.max_input_tokens:
                    raise ValidationError(
                        f"{request['request_id']} input has {count} tokens; "
                        f"limit is {self.config.max_input_tokens}"
                    )
                prompts.append(prompt)
                input_tokens.append(count)
                params.append(
                    SamplingParams(
                        temperature=request.get(
                            "temperature", self.config.temperature
                        ),
                        top_p=request.get("top_p", self.config.top_p),
                        seed=request.get("seed", self.config.seed),
                        max_tokens=max_new_tokens,
                        structured_outputs=StructuredOutputsParams(
                            json=request["json_schema"],
                            disable_additional_properties=True,
                        ),
                    )
                )
            started = time.perf_counter()
            outputs = self._engine.generate(
                prompts, params, use_tqdm=False
            )
            elapsed = time.perf_counter() - started
            gpu_memory = self._update_gpu_peak()
            next_pending = []
            for position, index in enumerate(pending):
                request = requests[index]
                output = outputs[position].outputs[0]
                raw = output.text.strip()
                error = None
                reasoning = ""
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, dict):
                        raise ValidationError("structured output must be an object")
                    results[index] = request["validator"](parsed)
                except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    errors[index] = error
                    next_pending.append(index)
                output_count = len(output.token_ids)
                call = VLLMCall(
                    request_id=request["request_id"],
                    role=request["role"],
                    prompt_version=request["prompt_version"],
                    attempt=attempt,
                    input_tokens=input_tokens[position],
                    output_tokens=output_count,
                    elapsed_seconds=elapsed,
                    output_tokens_per_second=(
                        output_count / elapsed if elapsed else 0.0
                    ),
                    finish_reason=output.finish_reason,
                    schema_constrained=True,
                    raw_output=raw,
                    reasoning=reasoning,
                    error=error,
                    gpu_memory_mib=gpu_memory,
                )
                self.calls.append(call)
                self._save_raw(call)
            pending = next_pending
            if not pending:
                return [item for item in results if item is not None]
        failed = ", ".join(f"{index}: {errors[index]}" for index in pending)
        raise ValidationError(f"vLLM structured generation failed: {failed}")

    def generate_json(
        self,
        *,
        role: str,
        prompt_name: str,
        prompt_version: str,
        payload: dict[str, Any],
        validator: Callable[[dict[str, Any]], dict[str, Any]],
        json_schema: dict[str, Any],
        request_id: str,
        seed: int | None = None,
    ) -> dict[str, Any]:
        request = {
            "request_id": request_id,
            "role": role,
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "payload": payload,
            "validator": validator,
            "json_schema": json_schema,
        }
        if seed is not None:
            request["seed"] = seed
        return self.generate_json_batch(
            [request], max_new_tokens=self.config.max_new_tokens
        )[0]

    def call_manifest(self) -> list[dict[str, Any]]:
        return [asdict(call) for call in self.calls]

    def model_manifest(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "backend": "vllm",
            "precision": "bfloat16",
            "quantization": None,
            "local_files_only": True,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "enforce_eager": self.enforce_eager,
            "reasoning_parser": "deepseek_r1",
            "physical_gpu_ids": self.physical_gpu_ids,
            "load_seconds": self.load_seconds,
            "peak_gpu_memory_mib": dict(self._peak_gpu_memory_mib),
        }

