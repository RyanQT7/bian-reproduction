"""Offline BF16 backend shared by the two logical Dual-7B roles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

from bian.data.validators import ValidationError
from bian.models.structured_output import parse_strict_json, strip_thinking


@dataclass(frozen=True)
class GenerationConfig:
    max_input_tokens: int = 4096
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 1.0
    retries: int = 2
    seed: int = 42


@dataclass
class ModelCall:
    role: str
    prompt_version: str
    attempt: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    peak_gpu_memory_mib: float
    raw_output: str
    thinking: str
    json_repair_used: bool
    error: str | None


class Dual7BBackend:
    """One resident model instance with separate logical roles and prompts."""

    def __init__(
        self,
        model_path: Path,
        *,
        config: GenerationConfig,
        prompt_dir: Path,
    ) -> None:
        self.model_path = model_path
        self.config = config
        self.prompt_dir = prompt_dir
        self._model = None
        self._tokenizer = None
        self.calls: list[ModelCall] = []

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._model is not None:
            return
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    def prompt_hash_inputs(self) -> list[Path]:
        return sorted(self.prompt_dir.glob("*.txt"))

    def _prompt(self, name: str, payload: dict[str, Any], retry_error: str | None) -> str:
        template = (self.prompt_dir / f"{name}.txt").read_text(encoding="utf-8")
        suffix = ""
        if retry_error:
            suffix = (
                "\n\nPREVIOUS OUTPUT FAILED VALIDATION: "
                + retry_error
                + "\nReturn only one corrected JSON object with the exact schema."
            )
        return template + "\n\nINPUT_JSON:\n" + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ) + suffix

    def generate_json(
        self,
        *,
        role: str,
        prompt_name: str,
        prompt_version: str,
        payload: dict[str, Any],
        validator: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        import torch

        self.load()
        assert self._model is not None and self._tokenizer is not None
        previous_error: str | None = None
        for attempt in range(1, self.config.retries + 2):
            prompt = self._prompt(prompt_name, payload, previous_error)
            rendered = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            # Close the R1 reasoning channel before generation. Any emitted thinking
            # is still separated and logged by parse_strict_json.
            rendered += "<think>\n</think>\n"
            inputs = self._tokenizer(
                rendered,
                return_tensors="pt",
                truncation=False,
            ).to("cuda:0")
            input_tokens = int(inputs["input_ids"].shape[-1])
            if input_tokens > self.config.max_input_tokens:
                del inputs
                raise ValidationError(
                    f"{role}/{prompt_name} input has {input_tokens} tokens, exceeding "
                    f"configured limit {self.config.max_input_tokens}; refusing to truncate"
                )
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.temperature > 0,
                    temperature=(
                        self.config.temperature if self.config.temperature > 0 else None
                    ),
                    top_p=self.config.top_p if self.config.temperature > 0 else None,
                    pad_token_id=self._tokenizer.eos_token_id,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            new_ids = outputs[0, input_tokens:]
            raw = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            thinking = ""
            repair_used = False
            error = None
            try:
                try:
                    parsed, thinking = parse_strict_json(raw)
                except ValidationError:
                    import json_repair

                    thinking, answer = strip_thinking(raw)
                    if answer.startswith("```json") and answer.endswith("```"):
                        answer = answer[len("```json") : -len("```")].strip()
                    elif answer.startswith("```") and answer.endswith("```"):
                        answer = answer[len("```") : -len("```")].strip()
                    parsed = json_repair.loads(answer)
                    if not isinstance(parsed, dict):
                        raise ValidationError("repaired model answer must be an object")
                    repair_used = True
                validated = validator(parsed)
            except (ValidationError, ValueError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                previous_error = error
            self.calls.append(
                ModelCall(
                    role=role,
                    prompt_version=prompt_version,
                    attempt=attempt,
                    input_tokens=input_tokens,
                    output_tokens=int(new_ids.numel()),
                    elapsed_seconds=elapsed,
                    peak_gpu_memory_mib=torch.cuda.max_memory_allocated(0) / 1024**2,
                    raw_output=raw,
                    thinking=thinking,
                    json_repair_used=repair_used,
                    error=error,
                )
            )
            del outputs, inputs
            if error is None:
                return validated
        raise ValidationError(
            f"{role}/{prompt_name} failed after {self.config.retries + 1} attempts: "
            f"{previous_error}"
        )

    def call_manifest(self) -> list[dict[str, Any]]:
        return [asdict(call) for call in self.calls]
