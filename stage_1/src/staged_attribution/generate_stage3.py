from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from .adapter_storage import (
    restore_and_validate_loaded_compact_adapter,
)
from .token_protocol import TokenProtocolError, parse_answer, special_tokens


GENERATION_SYSTEM_MESSAGE = (
    "You are a professional spatial-temporal generation expert who needs to generate the "
    "possible travel pattern and sampling type for users based on their trip information."
)
THINK_BLOCK = "<think>\n\n</think>\n\n"
_SEQUENCE = re.compile(r"<t_[^>]+>, <\|t_begin\|>.*?<\|t_end\|>", re.DOTALL)


class GenerationError(ValueError):
    """Raised when Stage3 generation population accounting is invalid."""


def format_generation_prompt(description: str) -> str:
    return (
        f"<|im_start|>system\n{GENERATION_SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{description}<|im_end|>\n"
        f"<|im_start|>assistant\n{THINK_BLOCK}"
    )


def _failure(traj_id: int, continuation: str, reason: str, message: str) -> dict[str, object]:
    return {
        "traj_id": int(traj_id),
        "status": "failed",
        "raw_continuation": continuation,
        "failure": {"reason": reason, "message": message},
    }


def parse_generation(traj_id: int, continuation: str) -> dict[str, object]:
    if not isinstance(continuation, str):
        return _failure(traj_id, repr(continuation), "non_string", "continuation is not text")
    matches = _SEQUENCE.findall(continuation)
    if len(matches) > 1:
        return _failure(
            traj_id, continuation, "duplicate_sequence", f"found {len(matches)} sequence envelopes"
        )
    if not matches:
        reason = "truncated" if "<|t_begin|>" in continuation or "<t_" in continuation else "missing_sequence"
        return _failure(traj_id, continuation, reason, "no complete HTP sequence envelope")
    try:
        parsed = parse_answer(matches[0])
    except TokenProtocolError as error:
        return _failure(traj_id, continuation, "invalid_protocol", str(error))
    return {
        "traj_id": int(traj_id),
        "status": "completed",
        "lens_type": parsed["lens_type"],
        "codes": parsed["codes"],
        "answer": matches[0],
        "raw_continuation": continuation,
        "failure": None,
    }


def validate_generation_population(
    records: Sequence[Mapping[str, object]], expected_traj_ids: Iterable[int]
) -> dict[str, int]:
    observed = [int(record["traj_id"]) for record in records]
    expected = [int(value) for value in expected_traj_ids]
    if len(set(observed)) != len(observed):
        raise GenerationError("duplicate generated traj_id")
    if len(set(expected)) != len(expected):
        raise GenerationError("duplicate expected traj_id")
    if set(observed) != set(expected):
        raise GenerationError(
            f"generation population mismatch: missing={sorted(set(expected)-set(observed))[:10]}, "
            f"extra={sorted(set(observed)-set(expected))[:10]}"
        )
    completed = sum(record.get("status") == "completed" for record in records)
    return {"population": len(records), "completed": completed, "failed": len(records) - completed}


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _GenerationProgressRecorder:
    """Persist Stage3 population accounting without affecting generation decisions."""

    def __init__(
        self,
        path: str | Path,
        population_total: int,
        batch_size: int,
        progress_interval: int,
    ) -> None:
        if population_total <= 0:
            raise GenerationError("generation progress population_total must be positive")
        if batch_size <= 0:
            raise GenerationError("generation batch_size must be positive")
        if progress_interval <= 0:
            raise GenerationError("generation progress_interval must be positive")
        self.path = Path(path)
        self.population_total = int(population_total)
        self.batch_size = int(batch_size)
        self.batches_total = (self.population_total + self.batch_size - 1) // self.batch_size
        self.progress_interval = int(progress_interval)
        self.started_at = _utc_now()
        self.started = time.monotonic()
        self.batches_completed = 0
        self.population_completed = 0
        self.outputs_completed = 0
        self.outputs_failed = 0
        self.last_traj_id: int | None = None
        self._write("loading_model")

    def _payload(
        self,
        status: str,
        *,
        error: Exception | None = None,
        audit: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        elapsed = max(0.0, time.monotonic() - self.started)
        rate = self.population_completed / elapsed if elapsed > 0 else 0.0
        remaining = self.population_total - self.population_completed
        eta = remaining / rate if rate > 0 else None
        return {
            "schema_version": 1,
            "stage": "stage3_generation",
            "status": status,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "batch_size": self.batch_size,
            "batches_total": self.batches_total,
            "batches_completed": self.batches_completed,
            "population_total": self.population_total,
            "population_completed": self.population_completed,
            "outputs_completed": self.outputs_completed,
            "outputs_failed": self.outputs_failed,
            "fraction_completed": self.population_completed / self.population_total,
            "elapsed_seconds": elapsed,
            "records_per_second": rate,
            "eta_seconds": eta,
            "last_traj_id": self.last_traj_id,
            "audit": dict(audit) if audit is not None else None,
            "error": (
                {"type": type(error).__name__, "message": str(error)}
                if error is not None
                else None
            ),
        }

    def _write(
        self,
        status: str,
        *,
        error: Exception | None = None,
        audit: Mapping[str, object] | None = None,
    ) -> None:
        _atomic_json(self.path, self._payload(status, error=error, audit=audit))

    def running(self) -> None:
        self._write("running")

    def update(self, batch_records: Sequence[Mapping[str, object]]) -> None:
        if not batch_records:
            raise GenerationError("generation progress batch must be non-empty")
        next_population = self.population_completed + len(batch_records)
        if next_population > self.population_total:
            raise GenerationError("generation progress exceeds population_total")
        statuses = [record.get("status") for record in batch_records]
        if any(status not in {"completed", "failed"} for status in statuses):
            raise GenerationError("generation progress found an invalid output status")
        self.batches_completed += 1
        self.population_completed = next_population
        self.outputs_completed += sum(status == "completed" for status in statuses)
        self.outputs_failed += sum(status == "failed" for status in statuses)
        self.last_traj_id = int(batch_records[-1]["traj_id"])
        if (
            self.batches_completed % self.progress_interval == 0
            or self.population_completed == self.population_total
        ):
            self._write("running")

    def complete(self, audit: Mapping[str, object]) -> None:
        if self.batches_completed != self.batches_total:
            raise GenerationError("generation ended before all batches completed")
        if self.population_completed != self.population_total:
            raise GenerationError("generation ended before the full population completed")
        if self.outputs_completed != int(audit["completed"]):
            raise GenerationError("generation progress completed count disagrees with audit")
        if self.outputs_failed != int(audit["failed"]):
            raise GenerationError("generation progress failed count disagrees with audit")
        self._write("completed", audit=audit)

    def fail(self, error: Exception) -> None:
        self._write("failed", error=error)


@torch.no_grad()
def run_hf_generation(
    frame: pd.DataFrame,
    adapter_path: str | Path,
    output_path: str | Path,
    num_roads: int,
    seed: int,
    batch_size: int = 16,
    max_new_tokens: int = 256,
    device: str = "cuda",
    progress_path: str | Path | None = None,
    progress_interval: int = 10,
    runtime_guard=None,
) -> dict[str, object]:
    """Generate every requested ID and preserve malformed outputs as explicit failures."""
    if frame["traj_id"].duplicated().any():
        raise GenerationError("generation frame contains duplicate traj_id")
    ordered = frame.sort_values("traj_id").reset_index(drop=True)
    progress = (
        _GenerationProgressRecorder(
            progress_path,
            population_total=len(ordered),
            batch_size=batch_size,
            progress_interval=progress_interval,
        )
        if progress_path is not None
        else None
    )
    results: list[dict[str, object]] = []
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, set_seed

        set_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        protocol = special_tokens(num_roads)
        protocol_ids = tokenizer.convert_tokens_to_ids(protocol)
        if any(value == tokenizer.unk_token_id for value in protocol_ids):
            raise GenerationError("generation tokenizer is missing protocol tokens")
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(adapter_path),
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        ).to(device)
        model.eval()
        restore_and_validate_loaded_compact_adapter(model, adapter_path)
        set_seed(seed)
        if progress is not None:
            progress.running()
        for start in range(0, len(ordered), batch_size):
            if runtime_guard is not None:
                runtime_guard()
            rows = ordered.iloc[start : start + batch_size]
            prompts = [format_generation_prompt(str(value)) for value in rows["desc"]]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_width = encoded["input_ids"].shape[1]
            continuation_ids = generated[:, prompt_width:]
            continuations = tokenizer.batch_decode(
                continuation_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            batch_records = [
                parse_generation(int(traj_id), continuation)
                for traj_id, continuation in zip(rows["traj_id"], continuations)
            ]
            results.extend(batch_records)
            if progress is not None:
                progress.update(batch_records)
        audit = validate_generation_population(results, ordered["traj_id"])
        _atomic_jsonl(Path(output_path), results)
        if progress is not None:
            progress.complete(audit)
    except Exception as error:
        _atomic_jsonl(Path(output_path).with_suffix(".partial.jsonl"), results)
        if progress is not None:
            progress.fail(error)
        raise
    return {
        "audit": audit,
        "records": results,
        "output_path": str(Path(output_path).resolve()),
        "progress_path": str(Path(progress_path).resolve()) if progress_path is not None else None,
    }
