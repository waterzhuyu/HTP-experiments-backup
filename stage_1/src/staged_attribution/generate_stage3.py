from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import torch

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
) -> dict[str, object]:
    """Generate every requested ID and preserve malformed outputs as explicit failures."""
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, set_seed

    if frame["traj_id"].duplicated().any():
        raise GenerationError("generation frame contains duplicate traj_id")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    protocol = special_tokens(num_roads)
    protocol_ids = tokenizer.convert_tokens_to_ids(protocol)
    if any(value == tokenizer.unk_token_id for value in protocol_ids):
        raise GenerationError("generation tokenizer is missing protocol tokens")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_path), torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
    ).to(device)
    model.eval()
    ordered = frame.sort_values("traj_id").reset_index(drop=True)
    results: list[dict[str, object]] = []
    for start in range(0, len(ordered), batch_size):
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
            continuation_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for traj_id, continuation in zip(rows["traj_id"], continuations):
            results.append(parse_generation(int(traj_id), continuation))
    audit = validate_generation_population(results, ordered["traj_id"])
    _atomic_jsonl(Path(output_path), results)
    return {"audit": audit, "records": results, "output_path": str(Path(output_path).resolve())}
