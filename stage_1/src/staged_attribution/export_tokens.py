from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch

from .token_protocol import TokenProtocolError, parse_answer, serialize_answer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _id_hash(values: Sequence[int]) -> str:
    payload = "\n".join(str(value) for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalize_record(record: Mapping[str, object]) -> dict[str, object]:
    required = {"traj_id", "lens_type", "codes", "road_seq"}
    missing = sorted(required - set(record))
    if missing:
        raise TokenProtocolError(f"token record missing fields: {missing}")
    traj_id = int(record["traj_id"])
    road_seq = [int(value) for value in record["road_seq"]]
    if not road_seq or any(value < 0 for value in road_seq):
        raise TokenProtocolError(f"traj_id={traj_id} road_seq must be non-empty non-negative IDs")
    answer = serialize_answer(record["lens_type"], record["codes"])
    parsed = parse_answer(answer)
    code_length = int(record.get("code_length", len(parsed["codes"])))
    if code_length != len(parsed["codes"]):
        raise TokenProtocolError(
            f"traj_id={traj_id} code_length {code_length} != {len(parsed['codes'])} events"
        )
    return {
        "traj_id": traj_id,
        "lens_type": parsed["lens_type"],
        "code_length": code_length,
        "codes": parsed["codes"],
        "road_seq": road_seq,
        "answer": answer,
    }


def export_token_records(
    records: Iterable[Mapping[str, object]],
    expected_traj_ids: Iterable[int],
    output_path: str | Path,
) -> dict[str, object]:
    rows = [_normalize_record(record) for record in records]
    observed = [int(row["traj_id"]) for row in rows]
    if len(set(observed)) != len(observed):
        raise TokenProtocolError("duplicate traj_id in token records")
    expected = [int(value) for value in expected_traj_ids]
    if len(set(expected)) != len(expected):
        raise TokenProtocolError("duplicate traj_id in expected population")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))[:10]
        extra = sorted(set(observed) - set(expected))[:10]
        raise TokenProtocolError(f"population mismatch: missing={missing}, extra={extra}")
    rows.sort(key=lambda row: int(row["traj_id"]))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    sorted_ids = [int(row["traj_id"]) for row in rows]
    return {
        "population": len(rows),
        "sha256": _sha256(path),
        "traj_id_sha256": _id_hash(sorted_ids),
        "min_traj_id": min(sorted_ids) if sorted_ids else None,
        "max_traj_id": max(sorted_ids) if sorted_ids else None,
    }


@torch.no_grad()
def encode_loader(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, object]]:
    model = model.to(device)
    model.eval()
    records: list[dict[str, object]] = []
    for batch in loader:
        device_batch = {
            name: value.to(device, non_blocking=device.type == "cuda")
            for name, value in batch.items()
        }
        output = model.encode_views(device_batch)
        for row in range(len(batch["traj_id"])):
            code_length = int(output["latent_lengths"][row])
            route_length = int(batch["route_lengths"][row])
            records.append(
                {
                    "traj_id": int(batch["traj_id"][row]),
                    "lens_type": output["lens_type"][row].long().cpu().tolist(),
                    "code_length": code_length,
                    "codes": output["codes"][row, :code_length].long().cpu().tolist(),
                    "road_seq": batch["route"][row, :route_length].long().cpu().tolist(),
                }
            )
    return records
