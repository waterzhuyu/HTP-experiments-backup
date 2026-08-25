from __future__ import annotations

from collections.abc import Mapping, Sequence

from .token_protocol import TokenProtocolError, serialize_answer


class GAttributionError(ValueError):
    """Raised when paired G-stage counterfactuals cannot be constructed exactly."""


def _unique(records: Sequence[Mapping[str, object]], label: str) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for record in records:
        traj_id = int(record["traj_id"])
        if traj_id in result:
            raise GAttributionError(f"duplicate {label} traj_id={traj_id}")
        result[traj_id] = record
    return result


def _validate_oracle(record: Mapping[str, object]) -> None:
    road_seq = [int(value) for value in record.get("road_seq", [])]
    if not road_seq or any(value < 0 for value in road_seq):
        raise GAttributionError("oracle road_seq must be non-empty non-negative IDs")
    try:
        serialize_answer(record["lens_type"], record["codes"])
    except (KeyError, TokenProtocolError) as error:
        raise GAttributionError(f"invalid oracle SID: {error}") from error


def _ready(
    traj_id: int,
    variant: str,
    oracle: Mapping[str, object],
    generated: Mapping[str, object],
    sid_source: str,
    sampling_source: str,
) -> dict[str, object]:
    sid_record = generated if sid_source == "generated" else oracle
    sampling_record = generated if sampling_source == "generated" else oracle
    return {
        "traj_id": traj_id,
        "variant": variant,
        "status": "ready",
        "codes": [[int(value) for value in row] for row in sid_record["codes"]],
        "lens_type": [int(value) for value in sampling_record["lens_type"]],
        "road_seq": [int(value) for value in oracle["road_seq"]],
        "sid_source": sid_source,
        "sampling_source": sampling_source,
        "road_source": "oracle_prompt",
        "failure": None,
    }


def _failed(
    traj_id: int,
    variant: str,
    oracle: Mapping[str, object],
    generated: Mapping[str, object],
    sid_source: str,
    sampling_source: str,
) -> dict[str, object]:
    reason = str(generated.get("failure", {}).get("reason", "unknown"))
    message = str(generated.get("failure", {}).get("message", "generation failed"))
    return {
        "traj_id": traj_id,
        "variant": variant,
        "status": "failed",
        "road_seq": [int(value) for value in oracle["road_seq"]],
        "sid_source": sid_source,
        "sampling_source": sampling_source,
        "road_source": "oracle_prompt",
        "failure": {"reason": f"generation_{reason}", "message": message},
    }


def build_g_variants(
    oracle_records: Sequence[Mapping[str, object]],
    generated_records: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    oracle_by_id = _unique(oracle_records, "oracle")
    generated_by_id = _unique(generated_records, "generated")
    if set(oracle_by_id) != set(generated_by_id):
        raise GAttributionError(
            f"oracle/generated population mismatch: "
            f"oracle_only={sorted(set(oracle_by_id)-set(generated_by_id))[:10]}, "
            f"generated_only={sorted(set(generated_by_id)-set(oracle_by_id))[:10]}"
        )
    definitions = {
        "g00": ("oracle", "oracle"),
        "g10": ("generated", "oracle"),
        "g01": ("oracle", "generated"),
        "g11": ("generated", "generated"),
    }
    result = {name: [] for name in definitions}
    for traj_id in sorted(oracle_by_id):
        oracle = oracle_by_id[traj_id]
        generated = generated_by_id[traj_id]
        _validate_oracle(oracle)
        generated_ok = generated.get("status") == "completed"
        if generated_ok:
            try:
                serialize_answer(generated["lens_type"], generated["codes"])
            except (KeyError, TokenProtocolError) as error:
                raise GAttributionError(f"completed generation has invalid SID: {error}") from error
        for variant, (sid_source, sampling_source) in definitions.items():
            needs_generated = sid_source == "generated" or sampling_source == "generated"
            if needs_generated and not generated_ok:
                row = _failed(
                    traj_id, variant, oracle, generated, sid_source, sampling_source
                )
            else:
                row = _ready(
                    traj_id, variant, oracle, generated, sid_source, sampling_source
                )
            result[variant].append(row)
    return result
