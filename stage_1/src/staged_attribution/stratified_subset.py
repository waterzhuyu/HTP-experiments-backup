from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


ALGORITHM = "lens3_hamilton_sha256_v1"
REGISTERED_STRATA = tuple(format(value, "03b") for value in range(8))


class StratifiedSubsetError(ValueError):
    """Raised when a frozen subset cannot be reproduced from aligned test evidence."""


def lens_type_bits(length: int) -> tuple[int, int, int]:
    """Return the exact three parity bits emitted by the HTP Encoder."""
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise StratifiedSubsetError("trajectory length must be a positive integer")
    current = length
    bits: list[int] = []
    for index in range(3):
        bits.append(current % 2)
        if index < 2:
            current = (current + 1) // 2
    return tuple(bits)  # type: ignore[return-value]


def lens_type_key(length: int) -> str:
    return "".join(map(str, lens_type_bits(length)))


def trajectory_id_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def hamilton_quotas(counts: Mapping[str, int], size: int) -> dict[str, int]:
    if not counts or any(not isinstance(value, int) or value < 0 for value in counts.values()):
        raise StratifiedSubsetError("stratum populations must be non-negative integers")
    population = sum(counts.values())
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= population:
        raise StratifiedSubsetError("subset size must be in [1,population]")
    exact = {key: size * count / population for key, count in counts.items()}
    quotas = {key: int(value // 1) for key, value in exact.items()}
    remainder = size - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    if sum(quotas.values()) != size or any(quotas[key] > counts[key] for key in counts):
        raise AssertionError("Hamilton allocation violated population constraints")
    return {key: quotas[key] for key in sorted(quotas)}


def stratified_subset(
    traj_ids: Iterable[int],
    lengths: Iterable[int],
    size: int,
    seed: int,
    *,
    require_all_registered_strata: bool = True,
) -> dict[str, object]:
    ids = [int(value) for value in traj_ids]
    point_lengths = [int(value) for value in lengths]
    if len(ids) != len(point_lengths):
        raise StratifiedSubsetError("trajectory IDs and lengths must align")
    if len(set(ids)) != len(ids):
        raise StratifiedSubsetError("subset population contains duplicate trajectory IDs")
    if any(value <= 0 for value in point_lengths):
        raise StratifiedSubsetError("trajectory lengths must be positive")
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for traj_id, length in zip(ids, point_lengths, strict=True):
        by_stratum[lens_type_key(length)].append(traj_id)
    if require_all_registered_strata and set(by_stratum) != set(REGISTERED_STRATA):
        raise StratifiedSubsetError("population must contain all eight registered lens strata")
    counts = {key: len(values) for key, values in sorted(by_stratum.items())}
    quotas = hamilton_quotas(counts, size)
    selected_by_stratum: dict[str, list[int]] = {}
    for key, values in sorted(by_stratum.items()):
        ranked = sorted(
            values,
            key=lambda value: (
                hashlib.sha256(f"{int(seed)}:{key}:{value}".encode()).digest(),
                value,
            ),
        )
        selected_by_stratum[key] = sorted(ranked[: quotas[key]])
    selected = sorted(value for values in selected_by_stratum.values() for value in values)
    if len(selected) != size or len(set(selected)) != size:
        raise AssertionError("selected subset is not an exact unique population")
    return {
        "algorithm": ALGORITHM,
        "seed": int(seed),
        "sample_size": int(size),
        "selected_traj_ids": selected,
        "selected_traj_id_sha256": trajectory_id_sha256(selected),
        "strata": {
            key: {
                "population": counts[key],
                "quota": quotas[key],
                "selected_traj_ids": selected_by_stratum[key],
                "selected_traj_id_sha256": trajectory_id_sha256(selected_by_stratum[key]),
            }
            for key in sorted(counts)
        },
    }


def verify_token_lens_mapping(
    token_records: Iterable[Mapping[str, object]],
    lengths_by_traj_id: Mapping[int, int],
) -> dict[str, object]:
    observed: dict[int, tuple[int, int, int]] = {}
    for record in token_records:
        traj_id = int(record["traj_id"])
        if traj_id in observed:
            raise StratifiedSubsetError(f"duplicate token trajectory ID {traj_id}")
        raw = record.get("lens_type")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise StratifiedSubsetError(f"trajectory {traj_id} has invalid lens_type")
        bits = tuple(int(value) for value in raw)
        if len(bits) != 3 or any(value not in (0, 1) for value in bits):
            raise StratifiedSubsetError(f"trajectory {traj_id} has invalid lens_type")
        observed[traj_id] = bits  # type: ignore[assignment]
    authority = {int(key): int(value) for key, value in lengths_by_traj_id.items()}
    if set(observed) != set(authority):
        raise StratifiedSubsetError("token and length-authority populations differ")
    mismatches = [
        traj_id for traj_id in sorted(observed)
        if observed[traj_id] != lens_type_bits(authority[traj_id])
    ]
    if mismatches:
        raise StratifiedSubsetError(f"lens_type mismatch for trajectory IDs {mismatches[:10]}")
    ids = sorted(observed)
    return {
        "population": len(ids),
        "traj_id_sha256": trajectory_id_sha256(ids),
        "formula_match": True,
    }
