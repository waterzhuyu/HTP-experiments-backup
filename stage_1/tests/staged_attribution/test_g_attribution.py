from __future__ import annotations

import pytest

from staged_attribution.g_attribution import GAttributionError, build_g_variants


def oracle(traj_id: int = 4) -> dict[str, object]:
    return {
        "traj_id": traj_id,
        "lens_type": [0, 1, 0],
        "codes": [[1, 2, 3, 4], [2, 3, 4, 5]],
        "road_seq": [7, 8],
    }


def generated(traj_id: int = 4) -> dict[str, object]:
    return {
        "traj_id": traj_id,
        "status": "completed",
        "lens_type": [1, 0, 1],
        "codes": [[6, 7, 8, 9]],
        "raw_continuation": "x",
    }


def test_four_counterfactuals_swap_only_sid_and_sampling() -> None:
    variants = build_g_variants([oracle()], [generated()])

    assert set(variants) == {"g00", "g10", "g01", "g11"}
    rows = {name: values[0] for name, values in variants.items()}
    assert all(row["road_seq"] == [7, 8] for row in rows.values())
    assert rows["g00"]["codes"] == oracle()["codes"]
    assert rows["g00"]["lens_type"] == oracle()["lens_type"]
    assert rows["g10"]["codes"] == generated()["codes"]
    assert rows["g10"]["lens_type"] == oracle()["lens_type"]
    assert rows["g01"]["codes"] == oracle()["codes"]
    assert rows["g01"]["lens_type"] == generated()["lens_type"]
    assert rows["g11"]["codes"] == generated()["codes"]
    assert rows["g11"]["lens_type"] == generated()["lens_type"]
    assert rows["g10"]["sid_source"] == "generated"
    assert rows["g01"]["sampling_source"] == "generated"


def test_failed_generation_is_retained_in_every_affected_variant() -> None:
    failure = {
        "traj_id": 4,
        "status": "failed",
        "failure": {"reason": "truncated", "message": "x"},
        "raw_continuation": "bad",
    }

    variants = build_g_variants([oracle()], [failure])

    assert variants["g00"][0]["status"] == "ready"
    assert variants["g10"][0]["status"] == "failed"
    assert variants["g01"][0]["status"] == "failed"
    assert variants["g11"][0]["status"] == "failed"
    assert all(len(rows) == 1 for rows in variants.values())
    assert variants["g11"][0]["failure"]["reason"] == "generation_truncated"


def test_rejects_population_or_oracle_route_errors() -> None:
    with pytest.raises(GAttributionError, match="population"):
        build_g_variants([oracle()], [generated(5)])
    with pytest.raises(GAttributionError, match="road_seq"):
        build_g_variants([dict(oracle(), road_seq=[])], [generated()])
