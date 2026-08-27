from __future__ import annotations

import numpy as np
import pytest

from staged_attribution.population_bootstrap import (
    BOOTSTRAP_METHOD,
    build_metric_contributions,
    g_metric_attribution,
    load_metric_contributions,
    paired_metric_effects,
    paired_metric_effects_stratified_seeds,
    population_regrets,
    save_metric_contributions,
)
from staged_attribution.stage4_metrics import evaluate_population


def geometries() -> dict[int, np.ndarray]:
    return {
        0: np.asarray([[0.0, 0.0], [0.01, 0.0]]),
        1: np.asarray([[0.0, 0.01], [0.01, 0.01]]),
    }


def population():
    real_gps = {
        1: np.asarray([[0.001, 0.001], [0.004, 0.001], [0.007, 0.001]]),
        2: np.asarray([[0.001, 0.009], [0.004, 0.009], [0.007, 0.009]]),
    }
    real_roads = {1: [0, 0, 0], 2: [1, 1, 1]}
    generated = [
        {"traj_id": 1, "status": "reconstructed", "gps_traj": real_gps[1].tolist(), "road_traj": [0, 0, 0]},
        {"traj_id": 2, "status": "reconstructed", "gps_traj": real_gps[2].tolist(), "road_traj": [1, 1, 1]},
    ]
    return real_gps, real_roads, generated


def test_population_regret_point_matches_stage4_metrics_without_failures() -> None:
    real_gps, real_roads, generated = population()
    kwargs = {
        "mbr": {"min_lon": 0.0, "max_lon": 0.02, "min_lat": 0.0, "max_lat": 0.02},
        "grid": {"grid_size_m": 100.0, "meters_per_lon": 111320.0, "meters_per_lat": 110540.0, "n_grid_x": 23, "n_grid_y": 23},
    }
    exact = evaluate_population(real_gps, real_roads, generated, geometries(), "porto", **kwargs)
    contributions = build_metric_contributions(real_gps, real_roads, generated, geometries(), "porto", **kwargs)
    regrets = population_regrets(contributions)

    assert regrets["validity_rate"][0] == 0.0
    for metric, value in exact["successful_only"]["metrics"].items():
        expected = value if metric.endswith("jsd") else 1.0 - value
        assert regrets[metric][0] == pytest.approx(expected)


def test_population_regret_retains_failures_and_no_success_population() -> None:
    real_gps, real_roads, generated = population()
    kwargs = {
        "mbr": {"min_lon": 0.0, "max_lon": 0.02, "min_lat": 0.0, "max_lat": 0.02},
        "grid": {"grid_size_m": 100.0, "meters_per_lon": 111320.0, "meters_per_lat": 110540.0, "n_grid_x": 23, "n_grid_y": 23},
    }
    generated[1] = {"traj_id": 2, "status": "failed", "failure": {"reason": "syntax"}}
    half = build_metric_contributions(real_gps, real_roads, generated, geometries(), "porto", **kwargs)
    half_regret = population_regrets(half)
    assert half_regret["validity_rate"][0] == 0.5
    assert all(0.5 <= half_regret[metric][0] <= 1.0 for metric in half_regret)

    failed = [
        {"traj_id": traj_id, "status": "failed", "failure": {"reason": "syntax"}}
        for traj_id in real_gps
    ]
    none = build_metric_contributions(real_gps, real_roads, failed, geometries(), "porto", **kwargs)
    none_regret = population_regrets(none)
    assert all(values[0] == 1.0 for values in none_regret.values())


def test_paired_effects_use_same_trajectory_bootstrap_and_are_deterministic() -> None:
    real_gps, real_roads, generated = population()
    kwargs = {
        "mbr": {"min_lon": 0.0, "max_lon": 0.02, "min_lat": 0.0, "max_lat": 0.02},
        "grid": {"grid_size_m": 100.0, "meters_per_lon": 111320.0, "meters_per_lat": 110540.0, "n_grid_x": 23, "n_grid_y": 23},
    }
    baseline = build_metric_contributions(real_gps, real_roads, generated, geometries(), "porto", **kwargs)
    candidate_records = list(generated)
    candidate_records[1] = {"traj_id": 2, "status": "failed", "failure": {"reason": "syntax"}}
    candidate = build_metric_contributions(real_gps, real_roads, candidate_records, geometries(), "porto", **kwargs)
    first = paired_metric_effects(baseline, candidate, n_resamples=100, seed=42, chunk_size=11)
    second = paired_metric_effects(baseline, candidate, n_resamples=100, seed=42, chunk_size=17)

    assert first == second
    assert len(first) == 9
    assert all(row["method"] == BOOTSTRAP_METHOD for row in first)
    assert all(row["estimate"] >= 0 for row in first)
    assert next(row for row in first if row["metric"] == "validity_rate")["estimate"] == 0.5

    reordered = build_metric_contributions(
        {2: real_gps[2], 1: real_gps[1]}, real_roads, generated, geometries(), "porto", **kwargs
    )
    assert paired_metric_effects(reordered, baseline, n_resamples=10, seed=1)


def test_bootstrap_rejects_unpaired_population() -> None:
    real_gps, real_roads, generated = population()
    kwargs = {
        "mbr": {"min_lon": 0.0, "max_lon": 0.02, "min_lat": 0.0, "max_lat": 0.02},
        "grid": {"grid_size_m": 100.0, "meters_per_lon": 111320.0, "meters_per_lat": 110540.0, "n_grid_x": 23, "n_grid_y": 23},
    }
    baseline = build_metric_contributions(real_gps, real_roads, generated, geometries(), "porto", **kwargs)
    subset = build_metric_contributions(
        {1: real_gps[1]}, {1: real_roads[1]}, generated[:1], geometries(), "porto", **kwargs
    )
    with pytest.raises(ValueError, match="paired populations"):
        paired_metric_effects(baseline, subset, n_resamples=10, seed=1)


def test_contribution_artifacts_round_trip_and_reject_tampering(tmp_path) -> None:
    real_gps, real_roads, generated = population()
    kwargs = {
        "mbr": {"min_lon": 0.0, "max_lon": 0.02, "min_lat": 0.0, "max_lat": 0.02},
        "grid": {"grid_size_m": 100.0, "meters_per_lon": 111320.0, "meters_per_lat": 110540.0, "n_grid_x": 23, "n_grid_y": 23},
    }
    original = build_metric_contributions(
        real_gps, real_roads, generated, geometries(), "porto", **kwargs
    )
    manifest = save_metric_contributions(original, tmp_path / "components")
    restored = load_metric_contributions(manifest)

    restored_regrets = population_regrets(restored)
    for metric, values in population_regrets(original).items():
        assert restored_regrets[metric] == pytest.approx(values)
    (manifest.parent / "histograms.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        load_metric_contributions(manifest)


def test_selection_effect_resamples_within_seed_and_equal_weights_seed_effects() -> None:
    real_gps, real_roads, generated = population()
    kwargs = {
        "mbr": {"min_lon": 0.0, "max_lon": 0.02, "min_lat": 0.0, "max_lat": 0.02},
        "grid": {"grid_size_m": 100.0, "meters_per_lon": 111320.0, "meters_per_lat": 110540.0, "n_grid_x": 23, "n_grid_y": 23},
    }
    baseline = build_metric_contributions(
        real_gps, real_roads, generated, geometries(), "porto", **kwargs
    )
    failed = list(generated)
    failed[1] = {"traj_id": 2, "status": "failed", "failure": {"reason": "syntax"}}
    candidate = build_metric_contributions(
        real_gps, real_roads, failed, geometries(), "porto", **kwargs
    )
    first = paired_metric_effects_stratified_seeds(
        {42: baseline, 314159: baseline},
        {42: candidate, 314159: baseline},
        n_resamples=100,
        seed=20260826,
        chunk_size=13,
    )
    second = paired_metric_effects_stratified_seeds(
        {314159: baseline, 42: baseline},
        {314159: baseline, 42: candidate},
        n_resamples=100,
        seed=20260826,
        chunk_size=17,
    )

    assert first == second
    validity = next(row for row in first if row["metric"] == "validity_rate")
    assert validity["estimate"] == pytest.approx(0.25)
    assert validity["n_pairs"] == 4
    assert validity["seed_strata"] == [42, 314159]

    with pytest.raises(ValueError, match="seed strata differ"):
        paired_metric_effects_stratified_seeds(
            {42: baseline}, {314159: candidate}, n_resamples=10, seed=1
        )


def test_g_attribution_uses_one_population_and_satisfies_shapley_identity() -> None:
    real_gps, real_roads, generated = population()
    kwargs = {
        "mbr": {"min_lon": 0.0, "max_lon": 0.02, "min_lat": 0.0, "max_lat": 0.02},
        "grid": {"grid_size_m": 100.0, "meters_per_lon": 111320.0, "meters_per_lat": 110540.0, "n_grid_x": 23, "n_grid_y": 23},
    }
    perfect = build_metric_contributions(
        real_gps, real_roads, generated, geometries(), "porto", **kwargs
    )
    one_failure = list(generated)
    one_failure[1] = {"traj_id": 2, "status": "failed", "failure": {"reason": "syntax"}}
    degraded = build_metric_contributions(
        real_gps, real_roads, one_failure, geometries(), "porto", **kwargs
    )
    rows = g_metric_attribution(
        {"g00": perfect, "g10": degraded, "g01": perfect, "g11": degraded},
        n_resamples=100,
        seed=42,
        chunk_size=13,
    )

    assert len(rows) == 9 * 4
    for metric in {row["metric"] for row in rows}:
        values = {row["component"]: row["estimate"] for row in rows if row["metric"] == metric}
        assert values["sid"] + values["sampling"] == pytest.approx(values["total"])
        assert values["interaction"] == pytest.approx(0.0)
        assert values["sampling"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="exactly"):
        g_metric_attribution({"g00": perfect}, n_resamples=10, seed=1)
