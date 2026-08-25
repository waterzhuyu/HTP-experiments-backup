from __future__ import annotations

import numpy as np
import pytest

from staged_attribution.stage4_metrics import (
    MetricPopulationError,
    evaluate_population,
    htp_aggregate_metrics,
    trajectory_sufficient_statistics,
)


def trajectories():
    gps = {
        1: np.asarray([[-8.60, 41.15], [-8.59, 41.15]]),
        2: np.asarray([[-8.61, 41.16], [-8.60, 41.16], [-8.59, 41.16]]),
    }
    roads = {1: [0, 0], 2: [1, 1, 1]}
    geometries = {
        0: np.asarray([[-8.60, 41.15], [-8.59, 41.15]]),
        1: np.asarray([[-8.61, 41.16], [-8.59, 41.16]]),
    }
    return gps, roads, geometries


def test_identical_populations_match_original_metric_orientations() -> None:
    gps, roads, geometries = trajectories()

    result = htp_aggregate_metrics(gps, gps, roads, roads, geometries, city="porto", top_k=2)

    assert result["travel_distance_jsd"] == pytest.approx(0.0)
    assert result["segment_distance_jsd"] == pytest.approx(0.0)
    assert result["radius_jsd"] == pytest.approx(0.0)
    assert result["grid_density_cosine"] == pytest.approx(1.0)
    assert result["grid_pattern_f1"] == pytest.approx(1.0)
    assert result["road_density_cosine"] == pytest.approx(1.0)
    assert result["road_pattern_f1"] == pytest.approx(1.0)
    assert result["point_to_road_jsd"] == pytest.approx(0.0)


def test_sufficient_statistics_are_per_trajectory_and_finite() -> None:
    gps, roads, geometries = trajectories()

    statistics = trajectory_sufficient_statistics(gps[1], roads[1], geometries, city="porto")

    assert statistics["points"] == 2
    assert statistics["travel_distance_km"] > 0
    assert statistics["mean_segment_distance_km"] == pytest.approx(statistics["travel_distance_km"])
    assert statistics["radius_km"] > 0
    assert statistics["mean_point_to_road_km"] < 1e-9
    assert statistics["unique_road_count"] == 1
    assert statistics["unique_grid_count"] >= 1


def test_population_aware_evaluation_keeps_failed_ids() -> None:
    gps, roads, geometries = trajectories()
    generated = [
        {"traj_id": 1, "status": "reconstructed", "gps_traj": gps[1].tolist(), "road_traj": roads[1]},
        {"traj_id": 2, "status": "failed", "failure": {"reason": "malformed"}},
    ]

    result = evaluate_population(gps, roads, generated, geometries, city="porto", invalid_penalty=2.0)

    assert result["population"] == 2
    assert result["successful"] == 1
    assert result["failed"] == 1
    assert result["validity_rate"] == 0.5
    assert result["failure_reasons"] == {"malformed": 1}
    assert len(result["per_trajectory"]) == 2
    failed = next(row for row in result["per_trajectory"] if row["traj_id"] == 2)
    assert failed["valid"] is False
    assert failed["penalized_error"] == 2.0
    assert result["population_mean_penalized_error"] >= 1.0
    assert result["successful_only"]["population"] == 1


def test_population_mismatch_is_rejected_instead_of_dropped() -> None:
    gps, roads, geometries = trajectories()
    with pytest.raises(MetricPopulationError, match="population"):
        evaluate_population(gps, roads, [], geometries, city="porto")
