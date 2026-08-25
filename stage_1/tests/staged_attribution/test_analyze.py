from __future__ import annotations

import pytest

from staged_attribution.analyze import (
    city_heterogeneity,
    e_contrasts,
    pareto_front,
    q_incremental_contrasts,
    select_interaction_cells,
)


def test_e_and_q_contrasts_follow_frozen_cell_structure() -> None:
    metrics = {
        "e0_q0": 10.0,
        "e1_q0": 9.0,
        "e2_q0": 8.0,
        "e3_q0": 7.0,
        "e4_q0": 6.0,
        "e5_q0": 5.0,
        "e6_q0": 4.0,
        "e6_q1": 3.5,
        "e6_q2": 3.0,
        "e6_q3": 2.0,
    }

    e = e_contrasts(metrics)
    q = q_incremental_contrasts(metrics)

    assert e["e6_q0_minus_e0_q0"] == -6.0
    assert e["gps_road_interaction"] == pytest.approx(7.0 - 9.0 - 10.0 + 10.0)
    assert e["gps_grid_interaction"] == pytest.approx(6.0 - 8.0 - 10.0 + 10.0)
    assert q == {
        "q1_minus_q0": -0.5,
        "q2_minus_q1": -0.5,
        "q3_minus_q2": -1.0,
        "q3_minus_q0": -2.0,
    }


def test_pareto_front_respects_metric_orientation() -> None:
    candidates = {
        "a": {"error": 2.0, "validity": 0.9},
        "b": {"error": 1.0, "validity": 0.8},
        "c": {"error": 1.0, "validity": 0.95},
        "d": {"error": 3.0, "validity": 0.7},
    }

    assert pareto_front(candidates, minimize=["error"], maximize=["validity"]) == ["c"]


def test_adaptive_interactions_and_city_heterogeneity_are_explicit() -> None:
    interactions = {
        "gps_road": {"estimate": -0.2, "ci_low": -0.4, "ci_high": -0.1},
        "gps_grid": {"estimate": 0.01, "ci_low": -0.1, "ci_high": 0.1},
        "road_grid": {"estimate": 0.3, "ci_low": 0.1, "ci_high": 0.5},
    }

    assert select_interaction_cells(interactions, threshold=0.15) == ["gps_road", "road_grid"]
    heterogeneity = city_heterogeneity({"porto": -0.4, "chengdu": 0.1})
    assert heterogeneity["range"] == pytest.approx(0.5)
    assert heterogeneity["sign_heterogeneous"] is True
    assert heterogeneity["city_values"] == {"porto": -0.4, "chengdu": 0.1}
