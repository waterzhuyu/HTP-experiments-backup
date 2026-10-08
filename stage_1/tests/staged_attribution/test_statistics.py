from __future__ import annotations

import pytest

from staged_attribution.statistics import (
    aggregate_seeds,
    g_shapley,
    paired_bootstrap,
)


def test_paired_bootstrap_uses_within_trajectory_differences() -> None:
    baseline = {1: 3.0, 2: 5.0, 3: 7.0, 4: 9.0}
    candidate = {1: 2.0, 2: 4.0, 3: 6.0, 4: 8.0}

    result = paired_bootstrap(baseline, candidate, n_resamples=500, seed=42)

    assert result["n_pairs"] == 4
    assert result["estimate"] == pytest.approx(-1.0)
    assert result["ci_low"] == pytest.approx(-1.0)
    assert result["ci_high"] == pytest.approx(-1.0)
    assert result["p_improvement"] == 1.0


def test_paired_bootstrap_rejects_unpaired_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="paired population"):
        paired_bootstrap({1: 1.0}, {2: 1.0}, n_resamples=10, seed=1)
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap({1: 1.0}, {1: float("nan")}, n_resamples=10, seed=1)


def test_g_shapley_and_interaction_identities() -> None:
    result = g_shapley({"g00": 10.0, "g10": 7.0, "g01": 8.0, "g11": 4.0})

    assert result["sid"] == pytest.approx(-3.5)
    assert result["sampling"] == pytest.approx(-2.5)
    assert result["interaction"] == pytest.approx(-1.0)
    assert result["total"] == pytest.approx(-6.0)
    assert result["sid"] + result["sampling"] == pytest.approx(result["total"])
    assert result["interaction"] == pytest.approx(4.0 - 7.0 - 8.0 + 10.0)


def test_seed_aggregation_reports_sample_uncertainty() -> None:
    result = aggregate_seeds([1.0, 2.0, 3.0])

    assert result["n"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx(1.0)
    assert result["sem"] == pytest.approx(1 / 3**0.5)
    assert result["ci_low"] < 2.0 < result["ci_high"]
