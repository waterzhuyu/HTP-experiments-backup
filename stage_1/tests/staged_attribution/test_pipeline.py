from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from staged_attribution.pipeline import (
    PipelineBackend,
    meters_per_normalized_coordinate,
    select_dataset_indices,
)


CONFIG_PATH = Path("stage_1/configs/staged_attribution/experiment.json")


def test_subset_selection_is_id_based_deterministic_and_sorted() -> None:
    ids = np.asarray([20, 10, 40, 30, 50])

    first = select_dataset_indices(ids, 3, seed=42, salt="train")
    second = select_dataset_indices(ids, 3, seed=42, salt="train")
    changed = select_dataset_indices(ids, 3, seed=43, salt="train")

    assert first.tolist() == second.tolist()
    assert first.tolist() == sorted(first.tolist())
    assert len(set(first.tolist())) == 3
    assert first.tolist() != changed.tolist()
    assert select_dataset_indices(ids, "full", seed=1, salt="x").tolist() == [0, 1, 2, 3, 4]


def test_subset_selection_rejects_impossible_population() -> None:
    with pytest.raises(RuntimeError, match="invalid requested population"):
        select_dataset_indices([1, 2], 3, seed=1, salt="x")


def test_normalized_coordinate_scale_uses_mbr_spans() -> None:
    metadata = {
        "mbr": {"min_lon": -8.69, "max_lon": -8.55, "min_lat": 41.13, "max_lat": 41.19},
        "grid": {"meters_per_lon": 83810.0, "meters_per_lat": 110540.0},
    }

    result = meters_per_normalized_coordinate(metadata)

    assert result[0] == pytest.approx(0.14 * 83810.0)
    assert result[1] == pytest.approx(0.06 * 110540.0)


def test_pipeline_backend_accepts_registered_noncore_adaptive_cell() -> None:
    backend = PipelineBackend(
        CONFIG_PATH,
        {
            "city": "porto",
            "seed": 42,
            "cell_id": "e2_q3",
            "budget": "controlled",
            "device": "cpu",
        },
    )

    assert backend.cell.cell_id == "e2_q3"
    assert backend.cell.views == frozenset({"grid"})
    assert backend.cell.q_weights["auxiliary"] > 0
