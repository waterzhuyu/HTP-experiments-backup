from __future__ import annotations

import numpy as np
import pytest

from staged_attribution.pipeline import (
    meters_per_normalized_coordinate,
    select_dataset_indices,
)


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
