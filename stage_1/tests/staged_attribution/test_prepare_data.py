from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from staged_attribution.prepare_data import (
    DataPreparationError,
    grid_spec,
    prepare_from_frame,
)


MBR = {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 0.01, "max_lat": 0.01}


def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "traj_id": [100, 200, 300],
            "geometry": [
                [[0.0, 0.0], [0.001, 0.001]],
                [[0.002, 0.002], [0.003, 0.003], [0.004, 0.004]],
                [[0.009, 0.009], [0.01, 0.01]],
            ],
            "opath": [[0, 1], [2, 2, 3], [4, 4]],
        }
    )


def test_prepare_from_frame_writes_exact_aligned_population(tmp_path: Path) -> None:
    output = tmp_path / "aligned"

    metadata = prepare_from_frame(
        tiny_frame(),
        train_indices=np.asarray([1, 0], dtype=np.int64),
        test_indices=np.asarray([2], dtype=np.int64),
        output_dir=output,
        city="fixture",
        mbr=MBR,
        num_roads=5,
        grid_size_m=100.0,
        source_manifest={"source": "fixture", "source_sha256": "abc"},
    )

    assert np.load(output / "train_traj_ids.npy").tolist() == [200, 100]
    assert np.load(output / "train_offsets.npy").tolist() == [0, 3, 5]
    assert np.load(output / "train_road.npy").tolist() == [2, 2, 3, 0, 1]
    gps = np.load(output / "train_gps.npy")
    assert np.allclose(gps[0], [0.2, 0.2])
    assert np.allclose(gps[-1], [0.1, 0.1])
    assert metadata["train"]["trajectories"] == 2
    assert metadata["test"]["trajectories"] == 1
    assert metadata["grid"]["grid_size_m"] == 100.0
    assert set(metadata["output_sha256"]) == {
        "train_gps.npy",
        "train_grid.npy",
        "train_road.npy",
        "train_offsets.npy",
        "train_traj_ids.npy",
        "test_gps.npy",
        "test_grid.npy",
        "test_road.npy",
        "test_offsets.npy",
        "test_traj_ids.npy",
    }
    assert all(len(value) == 64 for value in metadata["output_sha256"].values())


def test_prepare_rejects_overlapping_split_indices(tmp_path: Path) -> None:
    with pytest.raises(DataPreparationError, match="indices overlap"):
        prepare_from_frame(
            tiny_frame(),
            train_indices=np.asarray([0, 1], dtype=np.int64),
            test_indices=np.asarray([1, 2], dtype=np.int64),
            output_dir=tmp_path / "aligned",
            city="fixture",
            mbr=MBR,
            num_roads=5,
        )


def test_prepare_rejects_unaligned_pointwise_road(tmp_path: Path) -> None:
    frame = tiny_frame()
    frame.at[0, "opath"] = [0]

    with pytest.raises(DataPreparationError, match="unaligned views"):
        prepare_from_frame(
            frame,
            train_indices=np.asarray([0, 1], dtype=np.int64),
            test_indices=np.asarray([2], dtype=np.int64),
            output_dir=tmp_path / "aligned",
            city="fixture",
            mbr=MBR,
            num_roads=5,
        )


def test_grid_spec_has_positive_fixed_support() -> None:
    spec = grid_spec(MBR, 100.0)

    assert spec["n_grid_x"] > 0
    assert spec["n_grid_y"] > 0
    assert spec["grid_size_m"] == 100.0
