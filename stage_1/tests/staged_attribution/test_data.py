from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from staged_attribution.data import (
    AlignedHTPDataset,
    AlignmentError,
    assert_split_disjoint,
    collate_aligned,
)


def record(traj_id: int, length: int, route: list[int]) -> dict[str, object]:
    return {
        "traj_id": traj_id,
        "gps_seq": np.zeros((length, 2), dtype=np.float32),
        "time_seq": np.arange(length, dtype=np.float32) * 15.0,
        "dxy": np.column_stack(
            (np.arange(length, dtype=np.float32), -np.arange(length, dtype=np.float32))
        ),
        "percent_dist": np.linspace(0.0, 1.0, length, dtype=np.float32),
        "path": np.asarray(route, dtype=np.int64),
    }


def write_fixture(root: Path) -> tuple[Path, Path]:
    aligned = root / "aligned"
    aligned.mkdir()
    metadata = {
        "city": "fixture",
        "num_grids": 8,
        "num_roads": 6,
        "view_definition": {
            "gps": "normalized longitude/latitude per observation",
            "grid": "100m grid ID per observation",
            "road": "pointwise road ID per observation",
        },
    }
    (aligned / "metadata.json").write_text(json.dumps(metadata))

    # Deliberately store aligned rows in 20,10 order and processed records in 10,20 order.
    np.save(aligned / "train_traj_ids.npy", np.asarray([20, 10], dtype=np.int64))
    np.save(aligned / "train_offsets.npy", np.asarray([0, 3, 5], dtype=np.int64))
    np.save(
        aligned / "train_gps.npy",
        np.asarray([[0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.8, 0.8], [0.9, 0.9]], dtype=np.float32),
    )
    np.save(aligned / "train_grid.npy", np.asarray([1, 1, 2, 6, 7], dtype=np.int32))
    np.save(aligned / "train_road.npy", np.asarray([3, 3, 4, 0, 1], dtype=np.int32))

    processed = root / "processed.pkl"
    with processed.open("wb") as handle:
        pickle.dump([record(10, 2, [0, 2, 4, 5]), record(20, 3, [3, 4])], handle)
    return aligned, processed


def test_joins_by_traj_id_and_allows_route_length_difference(tmp_path: Path) -> None:
    aligned, processed = write_fixture(tmp_path)

    dataset = AlignedHTPDataset(aligned, processed, "train")

    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]
    assert first["traj_id"] == 20
    assert first["gps"].shape == (3, 2)
    assert first["route"].tolist() == [3, 4]
    assert second["traj_id"] == 10
    assert second["gps"].shape == (2, 2)
    assert second["route"].tolist() == [0, 2, 4, 5]
    assert len(second["route"]) != len(second["gps"])
    assert second["point_road"].tolist() == [0, 1]


def test_collate_preserves_raw_ids_and_masks_padding(tmp_path: Path) -> None:
    aligned, processed = write_fixture(tmp_path)
    dataset = AlignedHTPDataset(aligned, processed, "train")

    batch = collate_aligned([dataset[0], dataset[1]])

    assert batch["gps"].shape == (2, 3, 2)
    assert batch["lengths"].tolist() == [3, 2]
    assert batch["route_lengths"].tolist() == [2, 4]
    assert batch["point_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["grid"][1].tolist() == [6, 7, -1]
    assert batch["point_road"][1].tolist() == [0, 1, -1]
    assert batch["route"][0].tolist() == [3, 4, -1, -1]
    assert torch.equal(batch["traj_id"], torch.tensor([20, 10]))


def test_rejects_point_target_length_mismatch(tmp_path: Path) -> None:
    aligned, processed = write_fixture(tmp_path)
    with processed.open("rb") as handle:
        records = pickle.load(handle)
    records[0]["dxy"] = records[0]["dxy"][:-1]
    with processed.open("wb") as handle:
        pickle.dump(records, handle)

    with pytest.raises(AlignmentError, match="dxy length"):
        AlignedHTPDataset(aligned, processed, "train")


def test_rejects_bad_offsets(tmp_path: Path) -> None:
    aligned, processed = write_fixture(tmp_path)
    np.save(aligned / "train_offsets.npy", np.asarray([0, 3, 9], dtype=np.int64))

    with pytest.raises(AlignmentError, match="final offset"):
        AlignedHTPDataset(aligned, processed, "train")


@pytest.mark.parametrize(
    ("filename", "values", "message"),
    [
        ("train_grid.npy", [1, 1, 2, 6, 8], "grid ID"),
        ("train_road.npy", [3, 3, 4, 0, 6], "road ID"),
    ],
)
def test_rejects_out_of_range_discrete_ids(
    tmp_path: Path,
    filename: str,
    values: list[int],
    message: str,
) -> None:
    aligned, processed = write_fixture(tmp_path)
    np.save(aligned / filename, np.asarray(values, dtype=np.int32))

    with pytest.raises(AlignmentError, match=message):
        AlignedHTPDataset(aligned, processed, "train")


def test_rejects_missing_or_duplicate_processed_ids(tmp_path: Path) -> None:
    aligned, processed = write_fixture(tmp_path)
    with processed.open("rb") as handle:
        records = pickle.load(handle)

    with processed.open("wb") as handle:
        pickle.dump([records[0], records[0]], handle)
    with pytest.raises(AlignmentError, match="duplicate processed traj_id"):
        AlignedHTPDataset(aligned, processed, "train")

    with processed.open("wb") as handle:
        pickle.dump([records[0]], handle)
    with pytest.raises(AlignmentError, match="population mismatch"):
        AlignedHTPDataset(aligned, processed, "train")


def test_rejects_train_test_traj_id_leakage(tmp_path: Path) -> None:
    aligned, _ = write_fixture(tmp_path)
    np.save(aligned / "test_traj_ids.npy", np.asarray([30, 10], dtype=np.int64))

    with pytest.raises(AlignmentError, match="split leakage"):
        assert_split_disjoint(aligned)


def test_accepts_disjoint_train_test_ids(tmp_path: Path) -> None:
    aligned, _ = write_fixture(tmp_path)
    np.save(aligned / "test_traj_ids.npy", np.asarray([30, 40], dtype=np.int64))

    assert assert_split_disjoint(aligned) == {"train": 2, "test": 2, "overlap": 0}
