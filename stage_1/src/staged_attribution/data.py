from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import shapely
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class AlignmentError(ValueError):
    """Raised when point-aligned views and HTP targets do not describe the same trips."""


@dataclass(frozen=True)
class RoadContext:
    """Normalized route geometry and target scaling shared by every E/Q cell."""

    geometry: torch.Tensor
    geometry_lengths: torch.Tensor
    z_score: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.geometry.ndim != 3 or self.geometry.shape[-1] != 2:
            raise AlignmentError("road geometry must have shape [roads,points,2]")
        if self.geometry_lengths.shape != (len(self.geometry),):
            raise AlignmentError("road geometry lengths must align with roads")
        if torch.any(self.geometry_lengths < 2) or torch.any(
            self.geometry_lengths > self.geometry.shape[1]
        ):
            raise AlignmentError("road geometry lengths are outside padded support")
        required = {
            "mean_dx", "std_dx", "mean_dy", "std_dy",
            "mean_percent_dist", "std_percent_dist",
        }
        if set(self.z_score) != required:
            raise AlignmentError(f"z_score keys must be exactly {sorted(required)}")
        if float(self.z_score["std_dx"]) <= 0 or float(self.z_score["std_dy"]) <= 0 or float(
            self.z_score["std_percent_dist"]
        ) <= 0:
            raise AlignmentError("z_score standard deviations must be positive")


def load_road_context(
    edge_info_path: str | Path,
    mbr: Mapping[str, float],
    z_score_path: str | Path,
) -> RoadContext:
    roads = pd.read_csv(edge_info_path)
    if "geometry" not in roads:
        raise AlignmentError("edge_info must contain geometry")
    lon_span = float(mbr["max_lon"]) - float(mbr["min_lon"])
    lat_span = float(mbr["max_lat"]) - float(mbr["min_lat"])
    if lon_span <= 0 or lat_span <= 0:
        raise AlignmentError("MBR spans must be positive")
    geometries = []
    for value in roads["geometry"]:
        coordinates = np.asarray(list(shapely.from_wkt(value).coords), dtype=np.float32)
        coordinates[:, 0] = (coordinates[:, 0] - float(mbr["min_lon"])) / lon_span
        coordinates[:, 1] = (coordinates[:, 1] - float(mbr["min_lat"])) / lat_span
        geometries.append(torch.as_tensor(coordinates, dtype=torch.float32))
    lengths = torch.tensor([len(value) for value in geometries], dtype=torch.long)
    padded = pad_sequence(geometries, batch_first=True, padding_value=0.0)
    with Path(z_score_path).open(encoding="utf-8") as handle:
        z_score = {name: float(value) for name, value in json.load(handle).items()}
    return RoadContext(padded, lengths, z_score)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _require_unique(values: np.ndarray, label: str) -> None:
    if len(np.unique(values)) != len(values):
        raise AlignmentError(f"duplicate {label}")


def assert_split_disjoint(aligned_root: str | Path) -> dict[str, int]:
    root = Path(aligned_root)
    train = np.load(root / "train_traj_ids.npy", mmap_mode="r")
    test = np.load(root / "test_traj_ids.npy", mmap_mode="r")
    _require_unique(np.asarray(train), "train traj_id")
    _require_unique(np.asarray(test), "test traj_id")
    overlap = np.intersect1d(train, test, assume_unique=True)
    if len(overlap):
        examples = [int(value) for value in overlap[:10]]
        raise AlignmentError(f"split leakage: {len(overlap)} overlapping traj_id values, examples={examples}")
    return {"train": len(train), "test": len(test), "overlap": 0}


class AlignedHTPDataset(Dataset):
    """Join point-aligned GPS/Grid/Road arrays to HTP route/decoder targets by traj_id."""

    POINT_FIELDS = ("gps_seq", "time_seq", "dxy", "percent_dist")

    def __init__(
        self,
        aligned_root: str | Path,
        processed_path: str | Path,
        split: str,
        indices: np.ndarray | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise AlignmentError(f"split must be train or test, got {split!r}")
        self.aligned_root = Path(aligned_root)
        self.processed_path = Path(processed_path)
        self.split = split
        with (self.aligned_root / "metadata.json").open(encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        self.gps = np.load(self.aligned_root / f"{split}_gps.npy", mmap_mode="r")
        self.grid = np.load(self.aligned_root / f"{split}_grid.npy", mmap_mode="r")
        self.point_road = np.load(self.aligned_root / f"{split}_road.npy", mmap_mode="r")
        self.offsets = np.load(self.aligned_root / f"{split}_offsets.npy", mmap_mode="r")
        self.traj_ids = np.load(self.aligned_root / f"{split}_traj_ids.npy", mmap_mode="r")

        records = _load_pickle(self.processed_path)
        if not isinstance(records, list):
            raise AlignmentError("processed trajectory artifact must contain a list")
        processed_ids = [int(record["traj_id"]) for record in records]
        if len(set(processed_ids)) != len(processed_ids):
            raise AlignmentError("duplicate processed traj_id")
        self.records_by_id = {int(record["traj_id"]): record for record in records}

        self._validate_arrays()
        aligned_ids = {int(value) for value in self.traj_ids}
        record_ids = set(self.records_by_id)
        if aligned_ids != record_ids:
            missing_processed = sorted(aligned_ids - record_ids)[:10]
            missing_aligned = sorted(record_ids - aligned_ids)[:10]
            raise AlignmentError(
                "population mismatch: "
                f"missing_processed={missing_processed}, missing_aligned={missing_aligned}"
            )
        self._validate_record_lengths()

        if indices is None:
            self.indices = np.arange(len(self.traj_ids), dtype=np.int64)
        else:
            selected = np.asarray(indices, dtype=np.int64)
            if selected.ndim != 1 or (len(selected) and (selected.min() < 0 or selected.max() >= len(self.traj_ids))):
                raise AlignmentError("indices are outside the aligned trajectory population")
            self.indices = selected

    def _validate_arrays(self) -> None:
        trajectory_count = len(self.traj_ids)
        if self.traj_ids.ndim != 1:
            raise AlignmentError("traj_ids must be one-dimensional")
        _require_unique(np.asarray(self.traj_ids), f"{self.split} traj_id")
        if self.offsets.ndim != 1 or len(self.offsets) != trajectory_count + 1:
            raise AlignmentError("offset count must equal trajectory count + 1")
        if int(self.offsets[0]) != 0:
            raise AlignmentError("first offset must be zero")
        if np.any(np.diff(self.offsets) <= 0):
            raise AlignmentError("offsets must be strictly increasing and trajectories non-empty")
        point_count = len(self.gps)
        if int(self.offsets[-1]) != point_count:
            raise AlignmentError(
                f"final offset {int(self.offsets[-1])} does not equal GPS point count {point_count}"
            )
        if self.gps.ndim != 2 or self.gps.shape[1] != 2:
            raise AlignmentError("GPS must have shape [points, 2]")
        if self.grid.ndim != 1 or len(self.grid) != point_count:
            raise AlignmentError("Grid point count must equal GPS point count")
        if self.point_road.ndim != 1 or len(self.point_road) != point_count:
            raise AlignmentError("Road point count must equal GPS point count")
        if not np.isfinite(self.gps).all():
            raise AlignmentError("GPS contains non-finite values")

        num_grids = int(self.metadata["num_grids"])
        num_roads = int(self.metadata["num_roads"])
        if point_count and (int(self.grid.min()) < 0 or int(self.grid.max()) >= num_grids):
            raise AlignmentError(f"grid ID outside [0, {num_grids})")
        if point_count and (int(self.point_road.min()) < 0 or int(self.point_road.max()) >= num_roads):
            raise AlignmentError(f"road ID outside [0, {num_roads})")

    def _validate_record_lengths(self) -> None:
        for row, value in enumerate(self.traj_ids):
            traj_id = int(value)
            point_count = int(self.offsets[row + 1] - self.offsets[row])
            record = self.records_by_id[traj_id]
            for field in self.POINT_FIELDS:
                if field not in record:
                    raise AlignmentError(f"traj_id={traj_id} missing {field}")
                field_length = len(record[field])
                if field_length != point_count:
                    short_name = "dxy" if field == "dxy" else field
                    raise AlignmentError(
                        f"traj_id={traj_id} {short_name} length {field_length} != aligned length {point_count}"
                    )
            if "path" not in record or len(record["path"]) == 0:
                raise AlignmentError(f"traj_id={traj_id} route path must be non-empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        row = int(self.indices[index])
        traj_id = int(self.traj_ids[row])
        start = int(self.offsets[row])
        end = int(self.offsets[row + 1])
        record = self.records_by_id[traj_id]
        return {
            "traj_id": traj_id,
            "gps": np.asarray(self.gps[start:end]).copy(),
            "grid": np.asarray(self.grid[start:end], dtype=np.int64).copy(),
            "point_road": np.asarray(self.point_road[start:end], dtype=np.int64).copy(),
            "time": np.asarray(record["time_seq"], dtype=np.float32).copy(),
            "dxy": np.asarray(record["dxy"], dtype=np.float32).copy(),
            "road_percent": np.asarray(record["percent_dist"], dtype=np.float32).copy(),
            "route": np.asarray(record["path"], dtype=np.int64).copy(),
        }


def _pad(values: Sequence[np.ndarray], dtype: torch.dtype, padding_value: float | int) -> torch.Tensor:
    tensors = [torch.as_tensor(value, dtype=dtype) for value in values]
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def collate_aligned(
    batch: list[dict[str, np.ndarray | int]],
    road_context: RoadContext | None = None,
) -> dict[str, torch.Tensor]:
    if not batch:
        raise AlignmentError("cannot collate an empty batch")
    lengths = torch.as_tensor([len(item["gps"]) for item in batch], dtype=torch.long)
    route_lengths = torch.as_tensor([len(item["route"]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    point_mask = torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)
    dxy_raw = _pad([item["dxy"] for item in batch], torch.float32, 0.0)
    percent_raw = _pad([item["road_percent"] for item in batch], torch.float32, 0.0)
    dxy = dxy_raw.clone()
    percent = percent_raw.clone()
    route = _pad([item["route"] for item in batch], torch.long, -1)
    result = {
        "traj_id": torch.as_tensor([int(item["traj_id"]) for item in batch], dtype=torch.long),
        "gps": _pad([item["gps"] for item in batch], torch.float32, 0.0),
        "grid": _pad([item["grid"] for item in batch], torch.long, -1),
        "point_road": _pad([item["point_road"] for item in batch], torch.long, -1),
        "time": _pad([item["time"] for item in batch], torch.float32, 0.0),
        "dxy": dxy,
        "road_percent": percent,
        "dxy_raw": dxy_raw,
        "road_percent_raw": percent_raw,
        "route": route,
        "lengths": lengths,
        "route_lengths": route_lengths,
        "point_mask": point_mask,
    }
    if road_context is not None:
        scale = road_context.z_score
        result["dxy"][..., 0] = (
            result["dxy"][..., 0] - float(scale["mean_dx"])
        ) / float(scale["std_dx"])
        result["dxy"][..., 1] = (
            result["dxy"][..., 1] - float(scale["mean_dy"])
        ) / float(scale["std_dy"])
        result["road_percent"] = (
            result["road_percent"] - float(scale["mean_percent_dist"])
        ) / float(scale["std_percent_dist"])
        result["dxy"] = result["dxy"] * point_mask.unsqueeze(-1)
        result["road_percent"] = result["road_percent"] * point_mask
        flat_route = torch.cat(
            [torch.as_tensor(item["route"], dtype=torch.long) for item in batch]
        )
        if torch.any(flat_route < 0) or torch.any(flat_route >= len(road_context.geometry)):
            raise AlignmentError("route ID outside road geometry population")
        result["route_geometry"] = road_context.geometry[flat_route]
        result["route_geometry_lengths"] = road_context.geometry_lengths[flat_route]
    return result


def make_collator(road_context: RoadContext) -> Callable[[list[dict[str, np.ndarray | int]]], dict[str, torch.Tensor]]:
    return partial(collate_aligned, road_context=road_context)
