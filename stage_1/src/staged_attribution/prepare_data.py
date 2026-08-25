from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .config import ExperimentConfig
from .data import AlignedHTPDataset, assert_split_disjoint


class DataPreparationError(ValueError):
    """Raised when source trajectories cannot produce the frozen aligned schema."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def grid_spec(mbr: Mapping[str, float], grid_size_m: float) -> dict[str, float | int]:
    if grid_size_m <= 0:
        raise DataPreparationError("grid_size_m must be positive")
    mid_lat_rad = math.radians((float(mbr["min_lat"]) + float(mbr["max_lat"])) / 2.0)
    meters_per_lon = 111_320.0 * math.cos(mid_lat_rad)
    meters_per_lat = 110_540.0
    width_m = (float(mbr["max_lon"]) - float(mbr["min_lon"])) * meters_per_lon
    height_m = (float(mbr["max_lat"]) - float(mbr["min_lat"])) * meters_per_lat
    if width_m <= 0 or height_m <= 0:
        raise DataPreparationError("MBR must have positive width and height")
    return {
        "grid_size_m": float(grid_size_m),
        "meters_per_lon": float(meters_per_lon),
        "meters_per_lat": float(meters_per_lat),
        "n_grid_x": int(math.ceil(width_m / grid_size_m)),
        "n_grid_y": int(math.ceil(height_m / grid_size_m)),
    }


def normalize_gps(gps: np.ndarray, mbr: Mapping[str, float]) -> np.ndarray:
    result = np.asarray(gps, dtype=np.float32).copy()
    result[:, 0] = (result[:, 0] - float(mbr["min_lon"])) / (
        float(mbr["max_lon"]) - float(mbr["min_lon"])
    )
    result[:, 1] = (result[:, 1] - float(mbr["min_lat"])) / (
        float(mbr["max_lat"]) - float(mbr["min_lat"])
    )
    return result


def gps_to_grid_ids(
    gps: np.ndarray,
    mbr: Mapping[str, float],
    spec: Mapping[str, float | int],
) -> np.ndarray:
    gx = np.floor(
        (gps[:, 0] - float(mbr["min_lon"]))
        * float(spec["meters_per_lon"])
        / float(spec["grid_size_m"])
    ).astype(np.int64)
    gy = np.floor(
        (gps[:, 1] - float(mbr["min_lat"]))
        * float(spec["meters_per_lat"])
        / float(spec["grid_size_m"])
    ).astype(np.int64)
    gx = np.clip(gx, 0, int(spec["n_grid_x"]) - 1)
    gy = np.clip(gy, 0, int(spec["n_grid_y"]) - 1)
    return gy * int(spec["n_grid_x"]) + gx


def _validate_indices(indices: np.ndarray, frame_size: int, split: str) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or not len(values):
        raise DataPreparationError(f"{split} indices must be a non-empty vector")
    if len(np.unique(values)) != len(values):
        raise DataPreparationError(f"duplicate {split} indices")
    if int(values.min()) < 0 or int(values.max()) >= frame_size:
        raise DataPreparationError(f"{split} indices outside source frame")
    return values


def _write_split(
    frame: pd.DataFrame,
    indices: np.ndarray,
    split: str,
    output_dir: Path,
    mbr: Mapping[str, float],
    spec: Mapping[str, float | int],
    num_roads: int,
) -> dict[str, int | float]:
    lengths = np.fromiter(
        (len(frame.iloc[int(index)].geometry) for index in indices),
        dtype=np.int64,
        count=len(indices),
    )
    if np.any(lengths <= 0):
        raise DataPreparationError(f"{split} contains an empty GPS trajectory")
    offsets = np.zeros(len(indices) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total_points = int(offsets[-1])
    gps_flat = np.empty((total_points, 2), dtype=np.float32)
    grid_flat = np.empty(total_points, dtype=np.int32)
    road_flat = np.empty(total_points, dtype=np.int32)
    traj_ids = np.empty(len(indices), dtype=np.int64)

    for output_index, frame_index in enumerate(indices):
        row = frame.iloc[int(frame_index)]
        gps = np.vstack(row.geometry).astype(np.float64, copy=False)
        road = np.asarray(row.opath, dtype=np.int64)
        if gps.ndim != 2 or gps.shape[1] != 2 or len(gps) != len(road):
            raise DataPreparationError(
                f"unaligned views at traj_id={row.traj_id}: gps={gps.shape}, road={road.shape}"
            )
        if not np.isfinite(gps).all():
            raise DataPreparationError(f"non-finite GPS at traj_id={row.traj_id}")
        if len(road) == 0 or int(road.min()) < 0 or int(road.max()) >= num_roads:
            raise DataPreparationError(f"invalid road IDs at traj_id={row.traj_id}")
        start = int(offsets[output_index])
        end = int(offsets[output_index + 1])
        gps_flat[start:end] = normalize_gps(gps, mbr)
        grid_flat[start:end] = gps_to_grid_ids(gps, mbr, spec).astype(np.int32)
        road_flat[start:end] = road.astype(np.int32)
        traj_ids[output_index] = int(row.traj_id)

    if len(np.unique(traj_ids)) != len(traj_ids):
        raise DataPreparationError(f"duplicate traj_id values in {split} split")
    np.save(output_dir / f"{split}_gps.npy", gps_flat)
    np.save(output_dir / f"{split}_grid.npy", grid_flat)
    np.save(output_dir / f"{split}_road.npy", road_flat)
    np.save(output_dir / f"{split}_offsets.npy", offsets)
    np.save(output_dir / f"{split}_traj_ids.npy", traj_ids)
    return {
        "trajectories": int(len(indices)),
        "points": total_points,
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
    }


def prepare_from_frame(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    output_dir: str | Path,
    city: str,
    mbr: Mapping[str, float],
    num_roads: int,
    grid_size_m: float = 100.0,
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required_columns = {"traj_id", "geometry", "opath"}
    if not required_columns.issubset(frame.columns):
        raise DataPreparationError(f"source frame missing columns {sorted(required_columns - set(frame.columns))}")
    train_indices = _validate_indices(train_indices, len(frame), "train")
    test_indices = _validate_indices(test_indices, len(frame), "test")
    overlap = np.intersect1d(train_indices, test_indices, assume_unique=True)
    if len(overlap):
        raise DataPreparationError(f"train/test indices overlap at {overlap[:10].tolist()}")
    if num_roads <= 0:
        raise DataPreparationError("num_roads must be positive")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spec = grid_spec(mbr, grid_size_m)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "city": city,
        **dict(source_manifest or {}),
        "mbr": {name: float(value) for name, value in mbr.items()},
        "grid": spec,
        "num_grids": int(spec["n_grid_x"]) * int(spec["n_grid_y"]),
        "num_roads": int(num_roads),
        "view_definition": {
            "gps": "normalized longitude/latitude per observation",
            "grid": "100m local-equirectangular grid ID per observation",
            "road": "released point-wise map-matched opath road ID per observation",
        },
    }
    metadata["train"] = _write_split(
        frame, train_indices, "train", output, mbr, spec, num_roads
    )
    metadata["test"] = _write_split(
        frame, test_indices, "test", output, mbr, spec, num_roads
    )
    output_files = [
        output / f"{split}_{kind}.npy"
        for split in ("train", "test")
        for kind in ("gps", "grid", "road", "offsets", "traj_ids")
    ]
    metadata["output_sha256"] = {path.name: file_sha256(path) for path in output_files}
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def audit_existing(config: ExperimentConfig, city: str) -> dict[str, Any]:
    if city not in config.cities:
        raise DataPreparationError(f"city {city!r} is not in the frozen registry")
    aligned_root = config.paths[f"{city}_aligned_root"]
    data_root = config.paths["data_root"] / city
    split_audit = assert_split_disjoint(aligned_root)
    result: dict[str, Any] = {"city": city, "split": split_audit, "populations": {}}
    for split in ("train", "test"):
        dataset = AlignedHTPDataset(
            aligned_root,
            data_root / f"processed_{split}_traj.pkl",
            split,
        )
        result["populations"][split] = len(dataset)
        del dataset
        gc.collect()
    metadata_path = aligned_root / "metadata.json"
    result["metadata"] = str(metadata_path)
    result["metadata_sha256"] = file_sha256(metadata_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or audit aligned GPS/Grid/Road HTP data.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--city", choices=("porto", "chengdu"), required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.load(args.config)
    if args.audit_only:
        print(json.dumps(audit_existing(config, args.city), indent=2, sort_keys=True))
        return

    data_root = config.paths["data_root"] / args.city
    output_dir = config.paths[f"{args.city}_aligned_root"]
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"refusing to overwrite non-empty aligned root: {output_dir}")
    stage1_config = yaml.safe_load(Path("config", args.city, "stage1_config.yaml").read_text())
    source_path = data_root / "traj.parquet"
    train_index_path = data_root / "train_index.npy"
    test_index_path = data_root / "test_index.npy"
    frame = pd.read_parquet(source_path, columns=["traj_id", "geometry", "opath"])
    num_roads = len(pd.read_csv(data_root / "rn" / "edge_info.csv", usecols=["from_node_id"]))
    metadata = prepare_from_frame(
        frame=frame,
        train_indices=np.load(train_index_path),
        test_indices=np.load(test_index_path),
        output_dir=output_dir,
        city=args.city,
        mbr=stage1_config["mbr"],
        num_roads=num_roads,
        grid_size_m=100.0,
        source_manifest={
            "source": str(source_path),
            "source_sha256": file_sha256(source_path),
            "split_source": {
                "train": str(train_index_path),
                "train_sha256": file_sha256(train_index_path),
                "test": str(test_index_path),
                "test_sha256": file_sha256(test_index_path),
            },
        },
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
