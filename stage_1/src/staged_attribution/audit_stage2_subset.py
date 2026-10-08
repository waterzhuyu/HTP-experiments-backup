from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .stage2_subset_manifest import (
    file_sha256,
    load_stage2_subset_manifest,
    write_immutable_json,
)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("summary values must be a non-empty vector")
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(quantiles[7]),
    }


def _ks_statistic(source: np.ndarray, selected: np.ndarray) -> float:
    support = np.unique(np.concatenate((source, selected)))
    source_sorted = np.sort(source)
    selected_sorted = np.sort(selected)
    source_cdf = np.searchsorted(source_sorted, support, side="right") / len(source_sorted)
    selected_cdf = np.searchsorted(selected_sorted, support, side="right") / len(
        selected_sorted
    )
    return float(np.max(np.abs(source_cdf - selected_cdf)))


def _lens_type_key(lengths: np.ndarray) -> np.ndarray:
    current = np.asarray(lengths, dtype=np.int64)
    b0 = current % 2
    current = (current + 1) // 2
    b1 = current % 2
    current = (current + 1) // 2
    b2 = current % 2
    return b0 * 4 + b1 * 2 + b2


def _lens_strata(source: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    source_keys = _lens_type_key(source)
    selected_keys = _lens_type_key(selected)
    result = {}
    for value in range(8):
        key = format(value, "03b")
        source_count = int(np.count_nonzero(source_keys == value))
        selected_count = int(np.count_nonzero(selected_keys == value))
        source_fraction = source_count / len(source)
        selected_fraction = selected_count / len(selected)
        result[key] = {
            "source_count": source_count,
            "selected_count": selected_count,
            "source_fraction": source_fraction,
            "selected_fraction": selected_fraction,
            "absolute_fraction_delta": abs(selected_fraction - source_fraction),
        }
    return {
        "strata": result,
        "maximum_absolute_fraction_delta": max(
            record["absolute_fraction_delta"] for record in result.values()
        ),
    }


def _js_divergence(source_counts: np.ndarray, selected_counts: np.ndarray) -> float:
    source = source_counts.astype(np.float64)
    selected = selected_counts.astype(np.float64)
    source /= source.sum()
    selected /= selected.sum()
    midpoint = 0.5 * (source + selected)
    source_mask = source > 0
    selected_mask = selected > 0
    divergence = 0.5 * np.sum(
        source[source_mask] * np.log2(source[source_mask] / midpoint[source_mask])
    )
    divergence += 0.5 * np.sum(
        selected[selected_mask]
        * np.log2(selected[selected_mask] / midpoint[selected_mask])
    )
    return float(divergence)


def _categorical_audit(
    source: np.ndarray,
    selected: np.ndarray,
    support_size: int,
) -> dict[str, float | int]:
    source_counts = np.bincount(source.astype(np.int64), minlength=support_size)
    selected_counts = np.bincount(selected.astype(np.int64), minlength=support_size)
    source_probability = source_counts / source_counts.sum()
    selected_probability = selected_counts / selected_counts.sum()
    source_support = int(np.count_nonzero(source_counts))
    selected_support = int(np.count_nonzero(selected_counts))
    return {
        "source_point_count": int(source_counts.sum()),
        "selected_point_count": int(selected_counts.sum()),
        "source_support": source_support,
        "selected_support": selected_support,
        "selected_source_support_coverage": selected_support / source_support,
        "total_variation_distance": float(
            0.5 * np.abs(source_probability - selected_probability).sum()
        ),
        "jensen_shannon_divergence_bits": _js_divergence(
            source_counts, selected_counts
        ),
    }


def audit_stage2_subset(
    manifest_path: str | Path,
    aligned_root: str | Path,
) -> dict[str, Any]:
    root = Path(aligned_root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    traj_ids_path = root / "train_traj_ids.npy"
    traj_ids = np.load(traj_ids_path, mmap_mode="r")
    with Path(manifest_path).open(encoding="utf-8") as handle:
        header = json.load(handle)
    selected_ids, _ = load_stage2_subset_manifest(
        manifest_path,
        traj_ids,
        str(header["city"]),
        int(header["requested"]),
        int(header["subset_seed"]),
    )
    row_by_id = {int(traj_id): row for row, traj_id in enumerate(traj_ids)}
    selected_rows = np.asarray([row_by_id[traj_id] for traj_id in selected_ids])
    offsets = np.load(root / "train_offsets.npy", mmap_mode="r")
    lengths = np.diff(offsets).astype(np.int64)
    if len(lengths) != len(traj_ids) or np.any(lengths <= 0):
        raise ValueError("aligned offsets do not define one non-empty span per trajectory")
    selected_lengths = lengths[selected_rows]
    selected_trajectory_mask = np.zeros(len(traj_ids), dtype=bool)
    selected_trajectory_mask[selected_rows] = True
    selected_point_mask = np.repeat(selected_trajectory_mask, lengths)

    gps = np.load(root / "train_gps.npy", mmap_mode="r")
    grid = np.load(root / "train_grid.npy", mmap_mode="r")
    road = np.load(root / "train_road.npy", mmap_mode="r")
    if not (len(gps) == len(grid) == len(road) == int(offsets[-1])):
        raise ValueError("aligned GPS/Grid/Road point populations differ")
    selected_gps = np.asarray(gps[selected_point_mask], dtype=np.float64)
    source_gps_mean = np.mean(gps, axis=0, dtype=np.float64)
    source_gps_std = np.std(gps, axis=0, dtype=np.float64)
    selected_gps_mean = selected_gps.mean(axis=0)
    selected_gps_std = selected_gps.std(axis=0)
    gps_labels = ("longitude_normalized", "latitude_normalized")
    gps_audit = {
        label: {
            "source_mean": float(source_gps_mean[index]),
            "selected_mean": float(selected_gps_mean[index]),
            "source_std": float(source_gps_std[index]),
            "selected_std": float(selected_gps_std[index]),
            "standardized_mean_difference": (
                float(
                    (selected_gps_mean[index] - source_gps_mean[index])
                    / source_gps_std[index]
                )
                if source_gps_std[index] > 0
                else 0.0
            ),
        }
        for index, label in enumerate(gps_labels)
    }

    source_length_summary = _summary(lengths)
    selected_length_summary = _summary(selected_lengths)
    pooled_std = float(
        np.sqrt((np.var(lengths) + np.var(selected_lengths)) / 2.0)
    )
    return {
        "schema_version": 1,
        "kind": "stage2_train_subset_representativeness_audit",
        "city": str(header["city"]),
        "manifest": {
            "path": str(Path(manifest_path).resolve()),
            "sha256": file_sha256(manifest_path),
        },
        "aligned_root": str(root.resolve()),
        "sampling_fraction": len(selected_rows) / len(traj_ids),
        "trajectory_length": {
            "source": source_length_summary,
            "selected": selected_length_summary,
            "ks_statistic": _ks_statistic(lengths, selected_lengths),
            "standardized_mean_difference": (
                float((selected_lengths.mean() - lengths.mean()) / pooled_std)
                if pooled_std > 0
                else 0.0
            ),
            **_lens_strata(lengths, selected_lengths),
        },
        "gps": gps_audit,
        "grid": _categorical_audit(
            np.asarray(grid),
            np.asarray(grid[selected_point_mask]),
            int(metadata["num_grids"]),
        ),
        "road": _categorical_audit(
            np.asarray(road),
            np.asarray(road[selected_point_mask]),
            int(metadata["num_roads"]),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a frozen Stage2 subset against aligned GPS/Grid/Road data."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_stage2_subset(args.manifest, args.aligned_root)
    write_immutable_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
