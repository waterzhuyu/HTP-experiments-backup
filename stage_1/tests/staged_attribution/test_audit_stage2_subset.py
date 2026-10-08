from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from staged_attribution.audit_stage2_subset import audit_stage2_subset
from staged_attribution.stage2_subset_manifest import (
    build_stage2_subset_manifest,
    write_immutable_json,
)


def test_audit_reports_gps_grid_road_and_lens_balance(tmp_path: Path) -> None:
    root = tmp_path / "aligned"
    root.mkdir()
    traj_ids = np.asarray([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    lengths = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    np.save(root / "train_traj_ids.npy", traj_ids)
    np.save(root / "train_offsets.npy", offsets)
    point_count = int(offsets[-1])
    gps = np.column_stack(
        (np.linspace(0.0, 1.0, point_count), np.linspace(1.0, 0.0, point_count))
    ).astype(np.float32)
    np.save(root / "train_gps.npy", gps)
    np.save(root / "train_grid.npy", np.arange(point_count, dtype=np.int64) % 4)
    np.save(root / "train_road.npy", np.arange(point_count, dtype=np.int64) % 5)
    (root / "metadata.json").write_text(
        json.dumps({"num_grids": 4, "num_roads": 5}), encoding="utf-8"
    )
    manifest = build_stage2_subset_manifest(
        traj_ids, "porto", 4, 20260826, root / "train_traj_ids.npy"
    )
    manifest_path = write_immutable_json(tmp_path / "subset.json", manifest)

    audit = audit_stage2_subset(manifest_path, root)

    assert audit["city"] == "porto"
    assert audit["sampling_fraction"] == 0.5
    assert audit["trajectory_length"]["source"]["count"] == 8
    assert audit["trajectory_length"]["selected"]["count"] == 4
    assert set(audit["trajectory_length"]["strata"]) == {
        "000", "001", "010", "011", "100", "101", "110", "111"
    }
    assert audit["gps"]["longitude_normalized"]["source_std"] > 0
    assert audit["grid"]["source_support"] == 4
    assert audit["road"]["source_support"] == 5
    assert 0 <= audit["grid"]["jensen_shannon_divergence_bits"] <= 1
    assert 0 <= audit["road"]["total_variation_distance"] <= 1
