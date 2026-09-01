from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from staged_attribution.pipeline import (
    PipelineBackend,
    meters_per_normalized_coordinate,
    select_dataset_indices,
    select_stage2_train_frame,
    trajectory_id_sha256,
)
from staged_attribution.population_bootstrap import load_metric_contributions
from staged_attribution.stage2_subset_manifest import (
    build_stage2_subset_manifest,
    write_immutable_json,
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


def test_stage2_fixed_subset_is_cell_and_training_seed_independent() -> None:
    frame = pd.DataFrame({"traj_id": [50, 10, 40, 20, 30], "value": range(5)})

    first, first_audit = select_stage2_train_frame(frame, 3, 20260826, "porto")
    repeated, repeated_audit = select_stage2_train_frame(frame, 3, 20260826, "porto")
    other_city, _ = select_stage2_train_frame(frame, 3, 20260826, "chengdu")
    other_subset_seed, _ = select_stage2_train_frame(frame, 3, 20260827, "porto")

    selected_ids = [int(value) for value in first["traj_id"]]
    assert selected_ids == sorted(selected_ids)
    assert first.equals(repeated)
    assert first_audit == repeated_audit
    assert first_audit["method"] == "sha256_rank_fixed_city_v1"
    assert first_audit["source_population"] == 5
    assert first_audit["population"] == 3
    assert first_audit["requested"] == 3
    assert first_audit["selected_traj_ids"] == selected_ids
    assert first_audit["selected_traj_id_sha256"] == trajectory_id_sha256(selected_ids)
    assert first_audit["selection_depends_on_training_seed"] is False
    assert first["traj_id"].tolist() != other_city["traj_id"].tolist()
    assert first["traj_id"].tolist() != other_subset_seed["traj_id"].tolist()


def test_stage2_full_population_preserves_frame_order_and_audits_hash() -> None:
    frame = pd.DataFrame({"traj_id": [30, 10, 20], "value": [3, 1, 2]})

    selected, audit = select_stage2_train_frame(frame, "full", 7, "porto")

    assert selected.equals(frame)
    assert audit["method"] == "full_population_v1"
    assert audit["source_population"] == 3
    assert audit["population"] == 3
    assert audit["selected_traj_ids"] is None
    assert audit["selected_traj_id_sha256"] == trajectory_id_sha256([30, 10, 20])


def test_stage2_selection_uses_and_audits_frozen_manifest(tmp_path) -> None:
    frame = pd.DataFrame({"traj_id": [50, 10, 40, 20, 30], "value": range(5)})
    source_path = tmp_path / "train_traj_ids.npy"
    np.save(source_path, frame["traj_id"].to_numpy(dtype=np.int64))
    manifest = build_stage2_subset_manifest(
        frame["traj_id"].tolist(), "porto", 3, 20260826, source_path
    )
    manifest_path = write_immutable_json(tmp_path / "subset.json", manifest)

    selected, audit = select_stage2_train_frame(
        frame, 3, 20260826, "porto", manifest_path
    )

    assert selected["traj_id"].tolist() == manifest["selected"]["traj_ids"]
    assert audit["method"] == "frozen_manifest_sha256_rank_fixed_city_v1"
    assert audit["manifest_path"] == str(manifest_path.resolve())
    assert len(audit["manifest_sha256"]) == 64


@pytest.mark.parametrize(
    "frame,requested,match",
    [
        (pd.DataFrame({"traj_id": [1, 1]}), 1, "unique traj_id"),
        (pd.DataFrame({"traj_id": [1, 2]}), 3, "invalid requested population"),
    ],
)
def test_stage2_subset_rejects_invalid_population(
    frame: pd.DataFrame, requested: int, match: str
) -> None:
    with pytest.raises(RuntimeError, match=match):
        select_stage2_train_frame(frame, requested, 1, "porto")


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


def test_primary_stage4_persists_population_aware_contributions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Dataset:
        metadata = {
            "mbr": {
                "min_lon": 0.0,
                "max_lon": 1.0,
                "min_lat": 0.0,
                "max_lat": 1.0,
            },
            "grid": {
                "grid_size_m": 100.0,
                "meters_per_lon": 100_000.0,
                "meters_per_lat": 100_000.0,
                "n_grid_x": 1000,
                "n_grid_y": 1000,
            },
        }
        records_by_id = {
            7: {"gps_seq": [[0.1, 0.1], [0.2, 0.2]]},
        }

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            assert index == 0
            return {"traj_id": 7, "point_road": [0, 0]}

    run_dir = tmp_path / "attempt"
    decode = run_dir / "decode"
    decode.mkdir(parents=True)
    record = {
        "traj_id": 7,
        "status": "reconstructed",
        "gps_traj": [[0.1, 0.1], [0.2, 0.2]],
        "road_traj": [0, 0],
        "failure": None,
    }
    for variant in ("g00", "g10", "g01", "g11"):
        (decode / f"{variant}_reconstructed.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
    backend = object.__new__(PipelineBackend)
    backend.city = "porto"
    monkeypatch.setattr(backend, "_dataset", lambda split: Dataset())
    monkeypatch.setattr(
        backend, "_road_geometries", lambda: {0: np.asarray([[0.0, 0.0], [1.0, 1.0]])}
    )

    artifacts = backend._stage4(run_dir)

    expected = {"stage4_metrics.json"} | {
        f"{variant}_contributions_manifest.json"
        for variant in ("g00", "g10", "g01", "g11")
    }
    assert set(artifacts) == expected
    metrics = json.loads(artifacts["stage4_metrics.json"].read_text())
    for variant in ("g00", "g10", "g01", "g11"):
        evidence = metrics[variant]["population_aware_regret"]
        assert evidence["method"] == "failure_weighted_population_regret_v1"
        assert all(value == pytest.approx(0.0) for value in evidence["metrics"].values())
        loaded = load_metric_contributions(
            artifacts[f"{variant}_contributions_manifest.json"]
        )
        assert loaded.traj_ids == (7,)
        assert loaded.valid.tolist() == [True]
