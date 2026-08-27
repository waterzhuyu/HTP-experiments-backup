from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from staged_attribution.stage2_subset_manifest import (
    Stage2SubsetManifestError,
    build_stage2_subset_manifest,
    freeze_configured_subsets,
    load_stage2_subset_manifest,
    select_fixed_traj_ids,
    trajectory_id_sha256,
    validate_stage2_subset_manifest,
    write_immutable_json,
)


def source_file(tmp_path: Path, ids: list[int]) -> Path:
    path = tmp_path / "train_traj_ids.npy"
    np.save(path, np.asarray(ids, dtype=np.int64))
    return path


def test_fixed_subset_is_deterministic_sorted_and_city_scoped() -> None:
    ids = list(range(100, 140))

    first = select_fixed_traj_ids(ids, 12, 20260826, "porto")
    repeated = select_fixed_traj_ids(ids, 12, 20260826, "porto")
    changed_city = select_fixed_traj_ids(ids, 12, 20260826, "chengdu")
    changed_seed = select_fixed_traj_ids(ids, 12, 20260827, "porto")

    assert first == sorted(first)
    assert first == repeated
    assert len(first) == len(set(first)) == 12
    assert first != changed_city
    assert first != changed_seed


def test_manifest_round_trip_validates_source_and_exact_selection(tmp_path: Path) -> None:
    ids = [50, 10, 40, 20, 30]
    source = source_file(tmp_path, ids)
    manifest = build_stage2_subset_manifest(ids, "porto", 3, 20260826, source)
    path = write_immutable_json(tmp_path / "subset.json", manifest)

    selected, loaded = load_stage2_subset_manifest(path, ids, "porto", 3, 20260826)

    assert selected == loaded["selected"]["traj_ids"]
    assert loaded["source"]["population"] == 5
    assert loaded["source"]["traj_id_sha256"] == trajectory_id_sha256(ids)
    assert loaded["selected"]["traj_id_sha256"] == trajectory_id_sha256(selected)
    assert loaded["selection_depends_on_cell"] is False
    assert loaded["selection_depends_on_training_seed"] is False


def test_manifest_validation_rejects_changed_source_file(tmp_path: Path) -> None:
    ids = [50, 10, 40, 20, 30]
    source = source_file(tmp_path, ids)
    manifest = build_stage2_subset_manifest(ids, "porto", 3, 20260826, source)
    np.save(source, np.asarray([50, 10, 40, 20, 31], dtype=np.int64))

    with pytest.raises(Stage2SubsetManifestError, match="source file hash"):
        validate_stage2_subset_manifest(manifest, ids, "porto", 3, 20260826)


def test_immutable_writer_is_idempotent_and_rejects_changed_content(tmp_path: Path) -> None:
    path = write_immutable_json(tmp_path / "subset.json", {"value": 1})

    assert write_immutable_json(path, {"value": 1}) == path
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_immutable_json(path, {"value": 2})


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(city="chengdu"), "city mismatch"),
        (lambda value: value["source"].update(population=4), "source population"),
        (lambda value: value["selected"].update(traj_ids=[10, 20, 999]), "selected traj_id hash"),
    ],
)
def test_manifest_validation_rejects_drift(tmp_path: Path, mutation, match: str) -> None:
    ids = [50, 10, 40, 20, 30]
    source = source_file(tmp_path, ids)
    manifest = build_stage2_subset_manifest(ids, "porto", 3, 20260826, source)
    mutation(manifest)

    with pytest.raises(Stage2SubsetManifestError, match=match):
        validate_stage2_subset_manifest(manifest, ids, "porto", 3, 20260826)


def test_manifest_json_contains_no_training_seed_or_cell_identifier(tmp_path: Path) -> None:
    ids = list(range(20))
    source = source_file(tmp_path, ids)
    manifest = build_stage2_subset_manifest(ids, "porto", 8, 7, source)
    payload = json.dumps(manifest, sort_keys=True)

    assert "cell_id" not in manifest
    assert "training_seed" not in manifest
    assert '"selection_depends_on_training_seed": false' in payload


def test_freeze_configured_subsets_writes_idempotent_city_index(tmp_path: Path) -> None:
    paths = {}
    for offset, city in enumerate(("porto", "chengdu")):
        aligned_root = tmp_path / city
        aligned_root.mkdir()
        np.save(
            aligned_root / "train_traj_ids.npy",
            np.arange(offset * 100, offset * 100 + 10, dtype=np.int64),
        )
        paths[f"{city}_aligned_root"] = aligned_root
    config = SimpleNamespace(
        experiment_id="test_experiment",
        cities=("porto", "chengdu"),
        paths=paths,
        budgets={
            "controlled": SimpleNamespace(
                stage2_train_trajectories=4,
                stage2_subset_seed=20260826,
            )
        },
    )

    first = freeze_configured_subsets(config, "controlled", tmp_path / "frozen")
    repeated = freeze_configured_subsets(config, "controlled", tmp_path / "frozen")

    assert first == repeated
    assert first["cities"]["porto"]["selected_population"] == 4
    assert first["cities"]["chengdu"]["selected_population"] == 4
    assert Path(first["index"]).exists()
