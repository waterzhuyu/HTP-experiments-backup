from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import staged_attribution.freeze_stratified_subsets as freezer_module

from staged_attribution.freeze_stratified_subsets import (
    FrozenSubsetError,
    freeze_stratified_subsets,
)
from staged_attribution.reevaluation_evidence import load_stratified_manifest
from staged_attribution.stratified_subset import REGISTERED_STRATA, lens_type_key


def _lengths_covering_all_strata() -> list[int]:
    grouped: dict[str, list[int]] = {key: [] for key in REGISTERED_STRATA}
    for length in range(1, 200):
        key = lens_type_key(length)
        if len(grouped[key]) < 2:
            grouped[key].append(length)
        if all(len(values) == 2 for values in grouped.values()):
            break
    assert all(len(values) == 2 for values in grouped.values())
    return [length for key in sorted(grouped) for length in grouped[key]]


def _aligned_root(path: Path) -> None:
    path.mkdir(parents=True)
    lengths = _lengths_covering_all_strata()
    ids = np.arange(100, 100 + len(lengths), dtype=np.int64)
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)]
    )
    np.save(path / "test_traj_ids.npy", ids, allow_pickle=False)
    np.save(path / "test_offsets.npy", offsets, allow_pickle=False)


def _config(tmp_path: Path) -> Path:
    source = Path("stage_1/configs/staged_attribution/experiment.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    for city in payload["cities"]:
        root = tmp_path / city
        _aligned_root(root)
        payload["paths"][f"{city}_aligned_root"] = str(root.resolve())
    payload["budgets"]["controlled"]["stage3_trajectories"] = 8
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_freeze_stratified_subsets_writes_batch_compatible_immutable_manifests(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    output = tmp_path / "subsets"

    first = freeze_stratified_subsets(config, output)
    second = freeze_stratified_subsets(config, output)

    assert first == second
    assert first["kind"] == "fixed_stratified_reevaluation_subsets"
    assert first["selection_depends_on_cell"] is False
    assert len(first["manifests"]) == 6
    assert Path(first["index_path"]).is_file()
    for row in first["manifests"]:
        path = Path(row["path"])
        assert path.name == f"controlled_{row['city']}_seed_{row['seed']}.json"
        manifest = load_stratified_manifest(
            path,
            city=str(row["city"]),
            seed=int(row["seed"]),
            expected_size=8,
        )
        assert manifest["selection_depends_on_cell"] is False
        assert manifest["source"]["population"] == 16
        assert set(manifest["strata"]) == set(REGISTERED_STRATA)
        assert all(value["quota"] == 1 for value in manifest["strata"].values())


def test_freeze_stratified_subsets_rejects_existing_content_drift(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    output = tmp_path / "subsets"
    result = freeze_stratified_subsets(config, output)
    path = Path(result["manifests"][0]["path"])
    payload = json.loads(path.read_text())
    payload["selected_traj_ids"] = list(reversed(payload["selected_traj_ids"]))
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FrozenSubsetError, match="refusing to replace immutable"):
        freeze_stratified_subsets(config, output)


def test_freeze_stratified_subsets_rejects_nonfixed_budget_and_bad_source(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(FrozenSubsetError, match="only for controlled"):
        freeze_stratified_subsets(config, tmp_path / "subsets", budget="confirmation")

    payload = json.loads(config.read_text())
    porto = Path(payload["paths"]["porto_aligned_root"])
    np.save(
        porto / "test_traj_ids.npy",
        np.asarray([2, 2], dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        porto / "test_offsets.npy",
        np.asarray([0, 2, 4], dtype=np.int64),
        allow_pickle=False,
    )
    with pytest.raises(FrozenSubsetError, match="nonempty/unique"):
        freeze_stratified_subsets(config, tmp_path / "subsets")


def test_freeze_stratified_subsets_accepts_unsorted_aligned_authority(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    baseline = freeze_stratified_subsets(config, tmp_path / "baseline")
    payload = json.loads(config.read_text())
    porto = Path(payload["paths"]["porto_aligned_root"])
    ids = np.load(porto / "test_traj_ids.npy", allow_pickle=False)
    offsets = np.load(porto / "test_offsets.npy", allow_pickle=False)
    lengths = np.diff(offsets)
    reversed_ids = ids[::-1].copy()
    reversed_lengths = lengths[::-1].copy()
    reversed_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(reversed_lengths, dtype=np.int64)]
    )
    np.save(porto / "test_traj_ids.npy", reversed_ids, allow_pickle=False)
    np.save(porto / "test_offsets.npy", reversed_offsets, allow_pickle=False)

    reordered = freeze_stratified_subsets(config, tmp_path / "reordered")
    baseline_rows = {
        (row["city"], row["seed"]): json.loads(Path(row["path"]).read_text())
        for row in baseline["manifests"]
    }
    reordered_rows = {
        (row["city"], row["seed"]): json.loads(Path(row["path"]).read_text())
        for row in reordered["manifests"]
    }
    for seed in payload["seeds"]:
        first = baseline_rows[("porto", seed)]
        second = reordered_rows[("porto", seed)]
        assert first["selected_traj_ids"] == second["selected_traj_ids"]
        assert first["source"]["traj_id_sha256"] == second["source"]["traj_id_sha256"]
        assert (
            first["source"]["traj_id_order_sha256"]
            != second["source"]["traj_id_order_sha256"]
        )


def test_freeze_stratified_subsets_rejects_noninteger_ids_and_nonzero_offset(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.read_text())
    porto = Path(payload["paths"]["porto_aligned_root"])
    np.save(
        porto / "test_traj_ids.npy",
        np.asarray([1.0, 2.0], dtype=np.float64),
        allow_pickle=False,
    )
    np.save(
        porto / "test_offsets.npy",
        np.asarray([0, 2, 4], dtype=np.int64),
        allow_pickle=False,
    )
    with pytest.raises(FrozenSubsetError, match="integer dtypes"):
        freeze_stratified_subsets(config, tmp_path / "subsets")

    np.save(
        porto / "test_traj_ids.npy",
        np.asarray([1, 2], dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        porto / "test_offsets.npy",
        np.asarray([1, 3, 5], dtype=np.int64),
        allow_pickle=False,
    )
    with pytest.raises(FrozenSubsetError, match="begin at zero"):
        freeze_stratified_subsets(config, tmp_path / "subsets")


def test_freeze_stratified_subsets_enforces_formal_provenance_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    revision = "a" * 40
    clean_state = {
        "worktree": "/synthetic/worktree",
        "revision": revision,
        "dirty": False,
    }
    monkeypatch.setattr(freezer_module, "_git_state", lambda: clean_state)
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()

    with pytest.raises(FrozenSubsetError, match="config SHA-256"):
        freeze_stratified_subsets(
            config,
            tmp_path / "bad_config",
            expected_config_sha256="0" * 64,
        )
    with pytest.raises(FrozenSubsetError, match="generator revision"):
        freeze_stratified_subsets(
            config,
            tmp_path / "bad_revision",
            expected_revision="b" * 40,
        )
    monkeypatch.setattr(
        freezer_module,
        "_git_state",
        lambda: {**clean_state, "dirty": True},
    )
    with pytest.raises(FrozenSubsetError, match="must be clean"):
        freeze_stratified_subsets(
            config,
            tmp_path / "dirty",
            require_clean=True,
        )

    monkeypatch.setattr(freezer_module, "_git_state", lambda: clean_state)
    result = freeze_stratified_subsets(
        config,
        tmp_path / "accepted",
        expected_revision=revision,
        expected_config_sha256=config_sha,
        require_clean=True,
    )
    manifest = json.loads(Path(result["manifests"][0]["path"]).read_text())
    assert manifest["generator_git"] == clean_state
    assert manifest["config"]["sha256"] == config_sha
