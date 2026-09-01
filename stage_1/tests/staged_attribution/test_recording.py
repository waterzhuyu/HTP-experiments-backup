from __future__ import annotations

import json
from pathlib import Path

import pytest

from staged_attribution.recording import RecordError, RunRecorder


REQUIRED = ("resolved_config", "checkpoint", "stage1_metrics")


def create_recorder(tmp_path: Path, budget: str = "smoke", dirty: bool = False) -> RunRecorder:
    return RunRecorder.create(
        run_dir=tmp_path / "run",
        experiment_id="exp_v1",
        cell_id="e0_q0",
        city="porto",
        seed=42,
        budget=budget,
        command=["uv", "run", "python", "-m", "staged_attribution.train_stage1"],
        git_revision="a" * 40,
        git_dirty=dirty,
        required_artifacts=REQUIRED,
    )


def test_state_machine_requires_artifacts_before_completion(tmp_path: Path) -> None:
    recorder = create_recorder(tmp_path)

    recorder.start()
    with pytest.raises(RecordError, match="missing required artifacts"):
        recorder.complete({})

    artifacts = {}
    for name in REQUIRED:
        path = tmp_path / f"{name}.json"
        path.write_text(name)
        artifacts[name] = path
    recorder.complete(artifacts)

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert set(manifest["artifacts"]) == set(REQUIRED)
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"].values())
    assert not list((tmp_path / "run").glob("*.tmp"))


def test_formal_budget_rejects_dirty_tree(tmp_path: Path) -> None:
    for budget in ("controlled", "confirmation"):
        with pytest.raises(RecordError, match="clean Git worktree"):
            create_recorder(tmp_path / budget, budget=budget, dirty=True)


def test_smoke_records_but_allows_dirty_tree(tmp_path: Path) -> None:
    recorder = create_recorder(tmp_path, budget="smoke", dirty=True)

    manifest = recorder.read()

    assert manifest["git"]["dirty"] is True
    assert manifest["status"] == "planned"


def test_verified_launch_identity_is_written_unchanged(tmp_path: Path) -> None:
    identity = {
        "kind": "frozen_phase_launch_identity",
        "phase": "core",
        "cells": ["e3_q2"],
        "selection_sha256": "b" * 64,
        "verified": True,
    }
    recorder = RunRecorder.create(
        run_dir=tmp_path / "run",
        experiment_id="exp_v1",
        cell_id="e3_q2",
        city="porto",
        seed=42,
        budget="controlled",
        command=["python", "-m", "staged_attribution.run_cell"],
        git_revision="a" * 40,
        git_dirty=False,
        required_artifacts=REQUIRED,
        launch_identity=identity,
    )

    assert recorder.read()["launch_identity"] == identity


def test_unverified_launch_identity_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RecordError, match="must be verified"):
        RunRecorder.create(
            run_dir=tmp_path / "run",
            experiment_id="exp_v1",
            cell_id="e3_q2",
            city="porto",
            seed=42,
            budget="controlled",
            command=["python", "-m", "staged_attribution.run_cell"],
            git_revision="a" * 40,
            git_dirty=False,
            required_artifacts=REQUIRED,
            launch_identity={"verified": False},
        )


def test_invalid_transition_is_rejected(tmp_path: Path) -> None:
    recorder = create_recorder(tmp_path)

    with pytest.raises(RecordError, match="planned -> completed"):
        recorder.complete({})
    recorder.start()
    with pytest.raises(RecordError, match="running -> running"):
        recorder.start()


def test_failure_preserves_error_and_partial_artifacts(tmp_path: Path) -> None:
    recorder = create_recorder(tmp_path)
    recorder.start()
    partial = tmp_path / "partial.log"
    partial.write_text("epoch 1")

    recorder.fail(
        error_type="RuntimeError",
        message="CUDA out of memory",
        traceback_text="Traceback...",
        partial_artifacts={"train_log": partial},
    )

    manifest = recorder.read()
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "type": "RuntimeError",
        "message": "CUDA out of memory",
        "traceback": "Traceback...",
    }
    assert manifest["artifacts"]["train_log"]["path"] == str(partial.resolve())


def test_resume_reads_existing_record_without_overwrite(tmp_path: Path) -> None:
    recorder = create_recorder(tmp_path)
    recorder.start()

    resumed = RunRecorder.resume(tmp_path / "run")

    assert resumed.read()["status"] == "running"
    assert resumed.read()["command"] == [
        "uv",
        "run",
        "python",
        "-m",
        "staged_attribution.train_stage1",
    ]


def test_duplicate_create_refuses_to_replace_manifest(tmp_path: Path) -> None:
    create_recorder(tmp_path)

    with pytest.raises(FileExistsError, match="run_manifest"):
        create_recorder(tmp_path)


def test_complete_rejects_nonexistent_or_directory_artifact(tmp_path: Path) -> None:
    recorder = create_recorder(tmp_path)
    recorder.start()
    directory = tmp_path / "directory"
    directory.mkdir()
    artifacts = {
        "resolved_config": directory,
        "checkpoint": tmp_path / "missing.pt",
        "stage1_metrics": tmp_path / "missing.json",
    }

    with pytest.raises(RecordError, match="regular file"):
        recorder.complete(artifacts)
