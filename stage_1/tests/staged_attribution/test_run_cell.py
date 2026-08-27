from __future__ import annotations

import json
from pathlib import Path

import pytest

from staged_attribution.run_cell import (
    CellRunError,
    build_cell_dag,
    resume_plan,
    run_cell,
    write_stage_marker,
)


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("artifact")
    return path


def test_dag_contains_every_stage_and_no_proxy_terminal() -> None:
    dag = build_cell_dag("e6_q3")

    assert [node.name for node in dag] == [
        "stage1", "tokens", "stage2_data", "stage2", "stage3", "g_attribution", "decode", "stage4"
    ]
    assert dag[-1].prerequisites == ("decode",)
    assert dag[-1].required_artifacts == ("stage4_metrics.json",)
    assert "stage2_plan.json" in dag[3].required_artifacts
    assert all(node.required_artifacts for node in dag)


def test_marker_requires_prerequisites_and_existing_artifacts(tmp_path: Path) -> None:
    dag = build_cell_dag("e0_q0")
    with pytest.raises(CellRunError, match="prerequisite"):
        write_stage_marker(tmp_path, dag[1], {"train_tokens.jsonl": touch(tmp_path / "train.jsonl")})

    with pytest.raises(CellRunError, match="missing artifact"):
        write_stage_marker(tmp_path, dag[0], {})

    artifacts = {
        name: touch(tmp_path / "stage1" / name)
        for name in dag[0].required_artifacts
    }
    marker = write_stage_marker(tmp_path, dag[0], artifacts)
    payload = json.loads(marker.read_text())
    assert payload["stage"] == "stage1"
    assert set(payload["artifacts"]) == set(dag[0].required_artifacts)


def test_resume_starts_at_first_incomplete_stage(tmp_path: Path) -> None:
    dag = build_cell_dag("e0_q0")
    for node in dag[:2]:
        artifacts = {name: touch(tmp_path / node.name / name) for name in node.required_artifacts}
        write_stage_marker(tmp_path, node, artifacts)

    plan = resume_plan(tmp_path, dag)

    assert plan["completed"] == ["stage1", "tokens"]
    assert plan["remaining"][0] == "stage2_data"
    assert plan["terminal_complete"] is False


def test_dry_run_resolves_cell_without_writing(tmp_path: Path) -> None:
    result = run_cell(
        config_path=Path("stage_1/configs/staged_attribution/experiment.json"),
        cell_id="e3_q0",
        city="porto",
        seed=42,
        budget="smoke",
        run_dir=tmp_path,
        dry_run=True,
    )

    assert result["cell_id"] == "e3_q0"
    assert result["views"] == ["gps", "road"]
    assert result["stages"][-1] == "stage4"
    assert result["will_execute"] is False
    assert not any(tmp_path.iterdir())


def test_dry_run_resolves_preregistered_adaptive_e_by_q_cell(tmp_path: Path) -> None:
    result = run_cell(
        config_path=Path("stage_1/configs/staged_attribution/experiment.json"),
        cell_id="e3_q2",
        city="porto",
        seed=42,
        budget="controlled",
        run_dir=tmp_path,
        dry_run=True,
    )

    assert result["cell_id"] == "e3_q2"
    assert result["views"] == ["gps", "road"]
    assert result["q_weights"]["invariance"] == pytest.approx(0.05)
    assert result["q_weights"]["auxiliary"] == pytest.approx(0.0)


def test_backend_runs_every_stage_and_resume_performs_no_proxy_completion(tmp_path: Path) -> None:
    calls = []

    def backend(node, resolved, run_dir):
        calls.append(node.name)
        return {
            name: touch(Path(run_dir) / node.name / name)
            for name in node.required_artifacts
        }

    first = run_cell(
        config_path=Path("stage_1/configs/staged_attribution/experiment.json"),
        cell_id="e0_q0",
        city="porto",
        seed=42,
        budget="smoke",
        run_dir=tmp_path,
        stage_backend=backend,
    )

    assert calls == ["stage1", "tokens", "stage2_data", "stage2", "stage3", "g_attribution", "decode", "stage4"]
    assert first["terminal_complete"] is True
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert set(manifest["artifacts"]) == {"stage4_complete"}

    calls.clear()
    resumed = run_cell(
        config_path=Path("stage_1/configs/staged_attribution/experiment.json"),
        cell_id="e0_q0", city="porto", seed=42, budget="smoke", run_dir=tmp_path,
        stage_backend=backend,
    )
