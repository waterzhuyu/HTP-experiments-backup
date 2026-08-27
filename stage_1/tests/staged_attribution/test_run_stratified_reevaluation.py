from __future__ import annotations

import json
from pathlib import Path

import pytest

from staged_attribution.run_stratified_reevaluation import (
    ReevaluationRunError,
    build_reevaluation_dag,
    run_stratified_reevaluation,
)


CONFIG_PATH = Path("stage_1/configs/staged_attribution/experiment.json")


class FakeBackend:
    def __init__(self, fail_at: str | None = None, escape_at: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_at = fail_at
        self.escape_at = escape_at

    def __call__(self, node, identity, root):
        del identity
        self.calls.append(node.name)
        if node.name == self.fail_at:
            raise RuntimeError("synthetic failure")
        result = {}
        for name in node.required_artifacts:
            path = root / node.name / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{node.name}:{name}")
            result[name] = path
        if node.name == self.escape_at:
            outside = root.parent / "outside.json"
            outside.write_text("outside")
            result[node.required_artifacts[0]] = outside
        return result


def _run(tmp_path: Path, backend: FakeBackend, name: str = "attempt_001"):
    return run_stratified_reevaluation(
        CONFIG_PATH,
        source_attempt=tmp_path / "source" / "attempt_001",
        subset_manifest=tmp_path / "subset.json",
        cell_id="e0_q0", city="porto", seed=42, budget="controlled",
        run_dir=tmp_path / name, device="cpu", stage_backend=backend,
        command=["test"], git_identity_fn=lambda: ("a" * 40, False),
    )


def test_reevaluation_dag_and_terminal_manifest_are_complete(tmp_path: Path) -> None:
    backend = FakeBackend()
    result = _run(tmp_path, backend)

    assert result["status"] == "completed"
    assert backend.calls == [node.name for node in build_reevaluation_dag()]
    manifest = json.loads((tmp_path / "attempt_001" / "reevaluation_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["git"]["dirty"] is False
    assert (tmp_path / "attempt_001" / "stage4" / "complete.json").is_file()

    second = FakeBackend()
    resumed = _run(tmp_path, second)
    assert resumed["status"] == "completed"
    assert second.calls == []


def test_failed_attempt_is_immutable_and_new_attempt_can_run(tmp_path: Path) -> None:
    backend = FakeBackend(fail_at="g_attribution")
    with pytest.raises(RuntimeError, match="synthetic failure"):
        _run(tmp_path, backend)
    manifest = json.loads((tmp_path / "attempt_001" / "reevaluation_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "RuntimeError"

    with pytest.raises(ReevaluationRunError, match="immutable"):
        _run(tmp_path, FakeBackend())
    assert _run(tmp_path, FakeBackend(), name="attempt_002")["status"] == "completed"


def test_resume_rejects_tamper_gap_identity_and_path_escape(tmp_path: Path) -> None:
    _run(tmp_path, FakeBackend())
    artifact = tmp_path / "attempt_001" / "stage3" / "generations.jsonl"
    artifact.write_text("tampered")
    with pytest.raises(ReevaluationRunError, match="size mismatch|SHA256 mismatch"):
        _run(tmp_path, FakeBackend())

    with pytest.raises(ReevaluationRunError, match="escapes"):
        _run(tmp_path, FakeBackend(escape_at="stage3"), name="attempt_002")
    escaped_manifest = json.loads((tmp_path / "attempt_002" / "reevaluation_manifest.json").read_text())
    assert escaped_manifest["status"] == "failed"

    root = tmp_path / "attempt_003"
    _run(tmp_path, FakeBackend(), name="attempt_003")
    (root / "stage3" / "complete.json").unlink()
    with pytest.raises(ReevaluationRunError, match="marker after an incomplete"):
        _run(tmp_path, FakeBackend(), name="attempt_003")

    manifest = json.loads((tmp_path / "attempt_003" / "reevaluation_manifest.json").read_text())
    manifest["seed"] = 314159
    (tmp_path / "attempt_003" / "reevaluation_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReevaluationRunError, match="identity mismatch"):
        _run(tmp_path, FakeBackend(), name="attempt_003")
