from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from staged_attribution.run_stratified_reevaluation import (
    ReevaluationRunError,
    build_reevaluation_dag,
    run_stratified_reevaluation,
)


CONFIG_PATH = Path("stage_1/configs/staged_attribution/experiment.json")
EVALUATION_REVISION = "a" * 40
SOURCE_REVISION = "b" * 40


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


def _run(
    tmp_path: Path,
    backend: FakeBackend,
    name: str = "attempt_001",
    *,
    expected_revision: str = EVALUATION_REVISION,
    expected_config_sha256: str | None = None,
    expected_source_audit_sha256: str | None = None,
):
    config_sha256 = expected_config_sha256 or hashlib.sha256(
        CONFIG_PATH.read_bytes()
    ).hexdigest()
    source_attempt = (tmp_path / "source" / "attempt_001").resolve()
    subset_manifest = tmp_path / "subset.json"
    if not subset_manifest.exists():
        subset_manifest.write_text("{}")
    source_audit = tmp_path / "source_audit.json"
    if not source_audit.exists():
        source_audit.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "completed_formal_source_evidence_audit",
                    "experiment_id": json.loads(CONFIG_PATH.read_text())[
                        "experiment_id"
                    ],
                    "budget": "controlled",
                    "config": {
                        "sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
                    },
                    "all_requested_sources_completed": True,
                    "missing_completed_source_count": 0,
                    "audited_completed_sources": [
                        {
                            "run_key": "controlled/porto/seed_42/e0_q0",
                            "cell_id": "e0_q0",
                            "city": "porto",
                            "seed": 42,
                            "budget": "controlled",
                            "source_attempt": str(source_attempt),
                            "source_revision": SOURCE_REVISION,
                            "fixed_subset": {
                                "path": str(subset_manifest.resolve()),
                                "sha256": hashlib.sha256(
                                    subset_manifest.read_bytes()
                                ).hexdigest(),
                            },
                        }
                    ],
                }
            )
        )
    source_audit_sha256 = expected_source_audit_sha256 or hashlib.sha256(
        source_audit.read_bytes()
    ).hexdigest()
    return run_stratified_reevaluation(
        CONFIG_PATH,
        source_attempt=source_attempt,
        subset_manifest=subset_manifest,
        cell_id="e0_q0", city="porto", seed=42, budget="controlled",
        run_dir=tmp_path / name, device="cpu", stage_backend=backend,
        expected_revision=expected_revision,
        expected_config_sha256=config_sha256,
        expected_source_revision=SOURCE_REVISION,
        source_audit_path=source_audit,
        expected_source_audit_sha256=source_audit_sha256,
        command=["test"], git_identity_fn=lambda: (EVALUATION_REVISION, False),
    )


def test_reevaluation_dag_and_terminal_manifest_are_complete(tmp_path: Path) -> None:
    backend = FakeBackend()
    result = _run(tmp_path, backend)

    assert result["status"] == "completed"
    assert backend.calls == [node.name for node in build_reevaluation_dag()]
    manifest = json.loads((tmp_path / "attempt_001" / "reevaluation_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["git"]["dirty"] is False
    assert manifest["config_sha256"] == hashlib.sha256(
        CONFIG_PATH.read_bytes()
    ).hexdigest()
    assert manifest["expected_source_revision"] == SOURCE_REVISION
    assert manifest["source_audit_sha256"] == hashlib.sha256(
        (tmp_path / "source_audit.json").read_bytes()
    ).hexdigest()
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


def test_identity_gate_rejects_revision_and_config_before_attempt_creation(
    tmp_path: Path,
) -> None:
    revision_attempt = tmp_path / "revision_attempt"
    with pytest.raises(ReevaluationRunError, match="revision mismatch"):
        _run(
            tmp_path,
            FakeBackend(),
            name=revision_attempt.name,
            expected_revision="c" * 40,
        )
    assert not revision_attempt.exists()

    config_attempt = tmp_path / "config_attempt"
    with pytest.raises(ReevaluationRunError, match="config SHA-256 mismatch"):
        _run(
            tmp_path,
            FakeBackend(),
            name=config_attempt.name,
            expected_config_sha256="0" * 64,
        )
    assert not config_attempt.exists()

    audit_attempt = tmp_path / "audit_attempt"
    with pytest.raises(ReevaluationRunError, match="audit SHA-256 mismatch"):
        _run(
            tmp_path,
            FakeBackend(),
            name=audit_attempt.name,
            expected_source_audit_sha256="0" * 64,
        )
    assert not audit_attempt.exists()


def test_source_audit_row_binding_rejects_rehashed_attempt_substitution(
    tmp_path: Path,
) -> None:
    _run(tmp_path, FakeBackend())
    source_audit = tmp_path / "source_audit.json"
    payload = json.loads(source_audit.read_text())
    payload["audited_completed_sources"][0]["source_attempt"] = str(
        (tmp_path / "source" / "attempt_999").resolve()
    )
    source_audit.write_text(json.dumps(payload))
    source_audit_sha256 = hashlib.sha256(source_audit.read_bytes()).hexdigest()

    with pytest.raises(ReevaluationRunError, match="audit attempt mismatch"):
        _run(
            tmp_path,
            FakeBackend(),
            name="attempt_002",
            expected_source_audit_sha256=source_audit_sha256,
        )
    assert not (tmp_path / "attempt_002").exists()


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
