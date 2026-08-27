from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from staged_attribution.reevaluation_evidence import (
    ReevaluationEvidenceError,
    audit_source_population,
    load_stratified_manifest,
    resolve_source_evidence,
    write_source_reference,
)
from staged_attribution.run_cell import StageNode, write_stage_marker
from staged_attribution.stratified_subset import stratified_subset


CONFIG_PATH = Path("stage_1/configs/staged_attribution/experiment.json")
REVISION = "a" * 40


def _source_attempt(tmp_path: Path) -> tuple[Path, Path]:
    attempt = tmp_path / "source" / "attempt_001"
    artifacts = {
        "stage1": {"stage1_model.pt": attempt / "stage1" / "stage1_model.pt"},
        "tokens": {"test_tokens.jsonl": attempt / "tokens" / "test_tokens.jsonl"},
        "stage2_data": {"llm_test.parquet": attempt / "stage2_data" / "llm_test.parquet"},
        "stage2": {"adapter_manifest.json": attempt / "stage2" / "adapter_manifest.json"},
    }
    for stage in artifacts.values():
        for path in stage.values():
            path.parent.mkdir(parents=True, exist_ok=True)
    artifacts["stage1"]["stage1_model.pt"].write_bytes(b"model")
    rows = [
        {"traj_id": 1, "lens_type": [0, 0, 1]},
        {"traj_id": 2, "lens_type": [1, 1, 0]},
    ]
    artifacts["tokens"]["test_tokens.jsonl"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    pd.DataFrame({"traj_id": [1, 2], "desc": ["a", "b"]}).to_parquet(
        artifacts["stage2_data"]["llm_test.parquet"], index=False
    )
    adapter = attempt / "stage2" / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "tokenizer.json").write_text("{}")
    artifacts["stage2"]["adapter_manifest.json"].write_text(
        json.dumps({"adapter": str(adapter.resolve()), "seed": 42}), encoding="utf-8"
    )
    for stage, values in artifacts.items():
        node = StageNode(stage, (), tuple(values))
        write_stage_marker(attempt, node, values)
    (attempt / "stage4").mkdir()
    (attempt / "stage4" / "complete.json").write_text("{}")
    (attempt / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "gps_road_grid_staged_attribution_v1_20260826",
                "cell_id": "e0_q0", "city": "porto", "seed": 42,
                "budget": "controlled", "status": "completed",
                "git": {"revision": REVISION, "dirty": False},
            }
        )
    )
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    np.save(aligned / "test_traj_ids.npy", np.asarray([1, 2], dtype=np.int64))
    np.save(aligned / "test_offsets.npy", np.asarray([0, 20, 41], dtype=np.int64))
    return attempt, aligned


def _subset(tmp_path: Path) -> Path:
    selected = stratified_subset(
        [1, 2], [20, 21], 2, 42, require_all_registered_strata=False
    )
    payload = {
        **selected,
        "schema_version": 1,
        "authoritative_revision": REVISION,
        "city": "porto",
        "source": {},
    }
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_source_subset_population_reference_is_hash_complete(tmp_path: Path) -> None:
    attempt, aligned = _source_attempt(tmp_path)
    evidence = resolve_source_evidence(
        CONFIG_PATH, attempt, cell_id="e0_q0", city="porto", seed=42,
        budget="controlled", expected_revision=REVISION,
    )
    subset = load_stratified_manifest(
        _subset(tmp_path), city="porto", seed=42, expected_size=2
    )
    audit = audit_source_population(evidence, subset, aligned)
    reference = write_source_reference(tmp_path / "reeval" / "source_reference.json", evidence, subset, audit)
    payload = json.loads(reference.read_text())

    assert audit["lens_formula_verified"] is True
    assert payload["source_revision"] == REVISION
    assert len(payload["source_artifacts"]["adapter_tree"]) == 2
    assert payload["subset"]["selected_traj_id_sha256"] == subset["selected_traj_id_sha256"]


def test_source_evidence_rejects_identity_escape_and_tampering(tmp_path: Path) -> None:
    attempt, _ = _source_attempt(tmp_path)
    manifest = json.loads((attempt / "run_manifest.json").read_text())
    manifest["seed"] = 314159
    (attempt / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReevaluationEvidenceError, match="identity mismatch"):
        resolve_source_evidence(
            CONFIG_PATH, attempt, cell_id="e0_q0", city="porto", seed=42,
            budget="controlled", expected_revision=REVISION,
        )

    manifest["seed"] = 42
    (attempt / "run_manifest.json").write_text(json.dumps(manifest))
    adapter_manifest = attempt / "stage2" / "adapter_manifest.json"
    adapter_manifest.write_text(json.dumps({"adapter": str(tmp_path.resolve())}))
    marker = json.loads((attempt / "stage2" / "complete.json").read_text())
    info = marker["artifacts"]["adapter_manifest.json"]
    info["size_bytes"] = adapter_manifest.stat().st_size
    info["sha256"] = hashlib.sha256(adapter_manifest.read_bytes()).hexdigest()
    (attempt / "stage2" / "complete.json").write_text(json.dumps(marker))
    with pytest.raises(ReevaluationEvidenceError, match="adapter path escapes"):
        resolve_source_evidence(
            CONFIG_PATH, attempt, cell_id="e0_q0", city="porto", seed=42,
            budget="controlled", expected_revision=REVISION,
        )


def test_subset_and_population_audit_reject_hash_or_lens_mismatch(tmp_path: Path) -> None:
    attempt, aligned = _source_attempt(tmp_path)
    evidence = resolve_source_evidence(
        CONFIG_PATH, attempt, cell_id="e0_q0", city="porto", seed=42,
        budget="controlled", expected_revision=REVISION,
    )
    path = _subset(tmp_path)
    payload = json.loads(path.read_text())
    payload["selected_traj_id_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ReevaluationEvidenceError, match="trajectory hash mismatch"):
        load_stratified_manifest(path, city="porto", seed=42, expected_size=2)

    path = _subset(tmp_path)
    subset = load_stratified_manifest(path, city="porto", seed=42, expected_size=2)
    token_path = evidence.test_tokens
    rows = [json.loads(line) for line in token_path.read_text().splitlines()]
    rows[0]["lens_type"] = [1, 1, 1]
    token_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    marker = json.loads((attempt / "tokens" / "complete.json").read_text())
    info = marker["artifacts"]["test_tokens.jsonl"]
    info["size_bytes"] = token_path.stat().st_size
    info["sha256"] = hashlib.sha256(token_path.read_bytes()).hexdigest()
    (attempt / "tokens" / "complete.json").write_text(json.dumps(marker))
    changed = resolve_source_evidence(
        CONFIG_PATH, attempt, cell_id="e0_q0", city="porto", seed=42,
        budget="controlled", expected_revision=REVISION,
    )
    with pytest.raises(ReevaluationEvidenceError, match="lens_type mismatch"):
        audit_source_population(changed, subset, aligned)
