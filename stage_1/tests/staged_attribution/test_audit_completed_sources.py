from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import staged_attribution.audit_completed_sources as audit_module
from staged_attribution.audit_completed_sources import (
    CompletedSourceAuditError,
    audit_completed_sources,
)
from staged_attribution.run_cell import StageNode, write_stage_marker
from staged_attribution.stratified_subset import stratified_subset


REVISION = "a" * 40


def _write_stage4(attempt: Path, ids: list[int]) -> None:
    path = attempt / "stage4" / "stage4_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        variant: {
            "population": len(ids),
            "per_trajectory": [{"traj_id": value} for value in ids],
        }
        for variant in ("g00", "g10", "g01", "g11")
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    write_stage_marker(
        attempt,
        StageNode("stage4", (), ("stage4_metrics.json",)),
        {"stage4_metrics.json": path},
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    np.save(aligned / "test_traj_ids.npy", np.asarray([1, 2], dtype=np.int64))
    np.save(aligned / "test_offsets.npy", np.asarray([0, 20, 41], dtype=np.int64))

    source_config = Path("stage_1/configs/staged_attribution/experiment.json")
    config_payload = json.loads(source_config.read_text(encoding="utf-8"))
    config_payload["paths"]["porto_aligned_root"] = str(aligned.resolve())
    config_payload["budgets"]["controlled"]["stage3_trajectories"] = 2
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    root = tmp_path / "sources"
    attempt = root / "controlled/porto/seed_42/e0_q0/attempt_001"
    stage1_model = attempt / "stage1/stage1_model.pt"
    test_tokens = attempt / "tokens/test_tokens.jsonl"
    llm_test = attempt / "stage2_data/llm_test.parquet"
    adapter_manifest = attempt / "stage2/adapter_manifest.json"
    for path in (stage1_model, test_tokens, llm_test, adapter_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    stage1_model.write_bytes(b"model")
    test_tokens.write_text(
        json.dumps({"traj_id": 1, "lens_type": [0, 0, 1]})
        + "\n"
        + json.dumps({"traj_id": 2, "lens_type": [1, 1, 0]})
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame({"traj_id": [1, 2]}).to_parquet(llm_test, index=False)
    adapter = attempt / "stage2/adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    adapter_manifest.write_text(
        json.dumps({"adapter": str(adapter.resolve())}), encoding="utf-8"
    )
    stages = {
        "stage1": {"stage1_model.pt": stage1_model},
        "tokens": {"test_tokens.jsonl": test_tokens},
        "stage2_data": {"llm_test.parquet": llm_test},
        "stage2": {"adapter_manifest.json": adapter_manifest},
    }
    for stage, artifacts in stages.items():
        write_stage_marker(
            attempt,
            StageNode(stage, (), tuple(artifacts)),
            artifacts,
        )
    _write_stage4(attempt, [1, 2])
    (attempt / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": config_payload["experiment_id"],
                "cell_id": "e0_q0",
                "city": "porto",
                "seed": 42,
                "budget": "controlled",
                "status": "completed",
                "git": {"revision": REVISION, "dirty": False},
            }
        ),
        encoding="utf-8",
    )

    subset_root = tmp_path / "subsets"
    subset_root.mkdir()
    selected = stratified_subset(
        [1, 2], [20, 21], 2, 42, require_all_registered_strata=False
    )
    subset = {
        **selected,
        "schema_version": 1,
        "city": "porto",
        "selection_depends_on_cell": False,
    }
    (subset_root / "controlled_porto_seed_42.json").write_text(
        json.dumps(subset), encoding="utf-8"
    )
    (subset_root / "index.json").write_text(
        json.dumps({"kind": "fixed_stratified_reevaluation_subsets"}),
        encoding="utf-8",
    )
    return config_path, root, subset_root


def _clean_git(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    state = {
        "worktree": "/synthetic/worktree",
        "revision": "b" * 40,
        "dirty": False,
    }
    monkeypatch.setattr(audit_module, "_git_state", lambda: state)
    return state


def test_completed_source_audit_is_exact_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, subsets = _fixture(tmp_path)
    git_state = _clean_git(monkeypatch)
    output = tmp_path / "audit/index.json"
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()

    first = audit_completed_sources(
        config,
        root,
        subsets,
        output_path=output,
        expected_source_revision=REVISION,
        expected_completed_count=1,
        expected_generator_revision=str(git_state["revision"]),
        expected_config_sha256=config_sha,
        require_clean=True,
    )
    second = audit_completed_sources(
        config,
        root,
        subsets,
        output_path=output,
        expected_source_revision=REVISION,
        expected_completed_count=1,
        expected_generator_revision=str(git_state["revision"]),
        expected_config_sha256=config_sha,
        require_clean=True,
    )

    assert first == second
    assert first["audited_completed_source_count"] == 1
    assert first["missing_completed_source_count"] == 59
    assert first["all_formal_core_sources_completed"] is False
    row = first["audited_completed_sources"][0]
    assert row["source_artifact_markers_verified"] is True
    assert row["stage4_population_audit"][
        "g_variants_use_exact_fixed_population"
    ] is True
    assert row["stage4_population_audit"]["traj_id_sha256"] == row[
        "fixed_subset"
    ]["selected_traj_id_sha256"]
    assert len(first["output_sha256"]) == 64

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CompletedSourceAuditError, match="refusing to replace"):
        audit_completed_sources(
            config,
            root,
            subsets,
            output_path=output,
            expected_source_revision=REVISION,
            expected_completed_count=1,
        )


def test_completed_source_audit_rejects_stage4_population_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, subsets = _fixture(tmp_path)
    _clean_git(monkeypatch)
    attempt = root / "controlled/porto/seed_42/e0_q0/attempt_001"
    _write_stage4(attempt, [1, 3])

    with pytest.raises(CompletedSourceAuditError, match="differs from the fixed subset"):
        audit_completed_sources(
            config,
            root,
            subsets,
            expected_source_revision=REVISION,
            expected_completed_count=1,
        )


def test_completed_source_audit_enforces_count_and_generator_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, subsets = _fixture(tmp_path)
    state = _clean_git(monkeypatch)

    with pytest.raises(CompletedSourceAuditError, match="count mismatch"):
        audit_completed_sources(
            config,
            root,
            subsets,
            expected_source_revision=REVISION,
            expected_completed_count=2,
        )
    with pytest.raises(CompletedSourceAuditError, match="generator revision"):
        audit_completed_sources(
            config,
            root,
            subsets,
            expected_generator_revision="c" * 40,
        )
    with pytest.raises(CompletedSourceAuditError, match="config SHA-256"):
        audit_completed_sources(
            config,
            root,
            subsets,
            expected_config_sha256="0" * 64,
        )
    monkeypatch.setattr(
        audit_module,
        "_git_state",
        lambda: {**state, "dirty": True},
    )
    with pytest.raises(CompletedSourceAuditError, match="must be clean"):
        audit_completed_sources(
            config,
            root,
            subsets,
            require_clean=True,
        )
