from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from staged_attribution.immutable_artifacts import ImmutableArtifactError
from staged_attribution.config import ExperimentConfig
from staged_attribution.population_bootstrap import (
    POPULATION_REGRET_METHOD,
    build_metric_contributions,
    population_regrets,
    save_metric_contributions,
)
from staged_attribution.report_confirmation_results import (
    ANALYSIS_SCHEMA_VERSION,
    CONFIRMATION_EVIDENCE,
    audit_confirmation_selection,
    collect_confirmation_runs,
    summarize_confirmation_results,
    write_confirmation_report,
)
from staged_attribution.report_results import ResultAuditError
from staged_attribution.run_experiment import build_jobs


REVISION = "a" * 40
EXPERIMENT_ID = "gps_road_grid_staged_attribution_v1_20260826"
CELLS = ("e0_q0", "e1_q0")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _traj_id_sha256(ids: list[int]) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def _write_marker(attempt: Path, stage: str, artifacts: dict[str, Path]) -> None:
    marker = {
        "stage": stage,
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in artifacts.items()
        },
    }
    path = attempt / stage / "complete.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker), encoding="utf-8")


def _geometries() -> dict[int, np.ndarray]:
    return {
        0: np.asarray([[0.0, 0.0], [0.01, 0.0]]),
        1: np.asarray([[0.0, 0.01], [0.01, 0.01]]),
    }


def _contributions(failed: bool):
    real_gps = {
        traj_id: np.asarray(
            [
                [0.001, 0.001 + 0.002 * (traj_id - 1)],
                [0.004, 0.001 + 0.002 * (traj_id - 1)],
                [0.007, 0.001 + 0.002 * (traj_id - 1)],
            ]
        )
        for traj_id in range(1, 5)
    }
    real_roads = {
        traj_id: [0, 0, 0] if traj_id <= 2 else [1, 1, 1]
        for traj_id in real_gps
    }
    generated = []
    for traj_id in real_gps:
        if failed and traj_id != 1:
            generated.append(
                {
                    "traj_id": traj_id,
                    "status": "failed",
                    "failure": {"reason": "syntax"},
                }
            )
        else:
            generated.append(
                {
                    "traj_id": traj_id,
                    "status": "reconstructed",
                    "gps_traj": real_gps[traj_id].tolist(),
                    "road_traj": real_roads[traj_id],
                }
            )
    return build_metric_contributions(
        real_gps,
        real_roads,
        generated,
        _geometries(),
        "porto",
        mbr={"min_lon": 0.0, "max_lon": 0.02, "min_lat": 0.0, "max_lat": 0.02},
        grid={
            "grid_size_m": 100.0,
            "meters_per_lon": 111320.0,
            "meters_per_lat": 110540.0,
            "n_grid_x": 23,
            "n_grid_y": 23,
        },
    )


def _stage4_payload(contribution) -> dict[str, object]:
    penalties = [0.1 if valid else 2.0 for valid in contribution.valid]
    return {
        "population": len(contribution.traj_ids),
        "population_mean_penalized_error": sum(penalties) / len(penalties),
        "per_trajectory": [
            {
                "traj_id": traj_id,
                "valid": bool(valid),
                "penalized_error": penalty,
            }
            for traj_id, valid, penalty in zip(
                contribution.traj_ids, contribution.valid, penalties, strict=True
            )
        ],
        "population_aware_regret": {
            "method": POPULATION_REGRET_METHOD,
            "metrics": {
                metric: float(values[0])
                for metric, values in population_regrets(contribution).items()
            },
        },
    }


def _test_config(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    source = Path("stage_1/configs/staged_attribution/experiment.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    manifests = {}
    for city in payload["cities"]:
        ids = list(range(1, 64_001))
        path = tmp_path / f"{city}_stage2_train_64000_seed20260826.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "frozen_stage2_train_subset",
                    "method": "sha256_rank_fixed_city_v1",
                    "requested": 64_000,
                    "subset_seed": 20260826,
                    "city": city,
                    "selection_depends_on_cell": False,
                    "selection_depends_on_training_seed": False,
                    "selected": {
                        "population": len(ids),
                        "traj_id_sha256": _traj_id_sha256(ids),
                        "traj_ids": ids,
                    },
                }
            )
        )
        manifests[city] = path.resolve()
    registry = {city: str(path) for city, path in manifests.items()}
    for budget in ("controlled", "confirmation"):
        payload["budgets"][budget]["stage2_subset_manifests"] = registry
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    ExperimentConfig.load(path)
    return path, manifests


def _write_attempt(
    root: Path,
    config: ExperimentConfig,
    manifests: dict[str, Path],
    job,
) -> Path:
    attempt = root / job.run_key / "attempt_001"
    attempt.mkdir(parents=True)
    (attempt / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": config.experiment_id,
                "cell_id": job.cell_id,
                "city": job.city,
                "seed": job.seed,
                "budget": "confirmation",
                "status": "completed",
                "git": {"revision": REVISION, "dirty": False},
            }
        ),
        encoding="utf-8",
    )

    stage1_path = attempt / "stage1" / "stage1_metrics.json"
    stage1_path.parent.mkdir(parents=True)
    stage1_path.write_text(
        json.dumps(
            {
                "losses": {"loss": 1.0, "base_loss": 1.0},
                "codebook": [
                    {
                        "active_fraction": 0.2,
                        "perplexity_fraction": 0.2,
                        "max_share": 0.2,
                    }
                    for _ in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_marker(attempt, "stage1", {"stage1_metrics.json": stage1_path})

    train_ids = list(range(1, 64_001))
    generation_ids = [1, 2, 3, 4]
    budget = config.budgets["confirmation"]
    stage2_path = attempt / "stage2_data" / "subset_manifest.json"
    stage2_path.parent.mkdir(parents=True)
    stage2_path.write_text(
        json.dumps(
            {
                "train": {
                    "population": len(train_ids),
                    "selection": {
                        "method": "frozen_manifest_sha256_rank_fixed_city_v1",
                        "population": budget.stage2_train_trajectories,
                        "requested": budget.stage2_train_trajectories,
                        "subset_seed": budget.stage2_subset_seed,
                        "city": job.city,
                        "selection_depends_on_training_seed": False,
                        "manifest_path": str(manifests[job.city]),
                        "manifest_sha256": _sha256(manifests[job.city]),
                        "selected_traj_id_sha256": _traj_id_sha256(train_ids),
                        "selected_traj_ids": train_ids,
                    }
                },
                "generation": {"population": len(generation_ids), "traj_ids": generation_ids},
            }
        ),
        encoding="utf-8",
    )
    _write_marker(
        attempt, "stage2_data", {"subset_manifest.json": stage2_path}
    )

    failed = job.cell_id == "e0_q0"
    contribution = _contributions(failed)
    stage4_path = attempt / "stage4" / "stage4_metrics.json"
    stage4_path.parent.mkdir(parents=True)
    stage4_path.write_text(
        json.dumps(
            {
                variant: _stage4_payload(contribution)
                for variant in ("g00", "g10", "g01", "g11")
            }
        ),
        encoding="utf-8",
    )
    artifacts = {"stage4_metrics.json": stage4_path}
    for variant in ("g00", "g10", "g01", "g11"):
        manifest = save_metric_contributions(
            contribution, attempt / "stage4" / f"{variant}_contributions"
        )
        artifacts[f"{variant}_contributions_manifest.json"] = manifest
    _write_marker(attempt, "stage4", artifacts)
    return attempt


def _selection(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    analysis = tmp_path / "selection_source.json"
    analysis.write_text(
        json.dumps(
            {
                "candidate_evidence": "stratified_population_aware",
                "revision": REVISION,
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "final_selection.json"
    selection.write_text(
        json.dumps(
            {
                "phase": "final",
                "confirmation_cells": list(CELLS),
                "source_revision": REVISION,
                "source_analysis": str(analysis.resolve()),
                "source_analysis_sha256": _sha256(analysis),
            }
        ),
        encoding="utf-8",
    )
    return selection, audit_confirmation_selection(selection, CELLS)


def _evidence_tree(tmp_path: Path):
    config_path, manifests = _test_config(tmp_path)
    config = ExperimentConfig.load(config_path)
    root = tmp_path / "confirmation"
    attempts = [
        _write_attempt(root, config, manifests, job)
        for job in build_jobs(config, "confirmation", CELLS)
    ]
    selection_path, selection_audit = _selection(tmp_path)
    return config_path, root, attempts, selection_path, selection_audit


def test_confirmation_collection_and_summary_are_full_population_audited(
    tmp_path: Path,
) -> None:
    config_path, root, _, _, selection = _evidence_tree(tmp_path)

    runs = collect_confirmation_runs(config_path, root, CELLS)
    assert len(runs) == 12
    assert {run.city for run in runs} == {"porto", "chengdu"}
    assert {run.seed for run in runs} == {42, 314159, 20260819}
    assert {run.evaluation_population for run in runs} == {4}

    summary = summarize_confirmation_results(
        runs, selection, n_resamples=500, bootstrap_seed=20260826
    )
    assert summary["analysis_schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert summary["candidate_evidence"] == CONFIRMATION_EVIDENCE
    assert summary["coverage"] == {
        "runs": 12,
        "cities": ["chengdu", "porto"],
        "seeds": [42, 314159, 20260819],
        "cells": ["e0_q0", "e1_q0"],
    }
    assert len(summary["paired_metric_effects"]) == 20
    assert summary["selection_audit"]["verified"] is True
    assert summary["run_revision_audit"] == {
        "uniform": True,
        "revision": REVISION,
    }
    assert summary["confirmation_gate"]["e1_q0"]["claim_supported"] is True
    assert not summary["confirmation_gate"]["e1_q0"]["significant_degradation"]
    assert len(summary["population_identity_by_city"]) == 2

    json_path, markdown_path = write_confirmation_report(
        summary, tmp_path / "report"
    )
    assert json.loads(json_path.read_text())["candidate_evidence"] == CONFIRMATION_EVIDENCE
    assert "claim_supported=true" in markdown_path.read_text()

    original_json = json_path.read_bytes()
    original_markdown = markdown_path.read_bytes()
    assert write_confirmation_report(summary, tmp_path / "report") == (
        json_path,
        markdown_path,
    )
    changed = dict(summary)
    changed["revision"] = "f" * 40
    with pytest.raises(ImmutableArtifactError, match="refusing to replace"):
        write_confirmation_report(changed, tmp_path / "report")
    assert json_path.read_bytes() == original_json
    assert markdown_path.read_bytes() == original_markdown


def test_confirmation_collection_rejects_nested_contribution_tamper(
    tmp_path: Path,
) -> None:
    config_path, root, attempts, _, _ = _evidence_tree(tmp_path)
    tampered = attempts[0] / "stage4" / "g11_contributions" / "histograms.npz"
    tampered.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        collect_confirmation_runs(config_path, root, CELLS)


def test_confirmation_selection_requires_exact_hashed_final_source(
    tmp_path: Path,
) -> None:
    _, _, _, selection_path, _ = _evidence_tree(tmp_path)
    with pytest.raises(ResultAuditError, match="exactly match"):
        audit_confirmation_selection(selection_path, ("e0_q0",))

    payload = json.loads(selection_path.read_text())
    payload["source_analysis_sha256"] = "0" * 64
    selection_path.write_text(json.dumps(payload))
    with pytest.raises(ResultAuditError, match="hash-mismatched"):
        audit_confirmation_selection(selection_path, CELLS)


def test_confirmation_collection_rejects_nonfull_generation_identity(
    tmp_path: Path,
) -> None:
    config_path, root, attempts, _, _ = _evidence_tree(tmp_path)
    stage2 = attempts[0] / "stage2_data" / "subset_manifest.json"
    payload = json.loads(stage2.read_text())
    payload["generation"]["traj_ids"] = [1, 3, 2, 4]
    stage2.write_text(json.dumps(payload))
    _write_marker(attempts[0], "stage2_data", {"subset_manifest.json": stage2})

    with pytest.raises(ResultAuditError, match="full, sorted, and unique"):
        collect_confirmation_runs(config_path, root, CELLS)


def test_confirmation_collection_rejects_wrapper_subset_identity_drift(
    tmp_path: Path,
) -> None:
    config_path, root, attempts, _, _ = _evidence_tree(tmp_path)
    stage2 = attempts[0] / "stage2_data" / "subset_manifest.json"
    payload = json.loads(stage2.read_text())
    payload["train"]["selection"]["selected_traj_ids"][-1] = 64_001
    payload["train"]["selection"]["selected_traj_id_sha256"] = _traj_id_sha256(
        payload["train"]["selection"]["selected_traj_ids"]
    )
    stage2.write_text(json.dumps(payload))
    _write_marker(attempts[0], "stage2_data", {"subset_manifest.json": stage2})

    with pytest.raises(ResultAuditError, match="Option B identity mismatch"):
        collect_confirmation_runs(config_path, root, CELLS)
