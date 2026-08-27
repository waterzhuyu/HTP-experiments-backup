from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from staged_attribution.config import ExperimentConfig
from staged_attribution.population_bootstrap import (
    build_metric_contributions,
    save_metric_contributions,
)
from staged_attribution.report_stratified_results import (
    CANDIDATE_EVIDENCE,
    CorrectedRunEvidence,
    collect_corrected_runs,
    summarize_corrected_results,
)
from staged_attribution.report_results import ResultAuditError
from staged_attribution.run_experiment import build_jobs
from staged_attribution.select_candidates import select_candidates


def _geometries() -> dict[int, np.ndarray]:
    return {
        0: np.asarray([[0.0, 0.0], [0.01, 0.0]]),
        1: np.asarray([[0.0, 0.01], [0.01, 0.01]]),
    }


def _contributions(failed: bool):
    real_gps = {
        1: np.asarray([[0.001, 0.001], [0.004, 0.001], [0.007, 0.001]]),
        2: np.asarray([[0.001, 0.009], [0.004, 0.009], [0.007, 0.009]]),
    }
    real_roads = {1: [0, 0, 0], 2: [1, 1, 1]}
    generated = [
        {
            "traj_id": 1,
            "status": "reconstructed",
            "gps_traj": real_gps[1].tolist(),
            "road_traj": [0, 0, 0],
        },
        {
            "traj_id": 2,
            "status": "failed" if failed else "reconstructed",
            **(
                {"failure": {"reason": "syntax"}}
                if failed
                else {"gps_traj": real_gps[2].tolist(), "road_traj": [1, 1, 1]}
            ),
        },
    ]
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


def _payload(failed: bool) -> dict[str, object]:
    penalties = [0.1, 2.0 if failed else 0.1]
    return {
        "population": 2,
        "population_mean_penalized_error": sum(penalties) / 2,
        "per_trajectory": [
            {"traj_id": 1, "valid": True, "penalized_error": penalties[0]},
            {"traj_id": 2, "valid": not failed, "penalized_error": penalties[1]},
        ],
    }


def _run(cell: str, seed: int, failed: bool) -> CorrectedRunEvidence:
    perfect = _contributions(False)
    degraded = _contributions(failed)
    variants = {"g00": perfect, "g10": degraded, "g01": perfect, "g11": degraded}
    payloads = {name: _payload(failed and name in {"g10", "g11"}) for name in variants}
    selected_hash = hashlib.sha256(b"1\n2").hexdigest()
    return CorrectedRunEvidence(
        run_key=f"controlled/porto/seed_{seed}/{cell}",
        cell_id=cell,
        city="porto",
        seed=seed,
        attempt=Path(f"/tmp/reevaluation/{seed}/{cell}"),
        evaluation_revision="b" * 40,
        source_attempt=Path(f"/tmp/source/{seed}/{cell}"),
        source_revision="a" * 40,
        subset_sha256="c" * 64,
        selected_traj_id_sha256=selected_hash,
        stage1={
            "losses": {"loss": 1.0, "base_loss": 1.0},
            "codebook": [
                {"active_fraction": 0.2, "perplexity_fraction": 0.2, "max_share": 0.2}
                for _ in range(4)
            ],
        },
        stage4=payloads,
        contributions=variants,
    )


def test_corrected_summary_is_selection_ready_and_seed_stratified() -> None:
    runs = [
        _run("e0_q0", 42, False),
        _run("e1_q0", 42, True),
        _run("e0_q0", 314159, False),
        _run("e1_q0", 314159, False),
    ]
    summary = summarize_corrected_results(
        runs, n_resamples=100, bootstrap_seed=20260826
    )

    assert summary["analysis_schema_version"] == 2
    assert summary["candidate_evidence"] == CANDIDATE_EVIDENCE
    assert summary["coverage"] == {
        "runs": 4,
        "cities": ["porto"],
        "seeds": [42, 314159],
        "cells": ["e0_q0", "e1_q0"],
    }
    assert len(summary["metric_rows"]) == 4 * 4 * 10
    assert len(summary["g_metric_attribution"]) == 4 * 10 * 4
    assert len(summary["paired_metric_effects"]) == 10
    validity = next(
        row for row in summary["paired_metric_effects"]
        if row["metric"] == "validity_rate"
    )
    assert validity["estimate"] == pytest.approx(0.25)
    assert validity["seed_strata"] == [42, 314159]
    penalty = next(
        row for row in summary["paired_metric_effects"]
        if row["metric"] == "population_mean_penalized_error"
    )
    assert penalty["estimate"] == pytest.approx(0.475)
    assert all(row["passed"] for row in summary["stage1_gates"])

    selection = select_candidates(summary, phase="core")
    assert selection["best_single"] == "e0_q0"
    assert selection["paired_effects"] == summary["paired_metric_effects"]


def test_corrected_summary_rejects_incomplete_cartesian_coverage() -> None:
    runs = [
        _run("e0_q0", 42, False),
        _run("e1_q0", 42, False),
        _run("e0_q0", 314159, False),
    ]
    with pytest.raises(ResultAuditError, match="Cartesian coverage"):
        summarize_corrected_results(runs, n_resamples=10)


def _write_marker(attempt: Path, stage: str, artifacts: dict[str, Path]) -> None:
    marker = {
        "stage": stage,
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in artifacts.items()
        },
    }
    marker_path = attempt / stage / "complete.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker))


def _write_collector_evidence(
    source_root: Path, reevaluation_root: Path, job
) -> Path:
    source = source_root / job.run_key / "attempt_001"
    source.mkdir(parents=True)
    (source / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "gps_road_grid_staged_attribution_v1_20260826",
                "cell_id": job.cell_id,
                "city": job.city,
                "seed": job.seed,
                "budget": "controlled",
                "status": "completed",
                "git": {"revision": "a" * 40, "dirty": False},
            }
        )
    )
    stage1 = source / "stage1" / "stage1_metrics.json"
    stage1.parent.mkdir(parents=True)
    stage1.write_text(
        json.dumps(
            {
                "losses": {"loss": 1.0},
                "codebook": [
                    {"active_fraction": 0.2, "perplexity_fraction": 0.2, "max_share": 0.2}
                    for _ in range(4)
                ],
            }
        )
    )
    _write_marker(source, "stage1", {"stage1_metrics.json": stage1})

    attempt = reevaluation_root / job.run_key / "attempt_001"
    attempt.mkdir(parents=True)
    (attempt / "reevaluation_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "gps_road_grid_staged_attribution_v1_20260826",
                "reevaluation": "fixed_stratified_population_aware_v1",
                "cell_id": job.cell_id,
                "city": job.city,
                "seed": job.seed,
                "budget": "controlled",
                "status": "completed",
                "source_attempt": str(source.resolve()),
                "git": {"revision": "b" * 40, "dirty": False},
            }
        )
    )
    selected_hash = hashlib.sha256(b"1\n2").hexdigest()
    reference = attempt / "source_reference" / "source_reference.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(
        json.dumps(
            {
                "source_attempt": str(source.resolve()),
                "source_revision": "a" * 40,
                "subset": {
                    "sha256": "c" * 64,
                    "selected_traj_id_sha256": selected_hash,
                },
            }
        )
    )
    _write_marker(attempt, "source_reference", {"source_reference.json": reference})
    stage4 = attempt / "stage4" / "stage4_metrics.json"
    stage4.parent.mkdir(parents=True)
    stage4.write_text(json.dumps({variant: _payload(False) for variant in ("g00", "g10", "g01", "g11")}))
    artifacts = {"stage4_metrics.json": stage4}
    contribution = _contributions(False)
    for variant in ("g00", "g10", "g01", "g11"):
        manifest = save_metric_contributions(
            contribution, attempt / "stage4" / f"{variant}_contributions"
        )
        artifacts[f"{variant}_contributions_manifest.json"] = manifest
    _write_marker(attempt, "stage4", artifacts)
    return attempt


def test_collector_verifies_complete_evidence_tree_and_rejects_tamper(tmp_path: Path) -> None:
    config_path = Path("stage_1/configs/staged_attribution/experiment.json")
    config = ExperimentConfig.load(config_path)
    source_root = tmp_path / "source"
    reevaluation_root = tmp_path / "reevaluation"
    attempts = [
        _write_collector_evidence(source_root, reevaluation_root, job)
        for job in build_jobs(config, "controlled", ["e0_q0"])
    ]

    runs = collect_corrected_runs(config_path, reevaluation_root, ["e0_q0"])
    assert len(runs) == 6
    assert {run.city for run in runs} == {"porto", "chengdu"}
    assert {run.seed for run in runs} == {42, 314159, 20260819}

    tampered = attempts[0] / "stage4" / "g11_contributions" / "histograms.npz"
    tampered.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        collect_corrected_runs(config_path, reevaluation_root, ["e0_q0"])
