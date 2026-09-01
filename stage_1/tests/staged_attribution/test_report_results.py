from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from staged_attribution.analyze import CORE_CELLS, staged_contrast_definitions
from staged_attribution.report_results import (
    G_VARIANTS,
    METRICS,
    ResultAuditError,
    RunEvidence,
    collect_core_runs,
    read_stage_artifact,
    summarize_stage4,
    write_report,
)
from staged_attribution.config import ExperimentConfig
from staged_attribution.run_experiment import build_jobs
from staged_attribution.population_bootstrap import (
    POPULATION_REGRET_METHOD,
    SELECTION_BOOTSTRAP_METHOD,
)
from staged_attribution.immutable_artifacts import ImmutableArtifactError
from staged_attribution.select_candidates import (
    SelectionError,
    select_candidates,
    write_selection,
)


CONFIG_PATH = Path("stage_1/configs/staged_attribution/experiment.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_artifact_is_confined_and_hash_verified(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt_001"
    artifact = attempt / "stage4" / "stage4_metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}")
    marker = {
        "stage": "stage4",
        "artifacts": {
            "stage4_metrics.json": {
                "path": str(artifact.resolve()),
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            }
        },
    }
    (artifact.parent / "complete.json").write_text(json.dumps(marker))

    assert read_stage_artifact(attempt, "stage4", "stage4_metrics.json") == artifact.resolve()
    artifact.write_text("tampered")
    with pytest.raises(ResultAuditError, match="size mismatch|SHA256 mismatch"):
        read_stage_artifact(attempt, "stage4", "stage4_metrics.json")


def _write_completed_attempt(root: Path, job, revision: str) -> Path:
    attempt = root / job.run_key / "attempt_001"
    artifact = attempt / "stage4" / "stage4_metrics.json"
    artifact.parent.mkdir(parents=True)
    payload = {variant: _variant_payload(1.0, offset) for variant, offset in zip(G_VARIANTS, (0, .1, .2, .4))}
    artifact.write_text(json.dumps(payload))
    marker = {
        "stage": "stage4",
        "artifacts": {
            "stage4_metrics.json": {
                "path": str(artifact.resolve()),
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            }
        },
    }
    (attempt / "stage4" / "complete.json").write_text(json.dumps(marker))
    stage1_artifact = attempt / "stage1" / "stage1_metrics.json"
    stage1_artifact.parent.mkdir(parents=True)
    stage1_artifact.write_text(json.dumps(_stage1_metrics()))
    stage1_marker = {
        "stage": "stage1",
        "artifacts": {
            "stage1_metrics.json": {
                "path": str(stage1_artifact.resolve()),
                "size_bytes": stage1_artifact.stat().st_size,
                "sha256": _sha256(stage1_artifact),
            }
        },
    }
    (attempt / "stage1" / "complete.json").write_text(json.dumps(stage1_marker))
    manifest = {
        "experiment_id": "gps_road_grid_staged_attribution_v1_20260826",
        "cell_id": job.cell_id,
        "city": job.city,
        "seed": job.seed,
        "budget": job.budget,
        "status": "completed",
        "git": {"revision": revision, "dirty": False},
    }
    (attempt / "run_manifest.json").write_text(json.dumps(manifest))
    return attempt


def test_collect_core_runs_requires_complete_same_revision_cartesian_evidence(tmp_path: Path) -> None:
    config = ExperimentConfig.load(CONFIG_PATH)
    jobs = build_jobs(config, "smoke")
    for job in jobs:
        _write_completed_attempt(tmp_path, job, "a" * 40)

    evidence = collect_core_runs(CONFIG_PATH, tmp_path, "smoke")

    assert len(evidence) == 10
    assert {run.cell_id for run in evidence} == {job.cell_id for job in jobs}
    changed = tmp_path / jobs[-1].run_key / "attempt_001" / "run_manifest.json"
    manifest = json.loads(changed.read_text())
    manifest["git"]["revision"] = "b" * 40
    changed.write_text(json.dumps(manifest))
    with pytest.raises(ResultAuditError, match="incomparable revisions"):
        collect_core_runs(CONFIG_PATH, tmp_path, "smoke")


def test_confirmation_collection_uses_the_same_explicit_selection(tmp_path: Path) -> None:
    config = ExperimentConfig.load(CONFIG_PATH)
    selected = ["e0_q0", "e6_q3"]
    jobs = build_jobs(config, "confirmation", selected)
    for job in jobs:
        _write_completed_attempt(tmp_path, job, "c" * 40)

    evidence = collect_core_runs(CONFIG_PATH, tmp_path, "confirmation", selected)

    assert len(evidence) == 12
    assert {run.cell_id for run in evidence} == set(selected)


def _variant_payload(base: float, offset: float) -> dict[str, object]:
    return {
        "population": 2,
        "successful": 2,
        "failed": 0,
        "validity_rate": 1.0,
        "failure_reasons": {},
        "population_mean_penalized_error": base + offset,
        "successful_only": {
            "population": 2,
            "metrics": {
                "travel_distance_jsd": base / 100 + offset,
                "segment_distance_jsd": base / 100 + offset,
                "radius_jsd": base / 100 + offset,
                "grid_density_cosine": 0.9 - base / 1000 - offset,
                "grid_pattern_f1": 0.8 - base / 1000 - offset,
                "road_density_cosine": 0.9 - base / 1000 - offset,
                "road_pattern_f1": 0.8 - base / 1000 - offset,
                "point_to_road_jsd": base / 100 + offset,
            },
        },
        "per_trajectory": [
            {"traj_id": 1, "valid": True, "penalized_error": base + offset - 0.1},
            {"traj_id": 2, "valid": True, "penalized_error": base + offset + 0.1},
        ],
    }


def _stage1_metrics() -> dict[str, object]:
    return {
        "losses": {"loss": 1.0, "base_loss": 1.0, "quant_loss": 0.1, "reconstruction_loss": 0.9},
        "codebook": [
            {"active_fraction": 0.2, "perplexity_fraction": 0.2, "max_share": 0.2}
            for _ in range(4)
        ],
    }


def _run(cell_id: str, base: float) -> RunEvidence:
    offsets = {"g00": 0.0, "g10": 0.1, "g01": 0.2, "g11": 0.4}
    stage4 = {variant: _variant_payload(base, offset) for variant, offset in offsets.items()}
    return RunEvidence(
        run_key=f"smoke/porto/seed_42/{cell_id}",
        cell_id=cell_id,
        city="porto",
        seed=42,
        budget="smoke",
        attempt=Path(f"/tmp/{cell_id}"),
        revision="a" * 40,
        stage4=stage4,
        stage1=_stage1_metrics(),
    )


def _corrected_selection_summary(summary: dict[str, object]) -> dict[str, object]:
    corrected = dict(summary)
    corrected["analysis_schema_version"] = 5
    corrected["candidate_evidence"] = "stratified_population_aware"
    corrected["metric_rows"] = [
        {**row, "method": POPULATION_REGRET_METHOD}
        for row in summary["metric_rows"]
    ]
    lookup = {
        (row["city"], row["seed"], row["cell_id"], row["metric"]): row["regret"]
        for row in corrected["metric_rows"] if row["variant"] == "g11"
    }
    effects = []
    for city in sorted({key[0] for key in lookup}):
        for cell in sorted({key[2] for key in lookup} - {"e0_q0"}):
            for metric, (_, group) in METRICS.items():
                values = [
                    lookup[(city, seed, cell, metric)] - lookup[(city, seed, "e0_q0", metric)]
                    for seed in sorted({key[1] for key in lookup if key[0] == city})
                ]
                estimate = sum(values) / len(values)
                effects.append(
                    {
                        "reference": "e0_q0", "city": city, "cell_id": cell,
                        "metric": metric, "group": group,
                        "method": SELECTION_BOOTSTRAP_METHOD,
                        "n_pairs": 2 * len(values), "n_resamples": 100,
                        "estimate": estimate, "ci_low": estimate, "ci_high": estimate,
                        "p_improvement": float(estimate < 0),
                        "p_degradation": float(estimate > 0),
                    }
                )
    corrected["paired_metric_effects"] = effects
    definitions = staged_contrast_definitions()
    definition_by_name = {
        definition["contrast"]: definition for definition in definitions
    }
    staged_rows = []
    for source in summary["e_q_contrasts"]:
        definition = definition_by_name[source["contrast"]]
        row = {
            "family": source["family"],
            "contrast": source["contrast"],
            "contrast_kind": definition["kind"],
            "inferential_role": definition["role"],
            "city": "porto",
            "metric": source["metric"],
            "group": METRICS[source["metric"]][1],
            "orientation": "minimize",
            "method": SELECTION_BOOTSTRAP_METHOD,
            "seed_strata": [42],
            "seed_populations": {42: 2},
            "n_pairs": 2,
            "estimate": source["estimate"],
            "ci_low": source["estimate"],
            "ci_high": source["estimate"],
            "p_improvement": float(source["estimate"] < 0),
            "p_degradation": float(source["estimate"] > 0),
            "seed": 20260826,
            "n_resamples": 100,
        }
        if definition["kind"] == "linear":
            row["coefficients"] = definition["coefficients"]
        else:
            row["expression"] = definition["expression"]
        staged_rows.append(row)
    corrected["staged_attribution"] = {
        "available": True,
        "required_cells": list(CORE_CELLS),
        "observed_cells": list(CORE_CELLS),
        "missing_cells": [],
        "method": SELECTION_BOOTSTRAP_METHOD,
        "unit": "trajectory_within_seed_equal_weight_across_seeds",
        "contrast_definitions": definitions,
        "rows": staged_rows,
        "city_direction_heterogeneity": [],
    }
    g_rows = []
    for source in summary["g_shapley"]:
        for component in ("sid", "sampling", "interaction", "total"):
            estimate = source[component]
            g_rows.append(
                {
                    "city": source["city"],
                    "cell_id": source["cell_id"],
                    "metric": source["metric"],
                    "component": component,
                    "group": METRICS[source["metric"]][1],
                    "orientation": "minimize",
                    "inferential_role": "primary",
                    "method": SELECTION_BOOTSTRAP_METHOD,
                    "seed_strata": [42],
                    "seed_populations": {42: 2},
                    "n_pairs": 2,
                    "estimate": estimate,
                    "ci_low": estimate,
                    "ci_high": estimate,
                    "p_improvement": float(estimate < 0),
                    "p_degradation": float(estimate > 0),
                    "seed": 20260826,
                    "n_resamples": 100,
                }
            )
    corrected["g_attribution"] = {
        "available": True,
        "required_variants": ["g00", "g10", "g01", "g11"],
        "components": ["sid", "sampling", "interaction", "total"],
        "observed_cells": list(CORE_CELLS),
        "method": SELECTION_BOOTSTRAP_METHOD,
        "unit": "trajectory_within_seed_equal_weight_across_seeds",
        "rows": g_rows,
        "city_direction_heterogeneity": [],
    }
    return corrected


def test_summary_freezes_g_e_q_and_paired_trajectory_attribution() -> None:
    bases = {
        "e0_q0": 10.0, "e1_q0": 9.0, "e2_q0": 8.0, "e3_q0": 7.0,
        "e4_q0": 6.0, "e5_q0": 5.0, "e6_q0": 4.0,
        "e6_q1": 3.5, "e6_q2": 3.0, "e6_q3": 2.0,
    }
    summary = summarize_stage4([_run(cell, base) for cell, base in bases.items()], n_resamples=100)

    assert summary["coverage"]["runs"] == 10
    assert len(summary["stage1_gates"]) == 10
    assert all(row["passed"] for row in summary["stage1_gates"])
    e_rows = {
        row["contrast"]: row["estimate"]
        for row in summary["e_q_contrasts"]
        if row["metric"] == "population_mean_penalized_error" and row["family"] == "E"
    }
    assert e_rows["gps_road_given_grid"] == pytest.approx(1.0)
    assert e_rows["gps_grid_given_road"] == pytest.approx(1.0)
    assert e_rows["road_grid_given_gps"] == pytest.approx(1.0)

    g = next(
        row for row in summary["g_shapley"]
        if row["cell_id"] == "e0_q0" and row["metric"] == "population_mean_penalized_error"
    )
    assert g["sid"] == pytest.approx(0.15)
    assert g["sampling"] == pytest.approx(0.25)
    assert g["interaction"] == pytest.approx(0.1)
    assert g["sid"] + g["sampling"] == pytest.approx(g["total"])
    assert len(summary["paired_e_q_penalized_error"]) == 9
    assert {row["component"] for row in summary["g_trajectory_bootstrap"]} == {
        "sid", "sampling", "interaction", "total"
    }

    with pytest.raises(SelectionError, match="schema version 5"):
        select_candidates(summary, phase="core")
    corrected = _corrected_selection_summary(summary)
    selection = select_candidates(corrected, phase="core")
    assert selection["best_single"] == "e2_q0"
    assert selection["best_pair"] == "e5_q0"
    assert selection["best_q"] == "e6_q3"
    assert selection["adaptive_cells"] == ["e2_q3", "e5_q3"]
    assert selection["confirmation_cells"] == ["e0_q0"]

    final = select_candidates(corrected, phase="final")
    assert final["confirmation_cells"][0] == "e0_q0"
    assert "e6_q3" in final["confirmation_cells"]


def test_candidate_selection_is_immutable_and_idempotent(tmp_path: Path) -> None:
    selection = {"schema_version": 1, "phase": "core", "adaptive_cells": []}
    output = tmp_path / "selection.json"
    assert write_selection(selection, output) == output
    original = output.read_bytes()
    assert write_selection(selection, output) == output

    changed = dict(selection)
    changed["phase"] = "final"
    with pytest.raises(ImmutableArtifactError, match="refusing to replace"):
        write_selection(changed, output)
    assert output.read_bytes() == original


def test_core_analysis_report_is_immutable_and_idempotent(tmp_path: Path) -> None:
    summary = {
        "revision": "a" * 40,
        "coverage": {
            "runs": 1,
            "cities": ["porto"],
            "seeds": [42],
            "cells": ["e0_q0"],
        },
        "metric_rows": [],
        "g_shapley": [],
        "g_trajectory_bootstrap": [],
        "e_q_contrasts": [],
        "paired_e_q_penalized_error": [],
        "city_heterogeneity": [],
    }
    output = tmp_path / "core_report"
    json_path, markdown_path = write_report(summary, output)
    original_json = json_path.read_bytes()
    original_markdown = markdown_path.read_bytes()
    assert write_report(summary, output) == (json_path, markdown_path)

    changed = dict(summary)
    changed["revision"] = "b" * 40
    with pytest.raises(ImmutableArtifactError, match="refusing to replace"):
        write_report(changed, output)
    assert json_path.read_bytes() == original_json
    assert markdown_path.read_bytes() == original_markdown


def test_summary_rejects_inconsistent_g_population() -> None:
    run = _run("e0_q0", 1.0)
    run.stage4["g11"]["per_trajectory"][1]["traj_id"] = 3

    with pytest.raises(ResultAuditError, match="same population"):
        summarize_stage4([run], n_resamples=10)
