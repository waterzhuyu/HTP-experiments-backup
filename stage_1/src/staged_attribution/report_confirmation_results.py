from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import ExperimentConfig
from .population_bootstrap import (
    POPULATION_REGRET_METHOD,
    PopulationMetricContributions,
    load_metric_contributions,
    population_regrets,
)
from .report_results import G_VARIANTS, METRICS, ResultAuditError, read_stage_artifact
from .report_stratified_results import summarize_corrected_results
from .run_experiment import ExperimentJob, build_jobs, completed_attempt


ANALYSIS_SCHEMA_VERSION = 3
CONFIRMATION_EVIDENCE = "confirmation_population_aware"
SELECTION_EVIDENCE = "stratified_population_aware"
ALLOWED_CLAIM_GROUPS = frozenset({"GPS", "Road", "Grid"})


@dataclass(frozen=True)
class ConfirmationRunEvidence:
    run_key: str
    cell_id: str
    city: str
    seed: int
    attempt: Path
    evaluation_revision: str
    source_attempt: Path
    source_revision: str
    subset_sha256: str
    selected_traj_id_sha256: str
    stage1: Mapping[str, object]
    stage4: Mapping[str, object]
    contributions: Mapping[str, PopulationMetricContributions]
    stage2_train_manifest_sha256: str
    stage2_train_traj_id_sha256: str
    evaluation_traj_id_sha256: str
    evaluation_population: int


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultAuditError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ResultAuditError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _traj_id_sha256(ids: Sequence[int]) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def audit_confirmation_selection(
    selection_path: str | Path,
    cell_ids: Sequence[str],
) -> dict[str, object]:
    selected = tuple(cell_ids)
    if not selected or len(set(selected)) != len(selected) or selected[0] != "e0_q0":
        raise ResultAuditError(
            "confirmation audit requires an explicit unique cell list beginning with e0_q0"
        )
    path = Path(selection_path).resolve()
    selection = _read_json(path, "final selection")
    declared = selection.get("confirmation_cells")
    if (
        selection.get("phase") != "final"
        or not isinstance(declared, list)
        or tuple(str(value) for value in declared) != selected
    ):
        raise ResultAuditError(
            "confirmation cells do not exactly match the frozen final selection"
        )
    source_revision = str(selection.get("source_revision", ""))
    if len(source_revision) != 40:
        raise ResultAuditError("final selection lacks a full source revision")
    source_path = Path(str(selection.get("source_analysis", ""))).resolve()
    source_sha = str(selection.get("source_analysis_sha256", ""))
    if not source_path.is_file() or len(source_sha) != 64 or _sha256(source_path) != source_sha:
        raise ResultAuditError("final selection source analysis is missing or hash-mismatched")
    source = _read_json(source_path, "final-selection source analysis")
    if (
        source.get("candidate_evidence") != SELECTION_EVIDENCE
        or source.get("revision") != source_revision
    ):
        raise ResultAuditError(
            "final selection does not reference audited stratified population-aware evidence"
        )
    return {
        "selection_path": str(path),
        "selection_sha256": _sha256(path),
        "phase": "final",
        "confirmation_cells": list(selected),
        "source_analysis": str(source_path),
        "source_analysis_sha256": source_sha,
        "source_revision": source_revision,
        "verified": True,
    }


def _stage2_identity(
    config: ExperimentConfig,
    attempt: Path,
    city: str,
) -> tuple[str, str, tuple[int, ...]]:
    path = read_stage_artifact(attempt, "stage2_data", "subset_manifest.json")
    manifest = _read_json(path, "Stage2 subset manifest")
    train = manifest.get("train")
    generation = manifest.get("generation")
    if not isinstance(train, Mapping) or not isinstance(generation, Mapping):
        raise ResultAuditError("Stage2 subset manifest lacks train/generation identities")
    selection = train.get("selection")
    if not isinstance(selection, Mapping):
        raise ResultAuditError("Stage2 train selection audit is missing")
    budget = config.budgets["confirmation"]
    configured_manifest = Path(budget.stage2_subset_manifests.get(city, "")).resolve()
    if not configured_manifest.is_file():
        raise ResultAuditError("configured confirmation Stage2 manifest is missing")
    configured_sha = _sha256(configured_manifest)
    frozen = _read_json(configured_manifest, "frozen confirmation Stage2 manifest")
    frozen_selected = frozen.get("selected")
    if not isinstance(frozen_selected, Mapping):
        raise ResultAuditError("frozen confirmation Stage2 manifest lacks selected identity")
    raw_frozen_ids = frozen_selected.get("traj_ids")
    if not isinstance(raw_frozen_ids, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in raw_frozen_ids
    ):
        raise ResultAuditError("frozen confirmation Stage2 trajectory IDs are invalid")
    frozen_ids = tuple(int(value) for value in raw_frozen_ids)
    frozen_hash = _traj_id_sha256(frozen_ids)
    expected_frozen = {
        "schema_version": 1,
        "kind": "frozen_stage2_train_subset",
        "method": "sha256_rank_fixed_city_v1",
        "requested": budget.stage2_train_trajectories,
        "subset_seed": budget.stage2_subset_seed,
        "city": city,
        "selection_depends_on_cell": False,
        "selection_depends_on_training_seed": False,
    }
    frozen_changed = {
        field: (frozen.get(field), value)
        for field, value in expected_frozen.items()
        if frozen.get(field) != value
    }
    if (
        frozen_changed
        or frozen_ids != tuple(sorted(frozen_ids))
        or len(set(frozen_ids)) != budget.stage2_train_trajectories
        or frozen_selected.get("population") != budget.stage2_train_trajectories
        or frozen_selected.get("traj_id_sha256") != frozen_hash
    ):
        raise ResultAuditError(
            f"frozen confirmation Stage2 identity mismatch: {frozen_changed}"
        )
    expected = {
        "method": "frozen_manifest_sha256_rank_fixed_city_v1",
        "population": budget.stage2_train_trajectories,
        "requested": budget.stage2_train_trajectories,
        "subset_seed": budget.stage2_subset_seed,
        "city": city,
        "selection_depends_on_training_seed": False,
        "manifest_path": str(configured_manifest),
        "manifest_sha256": configured_sha,
    }
    changed = {
        field: (selection.get(field), value)
        for field, value in expected.items()
        if selection.get(field) != value
    }
    raw_train_ids = selection.get("selected_traj_ids")
    if not isinstance(raw_train_ids, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in raw_train_ids
    ):
        raise ResultAuditError("confirmation Stage2 train trajectory IDs are invalid")
    train_ids = tuple(int(value) for value in raw_train_ids)
    train_hash = _traj_id_sha256(train_ids)
    if (
        changed
        or train.get("population") != budget.stage2_train_trajectories
        or train_ids != frozen_ids
        or train_hash != frozen_hash
        or selection.get("selected_traj_id_sha256") != frozen_hash
    ):
        raise ResultAuditError(f"confirmation Stage2 Option B identity mismatch: {changed}")
    raw_ids = generation.get("traj_ids")
    if not isinstance(raw_ids, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in raw_ids
    ):
        raise ResultAuditError("confirmation generation trajectory IDs are invalid")
    ids = tuple(int(value) for value in raw_ids)
    if (
        ids != tuple(sorted(ids))
        or len(set(ids)) != len(ids)
        or int(generation.get("population", -1)) != len(ids)
        or budget.stage3_trajectories != "full"
    ):
        raise ResultAuditError("confirmation generation population is not full, sorted, and unique")
    return configured_sha, train_hash, ids


def _validate_stage4_evidence(
    run_key: str,
    stage4: Mapping[str, object],
    contributions: Mapping[str, PopulationMetricContributions],
    expected_ids: tuple[int, ...],
) -> None:
    if set(stage4) != set(G_VARIANTS) or set(contributions) != set(G_VARIANTS):
        raise ResultAuditError(f"{run_key} lacks the exact four G variants")
    for variant in G_VARIANTS:
        payload = stage4[variant]
        if not isinstance(payload, Mapping):
            raise ResultAuditError(f"{run_key}/{variant} Stage4 payload is invalid")
        contribution = contributions[variant]
        if contribution.traj_ids != expected_ids:
            raise ResultAuditError(
                f"{run_key}/{variant} contribution population differs from full generation"
            )
        rows = payload.get("per_trajectory")
        if not isinstance(rows, list) or len(rows) != len(expected_ids):
            raise ResultAuditError(f"{run_key}/{variant} penalty population mismatch")
        observed_ids = tuple(sorted(int(row["traj_id"]) for row in rows))
        if observed_ids != expected_ids:
            raise ResultAuditError(f"{run_key}/{variant} Stage4/contribution IDs differ")
        declared = payload.get("population_aware_regret")
        if not isinstance(declared, Mapping) or declared.get("method") != POPULATION_REGRET_METHOD:
            raise ResultAuditError(f"{run_key}/{variant} lacks population-aware regret metadata")
        declared_metrics = declared.get("metrics")
        recalculated = population_regrets(contribution)
        if not isinstance(declared_metrics, Mapping) or set(declared_metrics) != set(recalculated):
            raise ResultAuditError(f"{run_key}/{variant} population-aware metric set differs")
        for metric, value in recalculated.items():
            observed = float(declared_metrics[metric])
            if not math.isfinite(observed) or not math.isclose(
                observed, float(value[0]), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ResultAuditError(
                    f"{run_key}/{variant}/{metric} population-aware regret mismatch"
                )


def _load_confirmation_run(
    config: ExperimentConfig,
    root: Path,
    job: ExperimentJob,
) -> ConfirmationRunEvidence:
    attempt = completed_attempt(root / job.run_key)
    if attempt is None:
        raise ResultAuditError(f"missing completed confirmation run: {job.run_key}")
    manifest = _read_json(attempt / "run_manifest.json", "confirmation run manifest")
    expected = {
        "experiment_id": config.experiment_id,
        "cell_id": job.cell_id,
        "city": job.city,
        "seed": job.seed,
        "budget": "confirmation",
        "status": "completed",
    }
    changed = {
        field: (manifest.get(field), value)
        for field, value in expected.items()
        if manifest.get(field) != value
    }
    git = manifest.get("git")
    if changed or not isinstance(git, Mapping) or bool(git.get("dirty")):
        raise ResultAuditError(f"confirmation run identity mismatch for {job.run_key}: {changed}")
    revision = str(git.get("revision", ""))
    if len(revision) != 40:
        raise ResultAuditError(f"confirmation run revision is invalid: {job.run_key}")
    stage1 = _read_json(
        read_stage_artifact(attempt, "stage1", "stage1_metrics.json"),
        "confirmation Stage1 metrics",
    )
    train_manifest_sha, train_ids_sha, generation_ids = _stage2_identity(
        config, attempt, job.city
    )
    stage4 = _read_json(
        read_stage_artifact(attempt, "stage4", "stage4_metrics.json"),
        "confirmation Stage4 metrics",
    )
    contributions = {
        variant: load_metric_contributions(
            read_stage_artifact(
                attempt, "stage4", f"{variant}_contributions_manifest.json"
            )
        )
        for variant in G_VARIANTS
    }
    _validate_stage4_evidence(job.run_key, stage4, contributions, generation_ids)
    evaluation_hash = _traj_id_sha256(generation_ids)
    return ConfirmationRunEvidence(
        run_key=job.run_key,
        cell_id=job.cell_id,
        city=job.city,
        seed=job.seed,
        attempt=attempt.resolve(),
        evaluation_revision=revision,
        source_attempt=attempt.resolve(),
        source_revision=revision,
        subset_sha256=train_manifest_sha,
        selected_traj_id_sha256=evaluation_hash,
        stage1=stage1,
        stage4=stage4,
        contributions=contributions,
        stage2_train_manifest_sha256=train_manifest_sha,
        stage2_train_traj_id_sha256=train_ids_sha,
        evaluation_traj_id_sha256=evaluation_hash,
        evaluation_population=len(generation_ids),
    )


def collect_confirmation_runs(
    config_path: str | Path,
    experiment_root: str | Path,
    cell_ids: Sequence[str],
) -> list[ConfirmationRunEvidence]:
    selected = tuple(cell_ids)
    if not selected:
        raise ResultAuditError("confirmation collection requires explicit final cells")
    config = ExperimentConfig.load(config_path)
    jobs = list(build_jobs(config, "confirmation", selected))
    runs = [
        _load_confirmation_run(config, Path(experiment_root), job)
        for job in jobs
    ]
    revisions = {run.evaluation_revision for run in runs}
    if len(revisions) != 1:
        raise ResultAuditError(f"confirmation runs span revisions: {sorted(revisions)}")
    train_by_city: dict[str, set[tuple[str, str]]] = defaultdict(set)
    evaluation_by_city: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for run in runs:
        train_by_city[run.city].add(
            (run.stage2_train_manifest_sha256, run.stage2_train_traj_id_sha256)
        )
        evaluation_by_city[run.city].add(
            (run.evaluation_population, run.evaluation_traj_id_sha256)
        )
    mismatched_train = {
        city: sorted(values) for city, values in train_by_city.items() if len(values) != 1
    }
    mismatched_evaluation = {
        city: sorted(values) for city, values in evaluation_by_city.items() if len(values) != 1
    }
    if mismatched_train:
        raise ResultAuditError(f"confirmation cross-run Stage2 subset mismatch: {mismatched_train}")
    if mismatched_evaluation:
        raise ResultAuditError(
            f"confirmation cross-run full-test population mismatch: {mismatched_evaluation}"
        )
    return runs


def _confirmation_gate(summary: Mapping[str, object]) -> dict[str, dict[str, object]]:
    gate_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in summary["stage1_gates"]:
        gate_rows[str(row["cell_id"])].append(row)
    effect_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in summary["paired_metric_effects"]:
        effect_rows[str(row["cell_id"])].append(row)
    result = {}
    for cell in sorted(set(effect_rows)):
        rows = effect_rows[cell]
        degradation = sorted(
            {
                (str(row["city"]), str(row["metric"]))
                for row in rows
                if row["group"] in ALLOWED_CLAIM_GROUPS | {"validity"}
                and float(row["ci_low"]) > 0
            }
        )
        improvement = sorted(
            {
                (str(row["city"]), str(row["metric"]))
                for row in rows
                if row["group"] in ALLOWED_CLAIM_GROUPS
                and float(row["ci_high"]) < 0
            }
        )
        stage1_passed = bool(gate_rows[cell]) and all(
            bool(row["passed"]) for row in gate_rows[cell]
        )
        result[cell] = {
            "stage1_gate_passed": stage1_passed,
            "significant_degradation": [
                {"city": city, "metric": metric} for city, metric in degradation
            ],
            "significant_improvement": [
                {"city": city, "metric": metric} for city, metric in improvement
            ],
            "claim_supported": stage1_passed and not degradation and bool(improvement),
        }
    return result


def summarize_confirmation_results(
    runs: Sequence[ConfirmationRunEvidence],
    selection_audit: Mapping[str, object],
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 20260826,
    confidence: float = 0.95,
) -> dict[str, object]:
    if not runs:
        raise ResultAuditError("confirmation summary requires completed runs")
    summary = summarize_corrected_results(
        runs,  # type: ignore[arg-type]
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    expected_cells = tuple(str(value) for value in selection_audit["confirmation_cells"])
    if set(summary["coverage"]["cells"]) != set(expected_cells):
        raise ResultAuditError("confirmation coverage differs from frozen final selection")
    revision = str(summary["revision"])
    summary["schema_version"] = ANALYSIS_SCHEMA_VERSION
    summary["analysis_schema_version"] = ANALYSIS_SCHEMA_VERSION
    summary["candidate_evidence"] = CONFIRMATION_EVIDENCE
    summary["phase"] = "confirmation"
    summary["selection_audit"] = dict(selection_audit)
    summary["run_revision_audit"] = {"uniform": True, "revision": revision}
    summary.pop("source_revision_audit", None)
    summary["population_identity_by_city"] = {
        city: {
            "stage2_train_manifest_sha256": next(
                run.stage2_train_manifest_sha256 for run in runs if run.city == city
            ),
            "stage2_train_traj_id_sha256": next(
                run.stage2_train_traj_id_sha256 for run in runs if run.city == city
            ),
            "evaluation_population": next(
                run.evaluation_population for run in runs if run.city == city
            ),
            "evaluation_traj_id_sha256": next(
                run.evaluation_traj_id_sha256 for run in runs if run.city == city
            ),
        }
        for city in summary["coverage"]["cities"]
    }
    summary["evidence_rows"] = [
        {
            "run_key": run.run_key,
            "attempt": str(run.attempt),
            "revision": run.evaluation_revision,
            "stage2_train_manifest_sha256": run.stage2_train_manifest_sha256,
            "stage2_train_traj_id_sha256": run.stage2_train_traj_id_sha256,
            "evaluation_population": run.evaluation_population,
            "evaluation_traj_id_sha256": run.evaluation_traj_id_sha256,
        }
        for run in runs
    ]
    summary["confirmation_gate"] = _confirmation_gate(summary)
    summary["final_claim_boundary"] = (
        "A superiority claim is permitted only for a candidate whose confirmation_gate."
        "claim_supported is true; all other outcomes, including negative results, remain evidence."
    )
    return summary


def write_confirmation_report(
    summary: Mapping[str, object],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "analysis.json"
    markdown_path = output / "analysis.md"
    coverage = summary["coverage"]
    lines = [
        "# Full confirmation population-aware result audit",
        "",
        f"- Confirmation revision: `{summary['revision']}`",
        f"- Verified final selection: `{summary['selection_audit']['selection_sha256']}`",
        f"- Completed runs: {coverage['runs']}",
        f"- Cities: {', '.join(coverage['cities'])}",
        f"- Seeds: {', '.join(str(value) for value in coverage['seeds'])}",
        f"- Cells: {', '.join(coverage['cells'])}",
        f"- Paired metric effects: {len(summary['paired_metric_effects'])}",
        f"- G-attribution rows: {len(summary['g_metric_attribution'])}",
        "",
        "## Confirmation claim gate",
        "",
    ]
    for cell, gate in summary["confirmation_gate"].items():
        lines.extend(
            [
                f"- `{cell}`: claim_supported={str(gate['claim_supported']).lower()}, "
                f"improvements={len(gate['significant_improvement'])}, "
                f"degradations={len(gate['significant_degradation'])}",
            ]
        )
    lines.extend(
        [
            "",
            "Trajectories are resampled within seed and seed effects are averaged equally.",
            "Failures remain in every primary GPS/Road/Grid population-level regret.",
            "POI is excluded from the registered experiment.",
            "",
        ]
    )
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_markdown = markdown_path.with_suffix(".md.tmp")
    temporary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_markdown.write_text("\n".join(lines), encoding="utf-8")
    temporary_json.replace(json_path)
    temporary_markdown.replace(markdown_path)
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and summarize full confirmation population evidence"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument("--confidence", type=float, default=0.95)
    arguments = parser.parse_args()
    selection = audit_confirmation_selection(arguments.selection, arguments.cells)
    runs = collect_confirmation_runs(
        arguments.config, arguments.experiment_root, arguments.cells
    )
    summary = summarize_confirmation_results(
        runs,
        selection,
        n_resamples=arguments.n_resamples,
        bootstrap_seed=arguments.bootstrap_seed,
        confidence=arguments.confidence,
    )
    paths = write_confirmation_report(summary, arguments.output_dir)
    print(
        json.dumps(
            {
                "analysis_json": str(paths[0]),
                "analysis_markdown": str(paths[1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
