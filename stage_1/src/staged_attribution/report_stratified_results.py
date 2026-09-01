from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analyze import (
    CORE_CELLS,
    E_BEST_PAIR_CONTRAST,
    staged_contrast_definitions,
)
from .config import ExperimentConfig
from .evaluate_stage1 import validity_gate
from .immutable_artifacts import canonical_json_bytes, write_immutable_artifact_set
from .population_bootstrap import (
    DOMAIN_METRICS,
    POPULATION_REGRET_METHOD,
    SELECTION_BOOTSTRAP_METHOD,
    PopulationMetricContributions,
    g_metric_attribution,
    g_metric_attribution_stratified_seeds,
    load_metric_contributions,
    paired_metric_effects_stratified_seeds,
    population_regrets,
)
from .report_results import G_VARIANTS, METRICS, ResultAuditError, read_stage_artifact
from .run_experiment import build_jobs
from .run_stratified_reevaluation_batch import completed_reevaluation_attempt


ANALYSIS_SCHEMA_VERSION = 5
CANDIDATE_EVIDENCE = "stratified_population_aware"
PENALTY_METHOD = "population_penalized_error_v1"


@dataclass(frozen=True)
class CorrectedRunEvidence:
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


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultAuditError(f"invalid {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ResultAuditError(f"{label} must be a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _penalty_vector(payload: Mapping[str, object], label: str) -> tuple[tuple[int, ...], np.ndarray]:
    rows = payload.get("per_trajectory")
    population = int(payload.get("population", -1))
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != population:
        raise ResultAuditError(f"{label} penalty population mismatch")
    parsed = sorted(
        (int(row["traj_id"]), float(row["penalized_error"]))
        for row in rows if isinstance(row, Mapping)
    )
    ids = tuple(traj_id for traj_id, _ in parsed)
    values = np.asarray([value for _, value in parsed], dtype=np.float64)
    if len(ids) != population or len(set(ids)) != population or not np.all(np.isfinite(values)):
        raise ResultAuditError(f"{label} has invalid per-trajectory penalties")
    observed = float(payload.get("population_mean_penalized_error", math.nan))
    if not math.isfinite(observed) or not math.isclose(observed, float(values.mean()), rel_tol=1e-9, abs_tol=1e-9):
        raise ResultAuditError(f"{label} population penalty does not equal its trajectory mean")
    return ids, values


def _validate_run_populations(run: CorrectedRunEvidence) -> None:
    contribution_ids = {variant: run.contributions[variant].traj_ids for variant in G_VARIANTS}
    if len(set(contribution_ids.values())) != 1:
        raise ResultAuditError(f"G contribution populations differ for {run.run_key}")
    expected_ids = next(iter(contribution_ids.values()))
    for variant in G_VARIANTS:
        ids, _ = _penalty_vector(run.stage4[variant], f"{run.run_key}/{variant}")
        if ids != expected_ids:
            raise ResultAuditError(f"Stage4/contribution population differs for {run.run_key}/{variant}")
    observed_hash = hashlib.sha256("\n".join(map(str, expected_ids)).encode()).hexdigest()
    if observed_hash != run.selected_traj_id_sha256:
        raise ResultAuditError(f"fixed subset identity mismatch for {run.run_key}")


def _load_run(config: ExperimentConfig, root: Path, job) -> CorrectedRunEvidence:
    attempt = completed_reevaluation_attempt(root / job.run_key)
    if attempt is None:
        raise ResultAuditError(f"missing completed corrected reevaluation: {job.run_key}")
    manifest = _read_json(attempt / "reevaluation_manifest.json", "reevaluation manifest")
    expected = {
        "experiment_id": config.experiment_id,
        "reevaluation": "fixed_stratified_population_aware_v1",
        "cell_id": job.cell_id,
        "city": job.city,
        "seed": job.seed,
        "budget": "controlled",
        "status": "completed",
    }
    changed = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if changed:
        raise ResultAuditError(f"corrected reevaluation identity mismatch for {job.run_key}: {changed}")
    git = manifest.get("git")
    if not isinstance(git, Mapping) or bool(git.get("dirty")):
        raise ResultAuditError(f"corrected reevaluation used a dirty worktree: {job.run_key}")
    evaluation_revision = str(git.get("revision", ""))
    if len(evaluation_revision) != 40:
        raise ResultAuditError(f"invalid reevaluation revision: {job.run_key}")

    reference_path = read_stage_artifact(attempt, "source_reference", "source_reference.json")
    reference = _read_json(reference_path, "source reference")
    source_attempt = Path(str(reference.get("source_attempt", ""))).resolve()
    declared_source = Path(str(manifest.get("source_attempt", ""))).resolve()
    if source_attempt != declared_source:
        raise ResultAuditError(f"source reference/manifest attempt mismatch for {job.run_key}")
    source_manifest = _read_json(source_attempt / "run_manifest.json", "source run manifest")
    source_expected = {
        "experiment_id": config.experiment_id,
        "cell_id": job.cell_id,
        "city": job.city,
        "seed": job.seed,
        "budget": "controlled",
        "status": "completed",
    }
    source_changed = {
        key: (source_manifest.get(key), value)
        for key, value in source_expected.items() if source_manifest.get(key) != value
    }
    if source_changed or bool(source_manifest.get("git", {}).get("dirty")):
        raise ResultAuditError(f"source run identity mismatch for {job.run_key}: {source_changed}")
    source_revision = str(source_manifest.get("git", {}).get("revision", ""))
    if source_revision != str(reference.get("source_revision", "")) or len(source_revision) != 40:
        raise ResultAuditError(f"source revision mismatch for {job.run_key}")
    stage1_path = read_stage_artifact(source_attempt, "stage1", "stage1_metrics.json")
    stage1 = _read_json(stage1_path, "Stage1 metrics")

    stage4_path = read_stage_artifact(attempt, "stage4", "stage4_metrics.json")
    stage4 = _read_json(stage4_path, "corrected Stage4 metrics")
    if set(stage4) != set(G_VARIANTS):
        raise ResultAuditError(f"corrected Stage4 variants differ for {job.run_key}")
    contributions = {
        variant: load_metric_contributions(
            read_stage_artifact(
                attempt, "stage4", f"{variant}_contributions_manifest.json"
            )
        )
        for variant in G_VARIANTS
    }
    subset = reference.get("subset")
    if not isinstance(subset, Mapping):
        raise ResultAuditError(f"source reference lacks subset identity for {job.run_key}")
    subset_sha = str(subset.get("sha256", ""))
    selected_sha = str(subset.get("selected_traj_id_sha256", ""))
    if len(subset_sha) != 64 or len(selected_sha) != 64:
        raise ResultAuditError(f"invalid subset hashes for {job.run_key}")
    run = CorrectedRunEvidence(
        run_key=job.run_key,
        cell_id=job.cell_id,
        city=job.city,
        seed=job.seed,
        attempt=attempt.resolve(),
        evaluation_revision=evaluation_revision,
        source_attempt=source_attempt,
        source_revision=source_revision,
        subset_sha256=subset_sha,
        selected_traj_id_sha256=selected_sha,
        stage1=stage1,
        stage4=stage4,
        contributions=contributions,
    )
    _validate_run_populations(run)
    return run


def collect_corrected_runs(
    config_path: str | Path,
    reevaluation_root: str | Path,
    cell_ids: Sequence[str] | None = None,
    *,
    allow_phase_source_revisions: bool = False,
) -> list[CorrectedRunEvidence]:
    config = ExperimentConfig.load(config_path)
    selected = tuple(cell_ids or CORE_CELLS)
    if len(set(selected)) != len(selected) or "e0_q0" not in selected:
        raise ResultAuditError("corrected collection requires unique cells including e0_q0")
    jobs = list(build_jobs(config, "controlled", selected))
    runs = [_load_run(config, Path(reevaluation_root), job) for job in jobs]
    evaluation_revisions = {run.evaluation_revision for run in runs}
    if len(evaluation_revisions) != 1:
        raise ResultAuditError(f"incomparable reevaluation revisions: {sorted(evaluation_revisions)}")
    source_by_cell: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        source_by_cell[run.cell_id].add(run.source_revision)
    unstable = {cell: sorted(values) for cell, values in source_by_cell.items() if len(values) != 1}
    if unstable:
        raise ResultAuditError(f"source revision varies within a cell: {unstable}")
    source_revisions = {next(iter(values)) for values in source_by_cell.values()}
    if len(source_revisions) != 1 and not allow_phase_source_revisions:
        raise ResultAuditError(f"incomparable source revisions: {sorted(source_revisions)}")
    subset_by_stratum: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for run in runs:
        subset_by_stratum[(run.city, run.seed)].add(
            (run.subset_sha256, run.selected_traj_id_sha256)
        )
    mismatched = {str(key): sorted(values) for key, values in subset_by_stratum.items() if len(values) != 1}
    if mismatched:
        raise ResultAuditError(f"cross-cell fixed subset mismatch: {mismatched}")
    return runs


def _interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


def _stratified_penalty_effect(
    baseline: Mapping[int, tuple[tuple[int, ...], np.ndarray]],
    candidate: Mapping[int, tuple[tuple[int, ...], np.ndarray]],
    *,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    strata = sorted(baseline)
    if not strata or set(strata) != set(candidate):
        raise ResultAuditError("penalty seed strata differ")
    samples = np.zeros(n_resamples, dtype=np.float64)
    estimate = 0.0
    populations = {}
    for current_seed in strata:
        baseline_ids, baseline_values = baseline[current_seed]
        candidate_ids, candidate_values = candidate[current_seed]
        if baseline_ids != candidate_ids:
            raise ResultAuditError(f"penalty paired populations differ for seed {current_seed}")
        population = len(baseline_ids)
        populations[current_seed] = population
        generator = np.random.default_rng(np.random.SeedSequence([int(seed), current_seed]))
        weights = generator.multinomial(
            population, np.full(population, 1 / population), size=n_resamples
        )
        samples += (weights @ (candidate_values - baseline_values) / population) / len(strata)
        estimate += float((candidate_values - baseline_values).mean()) / len(strata)
    low, high = _interval(samples, confidence)
    return {
        "metric": "population_mean_penalized_error",
        "method": SELECTION_BOOTSTRAP_METHOD,
        "seed_strata": strata,
        "seed_populations": populations,
        "n_pairs": sum(populations.values()),
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "p_improvement": float(np.mean(samples < 0)),
        "p_degradation": float(np.mean(samples > 0)),
        "seed": int(seed),
        "n_resamples": int(n_resamples),
    }


def _apply_staged_contrast(
    values: Mapping[str, np.ndarray],
    definition: Mapping[str, object],
) -> np.ndarray:
    if definition["kind"] == "linear":
        coefficients = definition["coefficients"]
        if not isinstance(coefficients, Mapping):
            raise ResultAuditError("linear staged contrast lacks coefficients")
        return sum(
            float(coefficient) * values[str(cell)]
            for cell, coefficient in coefficients.items()
        )
    if definition["contrast"] != E_BEST_PAIR_CONTRAST:
        raise ResultAuditError("unknown nonlinear staged contrast")
    return values["e6_q0"] - np.minimum.reduce(
        [values["e3_q0"], values["e4_q0"], values["e5_q0"]]
    )


def _staged_city_direction_heterogeneity(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["family"]), str(row["contrast"]), str(row["metric"]))
        city = str(row["city"])
        if city in grouped[key]:
            raise ResultAuditError(f"duplicate staged city effect: {key}/{city}")
        grouped[key][city] = row
    result = []
    for (family, contrast, metric), by_city in sorted(grouped.items()):
        estimates = {
            city: float(row["estimate"])
            for city, row in sorted(by_city.items())
        }
        directions = {
            city: (
                "improvement" if value < 0
                else "degradation" if value > 0
                else "neutral"
            )
            for city, value in estimates.items()
        }
        opposed = "improvement" in directions.values() and "degradation" in directions.values()
        result.append(
            {
                "family": family,
                "contrast": contrast,
                "metric": metric,
                "group": METRICS[metric][1],
                "city_estimates": estimates,
                "city_directions": directions,
                "opposed_direction": opposed,
                "direction_consistent": not opposed,
                "estimate_range": max(estimates.values()) - min(estimates.values()),
                "method": "descriptive_city_direction_v1",
            }
        )
    return result


def summarize_staged_attribution(
    runs: Sequence[CorrectedRunEvidence],
    cells: Sequence[str],
    cities: Sequence[str],
    seeds: Sequence[int],
    *,
    n_resamples: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, object]:
    """Estimate registered E/Q contrasts on the corrected fixed population."""
    missing = [cell for cell in CORE_CELLS if cell not in cells]
    base = {
        "available": not missing,
        "required_cells": list(CORE_CELLS),
        "observed_cells": list(cells),
        "missing_cells": missing,
        "method": SELECTION_BOOTSTRAP_METHOD,
        "unit": "trajectory_within_seed_equal_weight_across_seeds",
        "contrast_definitions": staged_contrast_definitions(),
    }
    if missing:
        return {
            **base,
            "reason": "complete registered core coverage is required for identifiable E/Q contrasts",
            "rows": [],
            "city_direction_heterogeneity": [],
        }

    lookup = {(run.city, run.seed, run.cell_id): run for run in runs}
    definitions = staged_contrast_definitions()
    rows: list[dict[str, object]] = []
    for city in cities:
        sampled_contrasts = {
            (str(definition["family"]), str(definition["contrast"]), metric): np.zeros(
                n_resamples, dtype=np.float64
            )
            for definition in definitions
            for metric in METRICS
        }
        point_contrasts = {key: 0.0 for key in sampled_contrasts}
        populations: dict[int, int] = {}
        for current_seed in seeds:
            current = {
                cell: lookup[(city, current_seed, cell)]
                for cell in CORE_CELLS
            }
            contribution_ids = {
                run.contributions["g11"].traj_ids for run in current.values()
            }
            if len(contribution_ids) != 1:
                raise ResultAuditError(
                    f"E/Q paired populations differ for {city}/seed_{current_seed}"
                )
            traj_ids = next(iter(contribution_ids))
            population = len(traj_ids)
            populations[int(current_seed)] = population
            generator = np.random.default_rng(
                np.random.SeedSequence([int(bootstrap_seed), int(current_seed)])
            )
            weights = generator.multinomial(
                population,
                np.full(population, 1 / population),
                size=n_resamples,
            )
            sampled_by_metric: dict[str, dict[str, np.ndarray]] = {
                metric: {} for metric in METRICS
            }
            point_by_metric: dict[str, dict[str, np.ndarray]] = {
                metric: {} for metric in METRICS
            }
            for cell, run in current.items():
                sampled = population_regrets(run.contributions["g11"], weights)
                points = population_regrets(run.contributions["g11"])
                penalty_ids, penalties = _penalty_vector(
                    run.stage4["g11"], f"{city}/seed_{current_seed}/{cell}/g11"
                )
                if penalty_ids != traj_ids:
                    raise ResultAuditError(
                        f"E/Q penalty population differs for {city}/seed_{current_seed}/{cell}"
                    )
                sampled["population_mean_penalized_error"] = weights @ penalties / population
                points["population_mean_penalized_error"] = np.asarray(
                    [float(penalties.mean())], dtype=np.float64
                )
                for metric in METRICS:
                    sampled_by_metric[metric][cell] = sampled[metric]
                    point_by_metric[metric][cell] = points[metric]
            for definition in definitions:
                family = str(definition["family"])
                contrast = str(definition["contrast"])
                for metric in METRICS:
                    key = (family, contrast, metric)
                    sampled_contrasts[key] += _apply_staged_contrast(
                        sampled_by_metric[metric], definition
                    ) / len(seeds)
                    point_contrasts[key] += float(
                        _apply_staged_contrast(point_by_metric[metric], definition)[0]
                    ) / len(seeds)

        for definition in definitions:
            family = str(definition["family"])
            contrast = str(definition["contrast"])
            for metric in METRICS:
                key = (family, contrast, metric)
                values = sampled_contrasts[key]
                low, high = _interval(values, confidence)
                row = {
                    "family": family,
                    "contrast": contrast,
                    "contrast_kind": definition["kind"],
                    "inferential_role": definition["role"],
                    "city": city,
                    "metric": metric,
                    "group": METRICS[metric][1],
                    "orientation": "minimize",
                    "method": SELECTION_BOOTSTRAP_METHOD,
                    "seed_strata": list(seeds),
                    "seed_populations": populations,
                    "n_pairs": sum(populations.values()),
                    "estimate": point_contrasts[key],
                    "ci_low": low,
                    "ci_high": high,
                    "p_improvement": float(np.mean(values < 0)),
                    "p_degradation": float(np.mean(values > 0)),
                    "seed": int(bootstrap_seed),
                    "n_resamples": int(n_resamples),
                }
                if definition["kind"] == "linear":
                    row["coefficients"] = dict(definition["coefficients"])
                else:
                    row["expression"] = definition["expression"]
                rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["family"]), str(row["contrast"]), str(row["city"]), str(row["metric"])
        )
    )
    return {
        **base,
        "rows": rows,
        "city_direction_heterogeneity": _staged_city_direction_heterogeneity(rows),
    }


def _g_penalty_attribution(
    run: CorrectedRunEvidence,
    *,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> list[dict[str, object]]:
    parsed = {variant: _penalty_vector(run.stage4[variant], f"{run.run_key}/{variant}") for variant in G_VARIANTS}
    if len({ids for ids, _ in parsed.values()}) != 1:
        raise ResultAuditError(f"G penalty populations differ for {run.run_key}")
    population = len(parsed["g00"][0])
    generator = np.random.default_rng(seed)
    weights = generator.multinomial(
        population, np.full(population, 1 / population), size=n_resamples
    )
    samples = {variant: weights @ values / population for variant, (_, values) in parsed.items()}
    points = {variant: float(values.mean()) for variant, (_, values) in parsed.items()}
    sampled = {
        "sid": 0.5 * ((samples["g10"] - samples["g00"]) + (samples["g11"] - samples["g01"])),
        "sampling": 0.5 * ((samples["g01"] - samples["g00"]) + (samples["g11"] - samples["g10"])),
        "interaction": samples["g11"] - samples["g10"] - samples["g01"] + samples["g00"],
        "total": samples["g11"] - samples["g00"],
    }
    point = {
        "sid": 0.5 * ((points["g10"] - points["g00"]) + (points["g11"] - points["g01"])),
        "sampling": 0.5 * ((points["g01"] - points["g00"]) + (points["g11"] - points["g10"])),
        "interaction": points["g11"] - points["g10"] - points["g01"] + points["g00"],
        "total": points["g11"] - points["g00"],
    }
    rows = []
    for component, values in sampled.items():
        low, high = _interval(values, confidence)
        rows.append(
            {
                "metric": "population_mean_penalized_error",
                "component": component,
                "method": PENALTY_METHOD,
                "n_pairs": population,
                "estimate": point[component],
                "ci_low": low,
                "ci_high": high,
                "p_improvement": float(np.mean(values < 0)),
                "p_degradation": float(np.mean(values > 0)),
                "seed": int(seed),
                "n_resamples": int(n_resamples),
            }
        )
    return rows


def _g_penalty_attribution_stratified_seeds(
    runs_by_seed: Mapping[int, CorrectedRunEvidence],
    *,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> list[dict[str, object]]:
    seed_ids = sorted(int(value) for value in runs_by_seed)
    if not seed_ids:
        raise ResultAuditError("G penalty attribution requires seed strata")
    components = ("sid", "sampling", "interaction", "total")
    sampled_components = {
        component: np.zeros(n_resamples, dtype=np.float64)
        for component in components
    }
    point_components = {component: 0.0 for component in components}
    populations: dict[int, int] = {}
    for current_seed in seed_ids:
        run = runs_by_seed[current_seed]
        parsed = {
            variant: _penalty_vector(
                run.stage4[variant], f"{run.run_key}/{variant}"
            )
            for variant in G_VARIANTS
        }
        if len({ids for ids, _ in parsed.values()}) != 1:
            raise ResultAuditError(
                f"G penalty populations differ for {run.run_key}"
            )
        population = len(parsed["g00"][0])
        populations[current_seed] = population
        generator = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(current_seed)])
        )
        weights = generator.multinomial(
            population, np.full(population, 1 / population), size=n_resamples
        )
        sampled = {
            variant: weights @ values / population
            for variant, (_, values) in parsed.items()
        }
        points = {
            variant: float(values.mean())
            for variant, (_, values) in parsed.items()
        }
        current_sampled = {
            "sid": 0.5 * (
                (sampled["g10"] - sampled["g00"])
                + (sampled["g11"] - sampled["g01"])
            ),
            "sampling": 0.5 * (
                (sampled["g01"] - sampled["g00"])
                + (sampled["g11"] - sampled["g10"])
            ),
            "interaction": sampled["g11"] - sampled["g10"] - sampled["g01"] + sampled["g00"],
            "total": sampled["g11"] - sampled["g00"],
        }
        current_points = {
            "sid": 0.5 * (
                (points["g10"] - points["g00"])
                + (points["g11"] - points["g01"])
            ),
            "sampling": 0.5 * (
                (points["g01"] - points["g00"])
                + (points["g11"] - points["g10"])
            ),
            "interaction": points["g11"] - points["g10"] - points["g01"] + points["g00"],
            "total": points["g11"] - points["g00"],
        }
        for component in components:
            sampled_components[component] += current_sampled[component] / len(seed_ids)
            point_components[component] += current_points[component] / len(seed_ids)
    rows = []
    for component, values in sampled_components.items():
        low, high = _interval(values, confidence)
        rows.append(
            {
                "metric": "population_mean_penalized_error",
                "component": component,
                "method": SELECTION_BOOTSTRAP_METHOD,
                "seed_strata": seed_ids,
                "seed_populations": populations,
                "n_pairs": sum(populations.values()),
                "estimate": point_components[component],
                "ci_low": low,
                "ci_high": high,
                "p_improvement": float(np.mean(values < 0)),
                "p_degradation": float(np.mean(values > 0)),
                "seed": int(seed),
                "n_resamples": int(n_resamples),
            }
        )
    return rows


def _g_city_direction_heterogeneity(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str], dict[str, Mapping[str, object]]
    ] = defaultdict(dict)
    for row in rows:
        key = (str(row["cell_id"]), str(row["component"]), str(row["metric"]))
        city = str(row["city"])
        if city in grouped[key]:
            raise ResultAuditError(f"duplicate G city effect: {key}/{city}")
        grouped[key][city] = row
    result = []
    for (cell, component, metric), by_city in sorted(grouped.items()):
        estimates = {
            city: float(row["estimate"])
            for city, row in sorted(by_city.items())
        }
        directions = {
            city: (
                "improvement" if value < 0
                else "degradation" if value > 0
                else "neutral"
            )
            for city, value in estimates.items()
        }
        opposed = "improvement" in directions.values() and "degradation" in directions.values()
        result.append(
            {
                "cell_id": cell,
                "component": component,
                "metric": metric,
                "group": METRICS[metric][1],
                "city_estimates": estimates,
                "city_directions": directions,
                "opposed_direction": opposed,
                "direction_consistent": not opposed,
                "estimate_range": max(estimates.values()) - min(estimates.values()),
                "method": "descriptive_city_direction_v1",
            }
        )
    return result


def summarize_g_attribution(
    runs: Sequence[CorrectedRunEvidence],
    cells: Sequence[str],
    cities: Sequence[str],
    seeds: Sequence[int],
    *,
    n_resamples: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, object]:
    """Estimate registered G components across equal-weight seed strata."""
    lookup = {(run.city, run.seed, run.cell_id): run for run in runs}
    rows: list[dict[str, object]] = []
    for city in cities:
        for cell in cells:
            runs_by_seed = {
                int(current_seed): lookup[(city, current_seed, cell)]
                for current_seed in seeds
            }
            current = g_metric_attribution_stratified_seeds(
                {
                    current_seed: run.contributions
                    for current_seed, run in runs_by_seed.items()
                },
                n_resamples=n_resamples,
                seed=bootstrap_seed,
                confidence=confidence,
            )
            current.extend(
                _g_penalty_attribution_stratified_seeds(
                    runs_by_seed,
                    n_resamples=n_resamples,
                    seed=bootstrap_seed,
                    confidence=confidence,
                )
            )
            for row in current:
                metric = str(row["metric"])
                rows.append(
                    {
                        "city": city,
                        "cell_id": cell,
                        "group": METRICS[metric][1],
                        "orientation": "minimize",
                        "inferential_role": "primary",
                        **row,
                    }
                )
    rows.sort(
        key=lambda row: (
            str(row["city"]), str(row["cell_id"]),
            str(row["component"]), str(row["metric"]),
        )
    )
    return {
        "available": True,
        "required_variants": list(G_VARIANTS),
        "components": ["sid", "sampling", "interaction", "total"],
        "observed_cells": list(cells),
        "method": SELECTION_BOOTSTRAP_METHOD,
        "unit": "trajectory_within_seed_equal_weight_across_seeds",
        "rows": rows,
        "city_direction_heterogeneity": _g_city_direction_heterogeneity(rows),
    }


def _stage1_gates(runs: Sequence[CorrectedRunEvidence]) -> list[dict[str, object]]:
    baselines = {
        (run.city, run.seed): float(run.stage4["g00"]["population_mean_penalized_error"])
        for run in runs if run.cell_id == "e0_q0"
    }
    rows = []
    for run in runs:
        oracle = float(run.stage4["g00"]["population_mean_penalized_error"])
        baseline = baselines.get((run.city, run.seed))
        if baseline is None:
            raise ResultAuditError(f"missing E0 fixed-subset oracle baseline for {run.city}/seed_{run.seed}")
        ratio = 1.0 if baseline == 0 and oracle == 0 else (math.inf if baseline == 0 else oracle / baseline)
        losses = run.stage1.get("losses")
        finite = isinstance(losses, Mapping) and bool(losses) and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in losses.values()
        )
        gate = validity_gate(run.stage1["codebook"], finite, ratio)
        rows.append(
            {
                "city": run.city,
                "seed": run.seed,
                "cell_id": run.cell_id,
                "oracle_error": oracle,
                "e0_oracle_error": baseline,
                "oracle_error_ratio": ratio,
                **gate,
            }
        )
    return rows


def summarize_city_direction_heterogeneity(
    effects: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Describe whether a candidate effect points in opposite directions by city."""
    grouped: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in effects:
        cell = str(row["cell_id"])
        metric = str(row["metric"])
        city = str(row["city"])
        key = (cell, metric)
        if city in grouped[key]:
            raise ResultAuditError(
                f"duplicate city effect for heterogeneity: {cell}/{metric}/{city}"
            )
        grouped[key][city] = row
    result = []
    for (cell, metric), by_city in sorted(grouped.items()):
        estimates = {
            city: float(row["estimate"])
            for city, row in sorted(by_city.items())
        }
        if not estimates or not all(math.isfinite(value) for value in estimates.values()):
            raise ResultAuditError(f"invalid city effects for {cell}/{metric}")
        groups = {str(row["group"]) for row in by_city.values()}
        if len(groups) != 1:
            raise ResultAuditError(f"metric group varies by city for {cell}/{metric}")
        directions = {
            city: (
                "improvement" if value < 0
                else "degradation" if value > 0
                else "neutral"
            )
            for city, value in estimates.items()
        }
        opposed = "improvement" in directions.values() and "degradation" in directions.values()
        result.append(
            {
                "cell_id": cell,
                "metric": metric,
                "group": next(iter(groups)),
                "city_estimates": estimates,
                "city_directions": directions,
                "opposed_direction": opposed,
                "direction_consistent": not opposed,
                "estimate_range": max(estimates.values()) - min(estimates.values()),
                "method": "descriptive_city_direction_v1",
            }
        )
    return result


def summarize_corrected_results(
    runs: Sequence[CorrectedRunEvidence],
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 20260826,
    confidence: float = 0.95,
) -> dict[str, object]:
    if not runs or n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("corrected summary requires runs and valid bootstrap settings")
    evaluation_revisions = {run.evaluation_revision for run in runs}
    if len(evaluation_revisions) != 1:
        raise ResultAuditError("corrected summary mixes reevaluation revisions")
    cells = sorted({run.cell_id for run in runs})
    cities = sorted({run.city for run in runs})
    seeds = sorted({run.seed for run in runs})
    expected = {(city, seed, cell) for city in cities for seed in seeds for cell in cells}
    observed = {(run.city, run.seed, run.cell_id) for run in runs}
    if observed != expected or len(observed) != len(runs) or "e0_q0" not in cells:
        raise ResultAuditError("corrected summary lacks complete city/seed/cell Cartesian coverage")

    metric_rows = []
    g_rows = []
    for run in runs:
        for variant in G_VARIANTS:
            point = population_regrets(run.contributions[variant])
            for metric, values in point.items():
                metric_rows.append(
                    {
                        "city": run.city,
                        "seed": run.seed,
                        "cell_id": run.cell_id,
                        "variant": variant,
                        "metric": metric,
                        "group": METRICS[metric][1],
                        "orientation": "minimize",
                        "value": float(values[0]),
                        "regret": float(values[0]),
                        "method": POPULATION_REGRET_METHOD,
                    }
                )
            penalty = float(run.stage4[variant]["population_mean_penalized_error"])
            metric_rows.append(
                {
                    "city": run.city,
                    "seed": run.seed,
                    "cell_id": run.cell_id,
                    "variant": variant,
                    "metric": "population_mean_penalized_error",
                    "group": METRICS["population_mean_penalized_error"][1],
                    "orientation": "minimize",
                    "value": penalty,
                    "regret": penalty,
                    "method": PENALTY_METHOD,
                }
            )
        current_g = g_metric_attribution(
            run.contributions,
            n_resamples=n_resamples,
            seed=bootstrap_seed + run.seed,
            confidence=confidence,
        )
        current_g.extend(
            _g_penalty_attribution(
                run,
                n_resamples=n_resamples,
                seed=bootstrap_seed + run.seed,
                confidence=confidence,
            )
        )
        for row in current_g:
            g_rows.append({"city": run.city, "seed": run.seed, "cell_id": run.cell_id, **row})

    lookup = {(run.city, run.seed, run.cell_id): run for run in runs}
    paired = []
    for city in cities:
        for cell in cells:
            if cell == "e0_q0":
                continue
            baseline_contributions = {
                seed: lookup[(city, seed, "e0_q0")].contributions["g11"] for seed in seeds
            }
            candidate_contributions = {
                seed: lookup[(city, seed, cell)].contributions["g11"] for seed in seeds
            }
            rows = paired_metric_effects_stratified_seeds(
                baseline_contributions,
                candidate_contributions,
                n_resamples=n_resamples,
                seed=bootstrap_seed,
                confidence=confidence,
            )
            baseline_penalties = {
                seed: _penalty_vector(
                    lookup[(city, seed, "e0_q0")].stage4["g11"],
                    f"{city}/seed_{seed}/e0_q0/g11",
                )
                for seed in seeds
            }
            candidate_penalties = {
                seed: _penalty_vector(
                    lookup[(city, seed, cell)].stage4["g11"],
                    f"{city}/seed_{seed}/{cell}/g11",
                )
                for seed in seeds
            }
            rows.append(
                _stratified_penalty_effect(
                    baseline_penalties,
                    candidate_penalties,
                    n_resamples=n_resamples,
                    seed=bootstrap_seed,
                    confidence=confidence,
                )
            )
            for row in rows:
                paired.append(
                    {
                        "reference": "e0_q0",
                        "city": city,
                        "cell_id": cell,
                        "group": METRICS[str(row["metric"])][1],
                        **row,
                    }
                )

    source_by_cell = {
        cell: sorted({run.source_revision for run in runs if run.cell_id == cell})
        for cell in cells
    }
    evaluation_revision = next(iter(evaluation_revisions))
    evidence_rows = [
        {
            "run_key": run.run_key,
            "attempt": str(run.attempt),
            "source_attempt": str(run.source_attempt),
            "source_revision": run.source_revision,
            "evaluation_revision": run.evaluation_revision,
            "subset_sha256": run.subset_sha256,
            "selected_traj_id_sha256": run.selected_traj_id_sha256,
        }
        for run in runs
    ]
    paired = sorted(
        paired,
        key=lambda row: (
            str(row["city"]), str(row["cell_id"]), str(row["metric"])
        ),
    )
    staged_attribution = summarize_staged_attribution(
        runs,
        cells,
        cities,
        seeds,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    g_attribution = summarize_g_attribution(
        runs,
        cells,
        cities,
        seeds,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "candidate_evidence": CANDIDATE_EVIDENCE,
        "revision": evaluation_revision,
        "evaluation_revision": evaluation_revision,
        "source_revision_audit": {
            "uniform": len({run.source_revision for run in runs}) == 1,
            "by_cell": source_by_cell,
        },
        "bootstrap": {
            "method": SELECTION_BOOTSTRAP_METHOD,
            "n_resamples": n_resamples,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "unit": "trajectory_within_seed_equal_weight_across_seeds",
        },
        "coverage": {"runs": len(runs), "cities": cities, "seeds": seeds, "cells": cells},
        "stage1_gates": _stage1_gates(runs),
        "metric_rows": metric_rows,
        # Kept for per-run traceability; cross-seed claims must use g_attribution.
        "g_metric_attribution": g_rows,
        "g_attribution": g_attribution,
        "paired_metric_effects": paired,
        "city_direction_heterogeneity": summarize_city_direction_heterogeneity(paired),
        "staged_attribution": staged_attribution,
        "evidence_rows": evidence_rows,
    }


def write_corrected_report(summary: Mapping[str, object], output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    json_path = output / "analysis.json"
    markdown_path = output / "analysis.md"
    coverage = summary["coverage"]
    markdown = "\n".join(
        [
            "# Fixed-stratified population-aware result audit",
            "",
            f"- Evaluation revision: `{summary['evaluation_revision']}`",
            f"- Completed reevaluations: {coverage['runs']}",
            f"- Cities: {', '.join(coverage['cities'])}",
            f"- Seeds: {', '.join(str(value) for value in coverage['seeds'])}",
            f"- Cells: {', '.join(coverage['cells'])}",
            f"- Paired metric effects: {len(summary['paired_metric_effects'])}",
            f"- City-direction rows: {len(summary['city_direction_heterogeneity'])}",
            f"- Opposed city directions: "
            f"{sum(bool(row['opposed_direction']) for row in summary['city_direction_heterogeneity'])}",
            f"- G-attribution rows: {len(summary['g_metric_attribution'])}",
            f"- Cross-seed G-attribution rows: {len(summary['g_attribution']['rows'])}",
            f"- Cross-seed G opposed city directions: "
            f"{sum(bool(row['opposed_direction']) for row in summary['g_attribution']['city_direction_heterogeneity'])}",
            f"- E/Q attribution available: "
            f"{str(summary['staged_attribution']['available']).lower()}",
            f"- E/Q attribution rows: {len(summary['staged_attribution']['rows'])}",
            f"- E/Q opposed city directions: "
            f"{sum(bool(row['opposed_direction']) for row in summary['staged_attribution']['city_direction_heterogeneity'])}",
            "",
            "All candidate effects resample trajectories within seed and average seed effects equally.",
            "Primary G components use the same trajectory-paired, equal-seed-weight bootstrap.",
            "Registered E/Q contrasts reuse the same paired trajectory weights across all core cells.",
            "Failures remain in every primary population-level regret.",
            "",
        ]
    )
    write_immutable_artifact_set(
        {
            json_path: canonical_json_bytes(summary),
            markdown_path: markdown.encode("utf-8"),
        },
        label="corrected analysis",
    )
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fixed-stratified corrected reevaluations")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reevaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells", nargs="+")
    parser.add_argument("--allow-phase-source-revisions", action="store_true")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument("--confidence", type=float, default=0.95)
    arguments = parser.parse_args()
    runs = collect_corrected_runs(
        arguments.config,
        arguments.reevaluation_root,
        arguments.cells,
        allow_phase_source_revisions=arguments.allow_phase_source_revisions,
    )
    summary = summarize_corrected_results(
        runs,
        n_resamples=arguments.n_resamples,
        bootstrap_seed=arguments.bootstrap_seed,
        confidence=arguments.confidence,
    )
    json_path, markdown_path = write_corrected_report(summary, arguments.output_dir)
    print(
        json.dumps(
            {
                "analysis": str(json_path.resolve()),
                "analysis_sha256": _sha256(json_path),
                "markdown": str(markdown_path.resolve()),
                "markdown_sha256": _sha256(markdown_path),
                "coverage": summary["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
