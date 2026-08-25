from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .analyze import city_heterogeneity, e_contrasts, q_incremental_contrasts
from .config import ExperimentConfig
from .evaluate_stage1 import validity_gate
from .run_experiment import build_jobs, completed_attempt
from .statistics import aggregate_seeds, g_shapley, paired_bootstrap


G_VARIANTS = ("g00", "g10", "g01", "g11")
METRICS = {
    "population_mean_penalized_error": ("minimize", "GPS/Road composite"),
    "validity_rate": ("maximize", "validity"),
    "travel_distance_jsd": ("minimize", "GPS"),
    "segment_distance_jsd": ("minimize", "GPS"),
    "radius_jsd": ("minimize", "GPS"),
    "grid_density_cosine": ("maximize", "Grid"),
    "grid_pattern_f1": ("maximize", "Grid"),
    "road_density_cosine": ("maximize", "Road"),
    "road_pattern_f1": ("maximize", "Road"),
    "point_to_road_jsd": ("minimize", "Road"),
}


class ResultAuditError(RuntimeError):
    """Raised when result evidence is incomplete, stale, or incomparable."""


@dataclass(frozen=True)
class RunEvidence:
    run_key: str
    cell_id: str
    city: str
    seed: int
    budget: str
    attempt: Path
    revision: str
    stage4: Mapping[str, object]
    stage1: Mapping[str, object] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_stage_artifact(attempt: str | Path, stage: str, name: str) -> Path:
    root = Path(attempt).resolve()
    marker_path = root / stage / "complete.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        info = marker["artifacts"][name]
        path = Path(info["path"]).resolve()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ResultAuditError(f"invalid {stage} marker for {root}: {error}") from error
    if marker.get("stage") != stage:
        raise ResultAuditError(f"stage marker name mismatch for {root}: {marker.get('stage')!r}")
    if not path.is_relative_to(root):
        raise ResultAuditError(f"artifact escapes immutable attempt: {path}")
    if not path.is_file():
        raise ResultAuditError(f"missing artifact: {path}")
    if path.stat().st_size != int(info["size_bytes"]):
        raise ResultAuditError(f"artifact size mismatch: {path}")
    if _sha256(path) != str(info["sha256"]):
        raise ResultAuditError(f"artifact SHA256 mismatch: {path}")
    return path


def _validate_stage4(stage4: Mapping[str, object], label: str) -> None:
    if set(stage4) != set(G_VARIANTS):
        raise ResultAuditError(f"{label} G variants must be exactly {G_VARIANTS}")
    populations = []
    trajectory_ids = []
    for variant in G_VARIANTS:
        payload = stage4[variant]
        if not isinstance(payload, Mapping):
            raise ResultAuditError(f"{label}/{variant} is not a mapping")
        population = int(payload["population"])
        rows = payload["per_trajectory"]
        if not isinstance(rows, Sequence) or len(rows) != population:
            raise ResultAuditError(f"{label}/{variant} per-trajectory population mismatch")
        ids = [int(row["traj_id"]) for row in rows]
        if len(set(ids)) != len(ids):
            raise ResultAuditError(f"{label}/{variant} duplicate traj_id")
        populations.append(population)
        trajectory_ids.append(set(ids))
    if len(set(populations)) != 1 or any(ids != trajectory_ids[0] for ids in trajectory_ids[1:]):
        raise ResultAuditError(f"{label} G variants do not use the exact same population")


def collect_core_runs(
    config_path: str | Path,
    experiment_root: str | Path,
    budget: str,
    cell_ids: Sequence[str] | None = None,
) -> tuple[RunEvidence, ...]:
    config = ExperimentConfig.load(config_path)
    root = Path(experiment_root)
    missing = []
    evidence = []
    for job in build_jobs(config, budget, cell_ids):
        attempt = completed_attempt(root / job.run_key)
        if attempt is None:
            missing.append(job.run_key)
            continue
        manifest = json.loads((attempt / "run_manifest.json").read_text(encoding="utf-8"))
        expected = {
            "experiment_id": config.experiment_id,
            "cell_id": job.cell_id,
            "city": job.city,
            "seed": job.seed,
            "budget": job.budget,
            "status": "completed",
        }
        changed = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
        if changed:
            raise ResultAuditError(f"manifest identity mismatch for {job.run_key}: {changed}")
        if bool(manifest["git"]["dirty"]):
            raise ResultAuditError(f"completed evidence is dirty for {job.run_key}")
        revision = str(manifest["git"]["revision"])
        if len(revision) != 40:
            raise ResultAuditError(f"non-full revision for {job.run_key}")
        stage4_path = read_stage_artifact(attempt, "stage4", "stage4_metrics.json")
        stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
        _validate_stage4(stage4, job.run_key)
        stage1_path = read_stage_artifact(attempt, "stage1", "stage1_metrics.json")
        stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
        evidence.append(
            RunEvidence(
                job.run_key, job.cell_id, job.city, job.seed, job.budget,
                attempt, revision, stage4, stage1,
            )
        )
    if missing:
        raise ResultAuditError(f"missing completed core runs ({len(missing)}): {missing[:10]}")
    revisions = {run.revision for run in evidence}
    if len(revisions) != 1:
        raise ResultAuditError(f"core runs span incomparable revisions: {sorted(revisions)}")
    return tuple(evidence)


def _variant_metrics(payload: Mapping[str, object]) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "population_mean_penalized_error": float(payload["population_mean_penalized_error"]),
        "validity_rate": float(payload["validity_rate"]),
    }
    successful = payload["successful_only"]
    metrics = successful.get("metrics") if isinstance(successful, Mapping) else None
    for metric in set(METRICS) - set(result):
        value = metrics.get(metric) if isinstance(metrics, Mapping) else None
        result[metric] = None if value is None else float(value)
    for metric, value in result.items():
        if value is not None and not math.isfinite(value):
            raise ResultAuditError(f"non-finite Stage4 metric {metric}")
    return result


def _regret(metric: str, value: float) -> float:
    orientation = METRICS[metric][0]
    return float(value) if orientation == "minimize" else 1.0 - float(value)


def _penalty_map(payload: Mapping[str, object]) -> dict[int, float]:
    return {int(row["traj_id"]): float(row["penalized_error"]) for row in payload["per_trajectory"]}


def summarize_stage4(
    runs: Sequence[RunEvidence],
    n_resamples: int = 2000,
    bootstrap_seed: int = 20260826,
) -> dict[str, object]:
    if not runs:
        raise ResultAuditError("cannot summarize an empty run set")
    revisions = {run.revision for run in runs}
    if len(revisions) != 1:
        raise ResultAuditError("summary requires one frozen revision")
    baseline_oracle = {
        (run.city, run.seed): float(run.stage4["g00"]["population_mean_penalized_error"])
        for run in runs
        if run.cell_id == "e0_q0"
    }
    stage1_gates = []
    for run in runs:
        if not run.stage1:
            continue
        baseline = baseline_oracle.get((run.city, run.seed))
        if baseline is None:
            raise ResultAuditError(f"missing E0 oracle baseline for {run.city}/seed_{run.seed}")
        oracle_error = float(run.stage4["g00"]["population_mean_penalized_error"])
        ratio = 1.0 if baseline == 0 and oracle_error == 0 else (
            float("inf") if baseline == 0 else oracle_error / baseline
        )
        losses = run.stage1.get("losses", {})
        all_losses_finite = isinstance(losses, Mapping) and bool(losses) and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in losses.values()
        )
        gate = validity_gate(run.stage1["codebook"], all_losses_finite, ratio)
        stage1_gates.append(
            {
                "city": run.city, "seed": run.seed, "cell_id": run.cell_id,
                "oracle_error": oracle_error, "e0_oracle_error": baseline,
                "oracle_error_ratio": ratio, **gate,
            }
        )

    metric_rows = []
    lookup: dict[tuple[str, int, str, str, str], float] = {}
    for run in runs:
        _validate_stage4(run.stage4, run.run_key)
        for variant in G_VARIANTS:
            for metric, value in _variant_metrics(run.stage4[variant]).items():
                if value is None:
                    continue
                regret = _regret(metric, value)
                lookup[(run.city, run.seed, run.cell_id, variant, metric)] = regret
                metric_rows.append(
                    {
                        "city": run.city, "seed": run.seed, "cell_id": run.cell_id,
                        "variant": variant, "metric": metric, "group": METRICS[metric][1],
                        "orientation": METRICS[metric][0], "value": value, "regret": regret,
                    }
                )

    g_rows = []
    g_trajectory_bootstrap = []
    for run in runs:
        for metric in METRICS:
            keys = [(run.city, run.seed, run.cell_id, variant, metric) for variant in G_VARIANTS]
            if all(key in lookup for key in keys):
                attribution = g_shapley({variant: lookup[key] for variant, key in zip(G_VARIANTS, keys)})
                g_rows.append({"city": run.city, "seed": run.seed, "cell_id": run.cell_id, "metric": metric, **attribution})
        penalty = {variant: _penalty_map(run.stage4[variant]) for variant in G_VARIANTS}
        if any(set(values) != set(penalty["g00"]) for values in penalty.values()):
            raise ResultAuditError(f"G penalty population mismatch for {run.run_key}")
        components = {"sid": {}, "sampling": {}, "interaction": {}, "total": {}}
        for traj_id in penalty["g00"]:
            value = g_shapley({variant: penalty[variant][traj_id] for variant in G_VARIANTS})
            for component in components:
                components[component][traj_id] = value[component]
        zeros = {traj_id: 0.0 for traj_id in penalty["g00"]}
        for offset, (component, values) in enumerate(components.items()):
            estimate = paired_bootstrap(
                zeros, values, n_resamples=n_resamples,
                seed=bootstrap_seed + run.seed + offset,
            )
            g_trajectory_bootstrap.append(
                {"city": run.city, "seed": run.seed, "cell_id": run.cell_id, "component": component, **estimate}
            )

    e_q_rows = []
    city_seed = sorted({(run.city, run.seed) for run in runs})
    for city, seed in city_seed:
        for metric in METRICS:
            values = {
                run.cell_id: lookup[(city, seed, run.cell_id, "g11", metric)]
                for run in runs
                if run.city == city and run.seed == seed
                and (city, seed, run.cell_id, "g11", metric) in lookup
            }
            core_values = {cell: values[cell] for cell in (
                "e0_q0", "e1_q0", "e2_q0", "e3_q0", "e4_q0", "e5_q0",
                "e6_q0", "e6_q1", "e6_q2", "e6_q3",
            ) if cell in values}
            if len(core_values) != 10:
                continue
            for family, contrasts in (
                ("E", e_contrasts(core_values)),
                ("Q", q_incremental_contrasts(core_values)),
            ):
                for contrast, estimate in contrasts.items():
                    e_q_rows.append(
                        {"city": city, "seed": seed, "metric": metric, "family": family,
                         "contrast": contrast, "estimate": estimate}
                    )

    paired_e_q = []
    for city, seed in city_seed:
        city_runs = {run.cell_id: run for run in runs if run.city == city and run.seed == seed}
        if "e0_q0" not in city_runs:
            continue
        baseline = _penalty_map(city_runs["e0_q0"].stage4["g11"])
        for offset, cell_id in enumerate(sorted(set(city_runs) - {"e0_q0"})):
            candidate = _penalty_map(city_runs[cell_id].stage4["g11"])
            estimate = paired_bootstrap(
                baseline, candidate, n_resamples=n_resamples,
                seed=bootstrap_seed + seed + 100 + offset,
            )
            paired_e_q.append({"city": city, "seed": seed, "cell_id": cell_id, **estimate})

    seed_groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        seed_groups[(row["city"], row["cell_id"], row["variant"], row["metric"])].append(row["regret"])
    seed_aggregates = [
        {"city": key[0], "cell_id": key[1], "variant": key[2], "metric": key[3], **aggregate_seeds(values)}
        for key, values in sorted(seed_groups.items())
    ]

    contrast_groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in e_q_rows:
        contrast_groups[(row["city"], row["metric"], row["family"], row["contrast"])].append(row["estimate"])
    city_contrast_means = {key: float(sum(values) / len(values)) for key, values in contrast_groups.items()}
    heterogeneity = []
    contrast_ids = sorted({(key[1], key[2], key[3]) for key in city_contrast_means})
    for metric, family, contrast in contrast_ids:
        values = {
            city: estimate
            for (city, current_metric, current_family, current_contrast), estimate in city_contrast_means.items()
            if (current_metric, current_family, current_contrast) == (metric, family, contrast)
        }
        if len(values) >= 2:
            heterogeneity.append(
                {"metric": metric, "family": family, "contrast": contrast, **city_heterogeneity(values)}
            )

    return {
        "schema_version": 1,
        "revision": next(iter(revisions)),
        "coverage": {
            "runs": len(runs),
            "cities": sorted({run.city for run in runs}),
            "seeds": sorted({run.seed for run in runs}),
            "cells": sorted({run.cell_id for run in runs}),
        },
        "stage1_gates": stage1_gates,
        "metric_rows": metric_rows,
        "g_shapley": g_rows,
        "g_trajectory_bootstrap": g_trajectory_bootstrap,
        "e_q_contrasts": e_q_rows,
        "paired_e_q_penalized_error": paired_e_q,
        "seed_aggregates": seed_aggregates,
        "city_heterogeneity": heterogeneity,
    }


def render_markdown(summary: Mapping[str, object]) -> str:
    coverage = summary["coverage"]
    lines = [
        "# GPS / Road / Grid staged-attribution result audit",
        "",
        f"- Frozen revision: `{summary['revision']}`",
        f"- Completed runs: {coverage['runs']}",
        f"- Cities: {', '.join(coverage['cities'])}",
        f"- Seeds: {', '.join(str(value) for value in coverage['seeds'])}",
        f"- Core cells: {len(coverage['cells'])}",
        "",
        "This file is an evidence index. Scientific selection and claims must use confirmation results.",
        "",
        "## Available audited analyses",
        "",
        f"- Stage4 metric rows: {len(summary['metric_rows'])}",
        f"- G Shapley rows: {len(summary['g_shapley'])}",
        f"- Trajectory-bootstrap G rows: {len(summary['g_trajectory_bootstrap'])}",
        f"- E/Q contrast rows: {len(summary['e_q_contrasts'])}",
        f"- Paired E/Q penalty rows: {len(summary['paired_e_q_penalized_error'])}",
        f"- City heterogeneity rows: {len(summary['city_heterogeneity'])}",
        "",
    ]
    return "\n".join(lines)


def write_report(summary: Mapping[str, object], output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "analysis.json"
    markdown_path = output / "analysis.md"
    json_tmp = json_path.with_suffix(".json.tmp")
    md_tmp = markdown_path.with_suffix(".md.tmp")
    json_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_tmp.write_text(render_markdown(summary), encoding="utf-8")
    json_tmp.replace(json_path)
    md_tmp.replace(markdown_path)
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and summarize completed staged-attribution runs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--budget", choices=("smoke", "controlled", "confirmation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument("--cells", nargs="+")
    arguments = parser.parse_args()
    runs = collect_core_runs(
        arguments.config, arguments.experiment_root, arguments.budget, arguments.cells
    )
    summary = summarize_stage4(runs, arguments.n_resamples, arguments.bootstrap_seed)
    paths = write_report(summary, arguments.output_dir)
    print(json.dumps({"analysis_json": str(paths[0]), "analysis_markdown": str(paths[1])}, indent=2))


if __name__ == "__main__":
    main()
