from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .analyze import pareto_front
from .immutable_artifacts import canonical_json_bytes, write_immutable_artifact_set
from .report_results import METRICS
from .population_bootstrap import POPULATION_REGRET_METHOD, SELECTION_BOOTSTRAP_METHOD


class SelectionError(RuntimeError):
    """Raised when a candidate decision cannot be reproduced from audited evidence."""


DOMAIN_GROUPS = frozenset({"GPS", "Road", "Grid"})
CORE_CELLS = (
    "e0_q0", "e1_q0", "e2_q0", "e3_q0", "e4_q0", "e5_q0",
    "e6_q0", "e6_q1", "e6_q2", "e6_q3",
)


def _metric_lookup(summary: Mapping[str, object]) -> dict[tuple[str, int, str, str], float]:
    lookup = {}
    for row in summary["metric_rows"]:
        if row["variant"] != "g11":
            continue
        key = (str(row["city"]), int(row["seed"]), str(row["cell_id"]), str(row["metric"]))
        if key in lookup:
            raise SelectionError(f"duplicate G11 metric row: {key}")
        lookup[key] = float(row["regret"])
    return lookup


def _paired_effects(
    summary: Mapping[str, object],
    reference: str = "e0_q0",
) -> list[dict[str, object]]:
    raw = summary.get("paired_metric_effects")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise SelectionError("candidate selection requires corrected paired_metric_effects")
    result: list[dict[str, object]] = []
    keys = set()
    for source in raw:
        if not isinstance(source, Mapping):
            raise SelectionError("paired metric effect row must be a mapping")
        row = dict(source)
        if row.get("method") != SELECTION_BOOTSTRAP_METHOD:
            raise SelectionError("paired metric effect does not use the registered trajectory bootstrap")
        if row.get("reference") != reference:
            raise SelectionError("paired metric effect uses an unexpected reference")
        metric = str(row.get("metric"))
        if metric not in METRICS:
            raise SelectionError(f"unknown paired metric {metric}")
        if row.get("group") != METRICS[metric][1]:
            raise SelectionError(f"paired metric group mismatch for {metric}")
        key = (str(row.get("city")), str(row.get("cell_id")), metric)
        if key in keys:
            raise SelectionError(f"duplicate paired metric effect {key}")
        keys.add(key)
        for field in ("estimate", "ci_low", "ci_high", "p_improvement", "p_degradation"):
            if not isinstance(row.get(field), (int, float)) or not math.isfinite(float(row[field])):
                raise SelectionError(f"paired metric effect has invalid {field}: {key}")
        if int(row.get("n_pairs", 0)) <= 0 or int(row.get("n_resamples", 0)) <= 0:
            raise SelectionError(f"paired metric effect lacks population/bootstrap coverage: {key}")
        result.append(row)
    cells = sorted({key[1] for key in keys})
    cities = sorted({key[0] for key in keys})
    expected = {
        (city, cell, metric)
        for city in cities for cell in cells for metric in METRICS
    }
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise SelectionError(f"incomplete paired metric effect Cartesian coverage: missing={missing[:5]} extra={extra[:5]}")
    return sorted(result, key=lambda row: (str(row["city"]), str(row["cell_id"]), str(row["metric"])))


def _gate_map(summary: Mapping[str, object]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in summary.get("stage1_gates", []):
        grouped[str(row["cell_id"])].append(row)
    return {
        cell: {
            "runs": len(rows),
            "passed": bool(rows) and all(bool(row["passed"]) for row in rows),
            "failed_runs": [
                {"city": row["city"], "seed": row["seed"], "reasons": row["reasons"]}
                for row in rows if not row["passed"]
            ],
        }
        for cell, rows in grouped.items()
    }


def _eligibility(
    cells: Sequence[str],
    effects: Sequence[Mapping[str, object]],
    gates: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    by_cell: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in effects:
        by_cell[str(row["cell_id"])].append(row)
    result = {}
    for cell in cells:
        rows = by_cell.get(cell, [])
        significant_degradation = [
            str(row["metric"]) for row in rows
            if row["group"] in DOMAIN_GROUPS | {"validity"} and float(row["ci_low"]) > 0
        ]
        significant_improvement = [
            str(row["metric"]) for row in rows
            if row["group"] in DOMAIN_GROUPS and float(row["ci_high"]) < 0
        ]
        gate_passed = bool(gates.get(cell, {}).get("passed", False))
        result[cell] = {
            "stage1_gate_passed": gate_passed,
            "significant_degradation": significant_degradation,
            "significant_improvement": significant_improvement,
            "eligible": gate_passed and not significant_degradation and bool(significant_improvement),
        }
    return result


def _mean_profiles(summary: Mapping[str, object]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in summary["metric_rows"]:
        if row["variant"] == "g11" and row["group"] in DOMAIN_GROUPS:
            if row.get("method") != POPULATION_REGRET_METHOD:
                raise SelectionError("domain profile is not population-aware primary evidence")
            grouped[(str(row["cell_id"]), str(row["metric"]))].append(float(row["regret"]))
    return {
        cell: {
            metric: float(np.mean(values))
            for (current_cell, metric), values in grouped.items()
            if current_cell == cell
        }
        for cell in sorted({key[0] for key in grouped})
    }


def _rank_family(
    cells: Sequence[str],
    profiles: Mapping[str, Mapping[str, float]],
    gates: Mapping[str, Mapping[str, object]],
) -> str | None:
    available = [
        cell for cell in cells
        if cell in profiles and len(profiles[cell]) == len(DOMAIN_GROUPS) + 5
        and bool(gates.get(cell, {}).get("passed", False))
    ]
    if not available:
        return None
    metrics = sorted(next(iter(profiles.values())))
    matrix = np.asarray([[profiles[cell][metric] for metric in metrics] for cell in available])
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (matrix - matrix.min(axis=0)) / scale
    score = {
        cell: (float(standardized[index].max()), float(standardized[index].sum()), cell)
        for index, cell in enumerate(available)
    }
    return min(available, key=score.__getitem__)


def select_candidates(summary: Mapping[str, object], phase: str = "core") -> dict[str, object]:
    if phase not in {"core", "final"}:
        raise SelectionError("selection phase must be core or final")
    if int(summary.get("analysis_schema_version", 0)) < 2:
        raise SelectionError("candidate selection requires corrected analysis schema version 2")
    if summary.get("candidate_evidence") != "stratified_population_aware":
        raise SelectionError("candidate selection requires stratified population-aware evidence")
    effects = _paired_effects(summary)
    gates = _gate_map(summary)
    profiles = _mean_profiles(summary)
    cells = sorted(profiles)
    eligibility = _eligibility([cell for cell in cells if cell != "e0_q0"], effects, gates)
    eligible_profiles = {cell: profiles[cell] for cell in cells if eligibility.get(cell, {}).get("eligible")}
    pareto = pareto_front(eligible_profiles, minimize=sorted(METRICS.keys() - {"validity_rate", "population_mean_penalized_error"}), maximize=[])

    best_single = _rank_family(("e0_q0", "e1_q0", "e2_q0"), profiles, gates)
    best_pair = _rank_family(("e3_q0", "e4_q0", "e5_q0"), profiles, gates)
    best_q = _rank_family(("e6_q0", "e6_q1", "e6_q2", "e6_q3"), profiles, gates)
    adaptive = []
    if best_q is not None:
        q_id = best_q.split("_")[1]
        if q_id != "q0":
            for base in (best_single, best_pair):
                if base is None:
                    continue
                cell = f"{base.split('_')[0]}_{q_id}"
                if cell not in profiles:
                    adaptive.append(cell)

    confirmation = ["e0_q0"]
    if phase == "final":
        confirmation.extend(cell for cell in pareto if cell != "e0_q0")
    return {
        "schema_version": 1,
        "phase": phase,
        "source_revision": summary["revision"],
        "best_single": best_single,
        "best_pair": best_pair,
        "best_q": best_q,
        "adaptive_cells": sorted(set(adaptive)),
        "pareto_eligible_cells": pareto,
        "confirmation_cells": sorted(set(confirmation), key=lambda value: (value != "e0_q0", value)),
        "eligibility": eligibility,
        "paired_effects": effects,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_selection(selection: Mapping[str, object], output: str | Path) -> Path:
    path = Path(output)
    write_immutable_artifact_set(
        {path: canonical_json_bytes(selection)},
        label="candidate selection",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze adaptive/confirmation cell selection")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--phase", choices=("core", "final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = json.loads(arguments.analysis.read_text(encoding="utf-8"))
    selection = select_candidates(summary, arguments.phase)
    selection["source_analysis"] = str(arguments.analysis.resolve())
    selection["source_analysis_sha256"] = _sha256(arguments.analysis)
    write_selection(selection, arguments.output)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
