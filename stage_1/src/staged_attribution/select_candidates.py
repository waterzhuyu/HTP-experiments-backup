from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .analyze import pareto_front, staged_contrast_definitions
from .immutable_artifacts import canonical_json_bytes, write_immutable_artifact_set
from .report_results import G_VARIANTS, METRICS
from .population_bootstrap import POPULATION_REGRET_METHOD, SELECTION_BOOTSTRAP_METHOD


class SelectionError(RuntimeError):
    """Raised when a candidate decision cannot be reproduced from audited evidence."""


DOMAIN_GROUPS = frozenset({"GPS", "Road", "Grid"})
CORE_CELLS = (
    "e0_q0", "e1_q0", "e2_q0", "e3_q0", "e4_q0", "e5_q0",
    "e6_q0", "e6_q1", "e6_q2", "e6_q3",
)
G_COMPONENTS = ("sid", "sampling", "interaction", "total")


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


def _audit_staged_attribution(summary: Mapping[str, object]) -> dict[str, object]:
    staged = summary.get("staged_attribution")
    coverage = summary.get("coverage")
    if not isinstance(staged, Mapping) or not isinstance(coverage, Mapping):
        raise SelectionError("candidate selection requires corrected E/Q staged attribution")
    cells = coverage.get("cells")
    cities = coverage.get("cities")
    if (
        staged.get("available") is not True
        or staged.get("required_cells") != list(CORE_CELLS)
        or staged.get("missing_cells") != []
        or not isinstance(cells, list)
        or not set(CORE_CELLS).issubset(map(str, cells))
        or not isinstance(cities, list)
        or not cities
    ):
        raise SelectionError("candidate selection requires complete registered core E/Q coverage")
    definitions = staged_contrast_definitions()
    if staged.get("contrast_definitions") != definitions:
        raise SelectionError("corrected E/Q contrast definitions differ from registration")
    rows = staged.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise SelectionError("corrected E/Q attribution rows are missing")
    definition_by_name = {
        str(definition["contrast"]): definition for definition in definitions
    }
    expected = {
        (str(city), str(definition["contrast"]), metric)
        for city in cities
        for definition in definitions
        for metric in METRICS
    }
    observed = set()
    for source in rows:
        if not isinstance(source, Mapping):
            raise SelectionError("corrected E/Q attribution row must be a mapping")
        key = (str(source.get("city")), str(source.get("contrast")), str(source.get("metric")))
        if key in observed or key[1] not in definition_by_name:
            raise SelectionError(f"duplicate or unknown corrected E/Q attribution row: {key}")
        observed.add(key)
        definition = definition_by_name[key[1]]
        metric = key[2]
        if (
            source.get("family") != definition["family"]
            or source.get("contrast_kind") != definition["kind"]
            or source.get("inferential_role") != definition["role"]
            or source.get("method") != SELECTION_BOOTSTRAP_METHOD
            or metric not in METRICS
            or source.get("group") != METRICS.get(metric, (None, None))[1]
            or int(source.get("n_pairs", 0)) <= 0
            or int(source.get("n_resamples", 0)) <= 0
        ):
            raise SelectionError(f"invalid corrected E/Q attribution identity: {key}")
        if definition["kind"] == "linear":
            if source.get("coefficients") != definition["coefficients"]:
                raise SelectionError(f"corrected E/Q coefficients differ from registration: {key}")
        elif source.get("expression") != definition["expression"]:
            raise SelectionError(f"corrected E/Q expression differs from registration: {key}")
        for field in ("estimate", "ci_low", "ci_high", "p_improvement", "p_degradation"):
            if not isinstance(source.get(field), (int, float)) or not math.isfinite(float(source[field])):
                raise SelectionError(f"invalid corrected E/Q {field}: {key}")
    if observed != expected:
        raise SelectionError("corrected E/Q attribution lacks complete city/contrast/metric coverage")
    return {
        "available": True,
        "families": ["E", "Q"],
        "contrasts": len(definitions),
        "rows": len(rows),
    }


def audit_g_attribution(summary: Mapping[str, object]) -> dict[str, object]:
    attribution = summary.get("g_attribution")
    coverage = summary.get("coverage")
    if not isinstance(attribution, Mapping) or not isinstance(coverage, Mapping):
        raise SelectionError("candidate selection requires cross-seed G attribution")
    cities = coverage.get("cities")
    seeds = coverage.get("seeds")
    cells = coverage.get("cells")
    if (
        attribution.get("available") is not True
        or attribution.get("required_variants") != list(G_VARIANTS)
        or attribution.get("components") != list(G_COMPONENTS)
        or attribution.get("method") != SELECTION_BOOTSTRAP_METHOD
        or attribution.get("unit") != "trajectory_within_seed_equal_weight_across_seeds"
        or not isinstance(cities, list)
        or not cities
        or not isinstance(seeds, list)
        or not seeds
        or not isinstance(cells, list)
        or not cells
        or attribution.get("observed_cells") != cells
    ):
        raise SelectionError("cross-seed G attribution identity is incomplete")
    rows = attribution.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise SelectionError("cross-seed G attribution rows are missing")
    expected = {
        (str(city), str(cell), metric, component)
        for city in cities
        for cell in cells
        for metric in METRICS
        for component in G_COMPONENTS
    }
    observed = set()
    expected_seed_ids = {int(value) for value in seeds}
    for source in rows:
        if not isinstance(source, Mapping):
            raise SelectionError("cross-seed G attribution row must be a mapping")
        key = (
            str(source.get("city")), str(source.get("cell_id")),
            str(source.get("metric")), str(source.get("component")),
        )
        if key in observed:
            raise SelectionError(f"duplicate cross-seed G attribution row: {key}")
        observed.add(key)
        metric = key[2]
        populations = source.get("seed_populations")
        try:
            population_seed_ids = {
                int(value) for value in populations
            } if isinstance(populations, Mapping) else set()
        except (TypeError, ValueError):
            population_seed_ids = set()
        if (
            source.get("method") != SELECTION_BOOTSTRAP_METHOD
            or source.get("inferential_role") != "primary"
            or source.get("orientation") != "minimize"
            or metric not in METRICS
            or source.get("group") != METRICS.get(metric, (None, None))[1]
            or key[3] not in G_COMPONENTS
            or source.get("seed_strata") != seeds
            or population_seed_ids != expected_seed_ids
            or not all(int(value) > 0 for value in populations.values())
            or int(source.get("n_pairs", 0)) <= 0
            or int(source.get("n_resamples", 0)) <= 0
        ):
            raise SelectionError(f"invalid cross-seed G attribution identity: {key}")
        for field in ("estimate", "ci_low", "ci_high", "p_improvement", "p_degradation"):
            if not isinstance(source.get(field), (int, float)) or not math.isfinite(float(source[field])):
                raise SelectionError(f"invalid cross-seed G {field}: {key}")
    if observed != expected:
        raise SelectionError("cross-seed G attribution lacks complete city/cell/metric/component coverage")
    return {
        "available": True,
        "components": list(G_COMPONENTS),
        "cells": len(cells),
        "rows": len(rows),
    }


def select_candidates(summary: Mapping[str, object], phase: str = "core") -> dict[str, object]:
    if phase not in {"core", "final"}:
        raise SelectionError("selection phase must be core or final")
    if int(summary.get("analysis_schema_version", 0)) < 5:
        raise SelectionError("candidate selection requires corrected analysis schema version 5")
    if summary.get("candidate_evidence") != "stratified_population_aware":
        raise SelectionError("candidate selection requires stratified population-aware evidence")
    staged_audit = _audit_staged_attribution(summary)
    g_audit = audit_g_attribution(summary)
    coverage = summary.get("coverage")
    if (
        phase == "core"
        and isinstance(coverage, Mapping)
        and set(map(str, coverage.get("cells", []))) != set(CORE_CELLS)
    ):
        raise SelectionError("core selection requires exactly the registered core cells")
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
        "staged_attribution_audit": staged_audit,
        "g_attribution_audit": g_audit,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionError(f"{label} must be a JSON object: {path}")
    return value


def _validate_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise SelectionError(f"{label} must be a full lowercase SHA-256 digest")
    return normalized


def _existing_non_symlink_file(value: str | Path, message: str) -> Path:
    """Return a canonical file path without hiding a symlink at the leaf."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise SelectionError(message)
    return candidate.resolve()


def _audit_final_selection_chain(
    selection: Mapping[str, object],
    source: Mapping[str, object],
) -> None:
    """Revalidate the frozen core decision that constrained a final selection."""
    prior_path = selection.get("prior_selection")
    prior_sha = selection.get("prior_selection_sha256")
    prior_cells = selection.get("prior_adaptive_cells")
    if not isinstance(prior_path, str) or not isinstance(prior_sha, str):
        raise SelectionError("final selection lacks its frozen core-selection identity")
    prior = audit_selection_derivation(prior_path, prior_sha, "core")
    if prior_cells != prior.get("adaptive_cells"):
        raise SelectionError("final/core adaptive-cell identity mismatch")
    if prior.get("source_revision") != selection.get("source_revision"):
        raise SelectionError("final/core selection revision mismatch")
    coverage = source.get("coverage")
    observed_cells = coverage.get("cells") if isinstance(coverage, Mapping) else None
    expected_cells = set(CORE_CELLS) | {
        str(value) for value in prior.get("adaptive_cells", [])
    }
    if (
        not isinstance(observed_cells, list)
        or set(map(str, observed_cells)) != expected_cells
    ):
        raise SelectionError(
            "final analysis coverage must exactly equal core plus frozen adaptive cells"
        )


def audit_selection_derivation(
    selection_path: str | Path,
    expected_sha256: str,
    expected_phase: str,
) -> dict[str, object]:
    """Verify that a frozen selection is an immutable deterministic analysis product."""
    if expected_phase not in {"core", "final"}:
        raise SelectionError("expected selection phase must be core or final")
    expected_digest = _validate_sha256(expected_sha256, "expected selection SHA-256")
    path = _existing_non_symlink_file(
        selection_path,
        "frozen selection is missing, symlinked, or hash-mismatched",
    )
    if _sha256(path) != expected_digest:
        raise SelectionError("frozen selection is missing, symlinked, or hash-mismatched")
    selection = _read_json_object(path, "frozen selection")
    if selection.get("phase") != expected_phase:
        raise SelectionError("frozen selection phase mismatch")
    source_path = _existing_non_symlink_file(
        str(selection.get("source_analysis", "")),
        "selection source analysis is missing, symlinked, or hash-mismatched",
    )
    source_digest = str(selection.get("source_analysis_sha256", ""))
    if len(source_digest) != 64 or _sha256(source_path) != source_digest:
        raise SelectionError(
            "selection source analysis is missing, symlinked, or hash-mismatched"
        )
    source = _read_json_object(source_path, "selection source analysis")
    if (
        source.get("candidate_evidence") != "stratified_population_aware"
        or source.get("revision") != selection.get("source_revision")
    ):
        raise SelectionError("selection source is not audited stratified population evidence")
    recomputed = select_candidates(source, expected_phase)
    changed = {
        field: (selection.get(field), expected)
        for field, expected in recomputed.items()
        if selection.get(field) != expected
    }
    if changed:
        raise SelectionError(
            f"frozen selection differs from deterministic recomputation: {sorted(changed)}"
        )
    if expected_phase == "final":
        _audit_final_selection_chain(selection, source)
    return selection


def freeze_selection(
    analysis_path: str | Path,
    phase: str,
    *,
    prior_selection_path: str | Path | None = None,
    expected_prior_selection_sha256: str | None = None,
) -> dict[str, object]:
    """Build a selection and bind final selection to the preceding core decision."""
    analysis = _existing_non_symlink_file(
        analysis_path,
        "selection analysis must be an existing non-symlink file",
    )
    summary = _read_json_object(analysis, "selection analysis")
    selection = select_candidates(summary, phase)
    selection["source_analysis"] = str(analysis)
    selection["source_analysis_sha256"] = _sha256(analysis)
    if phase == "core":
        if prior_selection_path is not None or expected_prior_selection_sha256 is not None:
            raise SelectionError("core selection must not declare a prior selection")
        return selection
    if prior_selection_path is None or expected_prior_selection_sha256 is None:
        raise SelectionError("final selection requires the frozen core selection and SHA-256")
    prior = audit_selection_derivation(
        prior_selection_path,
        expected_prior_selection_sha256,
        "core",
    )
    if prior.get("source_revision") != selection.get("source_revision"):
        raise SelectionError("final/core selection revision mismatch")
    coverage = summary.get("coverage")
    observed_cells = coverage.get("cells") if isinstance(coverage, Mapping) else None
    expected_cells = set(CORE_CELLS) | {
        str(value) for value in prior.get("adaptive_cells", [])
    }
    if not isinstance(observed_cells, list) or set(map(str, observed_cells)) != expected_cells:
        raise SelectionError(
            "final analysis coverage must exactly equal core plus frozen adaptive cells"
        )
    prior_path = Path(prior_selection_path).resolve()
    selection["prior_selection"] = str(prior_path)
    selection["prior_selection_sha256"] = _sha256(prior_path)
    selection["prior_adaptive_cells"] = list(prior.get("adaptive_cells", []))
    return selection


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
    parser.add_argument("--prior-selection", type=Path)
    parser.add_argument("--expected-prior-selection-sha256")
    arguments = parser.parse_args()
    selection = freeze_selection(
        arguments.analysis,
        arguments.phase,
        prior_selection_path=arguments.prior_selection,
        expected_prior_selection_sha256=arguments.expected_prior_selection_sha256,
    )
    write_selection(selection, arguments.output)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
