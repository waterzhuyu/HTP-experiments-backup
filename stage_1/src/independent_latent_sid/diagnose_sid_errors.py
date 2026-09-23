"""Frozen, paired SID error diagnostic. Does not load or update a model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from hierarchical_rqvae.analyze import digest, read
from hierarchical_rqvae.preflight import atomic_json
from independent_latent_sid.analyze_results import CELLS, verify_completed


def jsonl_by_id(path: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [int(row["traj_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate trajectory in {path}")
    return dict(zip(ids, rows, strict=True))


def classify(target: list[list[int]], generated: list[list[int]] | None) -> dict:
    if not target or any(len(row) != 4 for row in target):
        raise ValueError("invalid target SID shape")
    if generated is not None and any(len(row) != 4 for row in generated):
        raise ValueError("invalid generated SID shape")
    generated = generated or []
    layer_wrong = [0] * 4
    tuple_correct = 0
    first_error = None
    for event, correct in enumerate(target):
        observed = generated[event] if event < len(generated) else None
        if observed == correct:
            tuple_correct += 1
        for layer in range(4):
            wrong = observed is None or observed[layer] != correct[layer]
            layer_wrong[layer] += int(wrong)
            if wrong and first_error is None:
                first_error = [event, layer + 1]
    return {
        "target_events": len(target),
        "generated_events": len(generated),
        "missing_events": max(0, len(target) - len(generated)),
        "extra_events": max(0, len(generated) - len(target)),
        "layer_wrong": layer_wrong,
        "tuple_correct": tuple_correct,
        "first_error": first_error,
    }


def bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 4:
        return "2-4"
    return "5+"


def diagnose_cell(run: Path, ids: tuple[int, ...]) -> dict:
    targets = jsonl_by_id(run / "tokens/test_tokens.jsonl")
    generated = jsonl_by_id(run / "stage3/generations.jsonl")
    if set(targets) != set(ids) or set(generated) != set(ids):
        raise ValueError("target or generation population mismatch")
    metrics = read(run / "stage4/stage4_metrics.json")
    scores = {}
    for group in ("g00", "g10", "g11"):
        rows = metrics[group]["per_trajectory"]
        mapped = {int(row["traj_id"]): row for row in rows}
        if len(rows) != len(ids) or set(mapped) != set(ids):
            raise ValueError(f"stage4 {group} population mismatch")
        scores[group] = mapped

    items = []
    totals = [0] * 4
    target_events = extra_events = missing_events = tuple_correct = 0
    failures = Counter()
    first = Counter()
    for tid in ids:
        truth = targets[tid]
        prediction = generated[tid]
        failure = prediction.get("failure")
        if failure is not None:
            failures[str(failure)] += 1
        raw_codes = prediction.get("codes")
        if raw_codes is None and failure is None:
            raise ValueError("missing generated codes without failure")
        row = classify(truth["codes"], raw_codes)
        for layer in range(4):
            totals[layer] += row["layer_wrong"][layer]
        target_events += row["target_events"]
        missing_events += row["missing_events"]
        extra_events += row["extra_events"]
        tuple_correct += row["tuple_correct"]
        first[str(row["first_error"])] += 1
        row.update(
            traj_id=tid,
            generation_failure=failure,
            g00_error=float(scores["g00"][tid]["penalized_error"]),
            g10_error=float(scores["g10"][tid]["penalized_error"]),
            g11_error=float(scores["g11"][tid]["penalized_error"]),
        )
        row["sid_gap"] = row["g10_error"] - row["g00_error"]
        row["total_wrong"] = sum(row["layer_wrong"])
        items.append(row)

    strata = {}
    for name in ("0", "1", "2-4", "5+"):
        group = [row for row in items if bucket(row["total_wrong"]) == name]
        strata[name] = {
            "trajectories": len(group),
            "mean_sid_gap": float(np.mean([r["sid_gap"] for r in group])) if group else None,
            "mean_g10_error": float(np.mean([r["g10_error"] for r in group])) if group else None,
        }
    for group in ("g00", "g10", "g11"):
        observed = float(np.mean([r[f"{group}_error"] for r in items]))
        expected = float(metrics[group]["population_mean_penalized_error"])
        if abs(observed - expected) > 1e-10:
            raise ValueError(f"{group} metric parity mismatch")
    return {
        "population": len(ids),
        "target_events": target_events,
        "layer_wrong": totals,
        "layer_accuracy": [1 - n / target_events for n in totals],
        "tuple_accuracy": tuple_correct / target_events,
        "missing_events": missing_events,
        "extra_events": extra_events,
        "generation_failures": dict(failures),
        "first_error": dict(first),
        "wrong_token_strata": strata,
        "per_trajectory": items,
    }


def run(root: Path, config: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("refusing overwrite")
    plan = read(config)
    preflight = Path(plan["ablation"]["preflight_path"]) / "manifest.json"
    preflight_sha = digest(preflight)
    config_sha = digest(config)
    if preflight_sha != plan["ablation"]["preflight_sha256"]:
        raise ValueError("preflight SHA mismatch")
    ids = tuple(sorted(map(int, read(preflight)["validation_ids"])))
    if len(ids) != 1024 or len(set(ids)) != 1024:
        raise ValueError("wrong validation population")
    paths = {cell: root / "pilot/porto/42" / cell / "attempt_001" for cell in CELLS}
    manifests = {}
    artifact_counts = {}
    for cell in CELLS:
        manifests[cell], artifact_counts[cell] = verify_completed(
            paths[cell], next(iter(manifests.values()), None), config_sha, preflight_sha, cell
        )
    cells = {cell: diagnose_cell(paths[cell], ids) for cell in CELLS}
    result = {
        "status": "completed_frozen_diagnostic",
        "analysis_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv,
        "config_sha256": config_sha,
        "preflight_sha256": preflight_sha,
        "source_manifests": {
            cell: {
                "sha256": digest(paths[cell] / "run_manifest.json"),
                "training_revision": manifests[cell]["revision"],
                "verified_artifact_count": artifact_counts[cell],
            }
            for cell in CELLS
        },
        "alignment": "target event index; missing events wrong, extra events reported separately",
        "scope": "descriptive token errors and G10-G00 association; no decoder intervention or causal attribution",
        "cells": cells,
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("dirty analysis workspace")
    result = run(args.root, args.config, args.output)
    print(json.dumps({cell: {k: v for k, v in data.items() if k != "per_trajectory"}
                      for cell, data in result["cells"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
