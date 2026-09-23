"""Production-singleton, first-wrong-code intervention on frozen SID generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from hierarchical_rqvae.analyze import digest, read
from hierarchical_rqvae.preflight import atomic_json
from independent_latent_sid.analyze_results import verify_completed
from independent_latent_sid.backend import Backend
from staged_attribution.data import make_collator
from staged_attribution.decode_stage3 import decode_variant_records, reconstruct_gps
from staged_attribution.stage4_metrics import trajectory_sufficient_statistics


METRICS = ("travel_distance_km", "mean_segment_distance_km", "radius_km", "mean_point_to_road_km")


def first_wrong(target: list[list[int]], generated: list[list[int]], layer: int | None):
    for event in range(min(len(target), len(generated))):
        indices = range(4) if layer is None else (layer,)
        if any(target[event][index] != generated[event][index] for index in indices):
            return event
    return None


def score(real: dict, generated: dict) -> float:
    return sum(abs(math.log1p(float(real[key])) - math.log1p(float(generated[key]))) for key in METRICS) / 4


@torch.no_grad()
def decode_one(model, record, context, geometry, scale):
    decoded = decode_variant_records(model, [record], {int(record["traj_id"]): context}, next(model.parameters()).device)[0]
    return reconstruct_gps(decoded, geometry, scale)


def compare_saved(actual: dict, saved: dict):
    if actual["status"] != saved["status"] or actual.get("decoded_length") != saved.get("decoded_length"):
        raise ValueError("production decode status/length parity failed")
    maxima = {}
    for key in ("pred_percent", "pred_dxy", "gps_traj"):
        maxima[key] = float(np.max(np.abs(np.asarray(actual[key]) - np.asarray(saved[key]))))
    if maxima["pred_percent"] > 1e-5 or maxima["pred_dxy"] > 1e-5 or maxima["gps_traj"] > 1e-7:
        raise ValueError(f"production decode parity failed: {maxima}")
    return maxima


def run_cell(backend, run_dir: Path, ids: list[int]) -> dict:
    model = backend._load_model(run_dir).eval().to(backend.device)
    dataset = backend._dataset("test")
    items = {int(item["traj_id"]): item for item in (dataset[i] for i in range(len(dataset)))}
    if len(items) != 1024:
        raise ValueError("wrong validation population")
    variants = read(run_dir / "g_attribution/g_variants.json")
    generated = {int(r["traj_id"]): r for r in variants["g10"]}
    target = {int(r["traj_id"]): r for r in variants["g00"]}
    saved = {int(r["traj_id"]): r for r in map(json.loads, (run_dir / "decode/g10_reconstructed.jsonl").read_text().splitlines())}
    stage4 = {int(r["traj_id"]): r for r in read(run_dir / "stage4/stage4_metrics.json")["g10"]["per_trajectory"]}
    if any(set(d) != set(items) for d in (generated, target, saved, stage4)):
        raise ValueError("population mismatch")
    geometry = backend._road_geometries()
    scale = backend._context().z_score
    collate = make_collator(backend._context())
    result = {"cell": backend.cell_id, "sample_ids": ids, "rows": [], "skipped": {}, "parity_max": {key: 0.0 for key in ("pred_percent", "pred_dxy", "gps_traj")}}
    started = time.monotonic()
    for tid in ids:
        source = generated[tid]
        if source["status"] != "ready" or saved[tid]["status"] != "reconstructed":
            result["skipped"][str(tid)] = "baseline G10 invalid"
            continue
        context_batch = collate([items[tid]])
        context = {name: context_batch[name] for name in ("route", "route_lengths", "route_geometry", "route_geometry_lengths")}
        original = decode_one(model, source, context, geometry, scale)
        maxima = compare_saved(original, saved[tid])
        for key, value in maxima.items():
            result["parity_max"][key] = max(value, result["parity_max"][key])
        if abs(score(stage4[tid]["real"], stage4[tid]["generated"]) - stage4[tid]["penalized_error"]) > 1e-10:
            raise ValueError("original G10 score parity failed")
        true_codes = target[tid]["codes"]
        baseline = float(stage4[tid]["penalized_error"])
        for layer in (0, 1, 2, 3, None):
            event = first_wrong(true_codes, source["codes"], layer)
            if event is None:
                continue
            changed = dict(source)
            changed["codes"] = [list(row) for row in source["codes"]]
            if layer is None:
                changed["codes"][event] = list(true_codes[event])
                intervention = "first_wrong_tuple"
            else:
                changed["codes"][event][layer] = true_codes[event][layer]
                intervention = f"first_wrong_layer_{layer + 1}"
            decoded = decode_one(model, changed, context, geometry, scale)
            if decoded["status"] == "reconstructed":
                actual_stats = trajectory_sufficient_statistics(
                    np.asarray(decoded["gps_traj"]), decoded["road_traj"], geometry, "porto",
                    mbr=dataset.metadata["mbr"], grid=dataset.metadata["grid"]
                )
                new_error = score(stage4[tid]["real"], actual_stats)
            else:
                new_error = 1.0
            result["rows"].append({
                "traj_id": tid, "intervention": intervention, "event": event,
                "original_error": baseline, "replaced_error": new_error,
                "improvement": baseline - new_error, "valid": decoded["status"] == "reconstructed",
            })
    result["elapsed_seconds"] = time.monotonic() - started
    result["parity_passed"] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", choices=("HV", "LF", "H1", "B2"), default=("HV", "LF"))
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.sample_size <= 0 or args.sample_size > 1024:
        raise ValueError("invalid output or sample size")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("dirty analysis worktree")
    plan = read(args.config)
    preflight_path = Path(plan["ablation"]["preflight_path"])
    ps = digest(preflight_path / "manifest.json")
    cs = digest(args.config)
    ids = tuple(sorted(map(int, read(preflight_path / "manifest.json")["validation_ids"])))
    if len(ids) != 1024 or len(set(ids)) != 1024 or ps != plan["ablation"]["preflight_sha256"]:
        raise ValueError("preflight mismatch")
    sample = sorted(ids, key=lambda tid: hashlib.sha256(f"diagnosis-20260923:{tid}".encode()).digest())[:args.sample_size]
    reference = None
    manifests = {}
    cells = {}
    for cell in args.cells:
        run_dir = args.root / "pilot/porto/42" / cell / "attempt_001"
        manifest, count = verify_completed(run_dir, reference, cs, ps, cell)
        if reference is None:
            reference = manifest
        manifests[cell] = {"sha256": digest(run_dir / "run_manifest.json"), "revision": manifest["revision"], "artifacts_verified": count}
        backend = Backend(args.config, preflight_path, cell, "pilot", 42, time.monotonic() + 7200)
        cells[cell] = run_cell(backend, run_dir, sample)
    result = {
        "status": "completed_first_wrong_intervention",
        "analysis_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv,
        "config_sha256": cs,
        "preflight_sha256": ps,
        "sample_rule": "SHA256(diagnosis-20260923:traj_id) ascending, common across cells",
        "sample_size": len(sample),
        "population_scope": "deterministic performance pilot" if len(sample) < 1024 else "full validation",
        "interpretation": "first wrong code replacement is an OOD diagnostic, not a causal additive effect",
        "source_manifests": manifests,
        "cells": cells,
    }
    atomic_json(args.output, result)
    print(json.dumps({cell: {"rows": len(value["rows"]), "skipped": len(value["skipped"]), "parity_max": value["parity_max"], "seconds": value["elapsed_seconds"]} for cell, value in cells.items()}))


if __name__ == "__main__":
    main()
