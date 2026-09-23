"""Read-only train/validation residual geometry for independent latent SID."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from hierarchical_rqvae.analyze import digest, read
from hierarchical_rqvae.preflight import atomic_json
from independent_latent_sid.analyze_results import verify_completed
from independent_latent_sid.backend import Backend
from independent_latent_sid.inspect_latents import geometry


def targets(output: dict, construction: str) -> list[torch.Tensor]:
    views = output["view_latents"]
    prefixes = output["prefix_vectors"].float()
    if construction == "conditional":
        return [
            views["grid"].float(),
            views["road"].float() - prefixes[:, 0],
            views["gps"].float() - prefixes[:, 1],
            views["gps"].float() - prefixes[:, 2],
        ]
    if construction == "late_fusion":
        fused = sum(value.float() for value in views.values()) / 3 ** 0.5
        return [fused] + [fused - prefixes[:, layer - 1] for layer in range(1, 4)]
    raise ValueError("unknown construction")


def summarize(target: torch.Tensor, code: torch.Tensor, book: torch.Tensor) -> dict:
    target = target.float()
    code = code.float()
    book = book.float()
    if len(target) != len(code):
        raise ValueError("event mismatch")
    distances = torch.cdist(target, book).square()
    nearest = distances.topk(2, largest=False).values
    selected = (target - code).square().mean(-1)
    if not torch.allclose(selected * target.shape[-1], nearest[:, 0], atol=1e-2, rtol=1e-2):
        raise ValueError("stored code is not nearest to conditional target")
    margin = nearest[:, 1] - nearest[:, 0]
    return {
        "events": len(target),
        "target_mean_norm": float(target.norm(dim=-1).mean()),
        "code_mean_norm": float(code.norm(dim=-1).mean()),
        "residual_mse_per_dimension": float(selected.mean()),
        "nearest_squared_distance_mean": float(nearest[:, 0].mean()),
        "nearest_margin_q10_q50_q90": [float(value) for value in torch.quantile(margin, torch.tensor([0.1, 0.5, 0.9]))],
        "target_effective_rank_covariance_entropy": geometry(target.cpu())["effective_rank"],
    }


@torch.no_grad()
def run_cell(backend, run_dir: Path) -> dict:
    model = backend._load_model(run_dir).eval().to(backend.device)
    result = {}
    for split, expected in (("train", 8192), ("test", 1024)):
        views = {name: [] for name in ("grid", "road", "gps")}
        layers = [[] for _ in range(4)]
        residuals = [[] for _ in range(4)]
        traj_ids = []
        for batch in backend._loader(split):
            batch = {name: value.to(backend.device) for name, value in batch.items()}
            output = model.encode_views(batch)
            traj_ids.extend(int(value) for value in batch["traj_id"].tolist())
            for name in views:
                views[name].append(output["view_latents"][name].float().cpu())
            for index, target in enumerate(targets(output, model.construction)):
                residuals[index].append(target.cpu())
                layers[index].append(output["layer_vectors"][:, index].float().cpu())
        if len(traj_ids) != expected or len(set(traj_ids)) != expected:
            raise ValueError("wrong trajectory population")
        latents = {name: torch.cat(parts) for name, parts in views.items()}
        layer_rows = []
        for index in range(4):
            book = model.rq_quant.codebooks[index].codebook.weight.detach().cpu()
            layer_rows.append(summarize(torch.cat(residuals[index]), torch.cat(layers[index]), book))
        result["validation" if split == "test" else split] = {
            "trajectories": expected,
            "events": len(next(iter(latents.values()))),
            "latent_geometry": {name: geometry(value) for name, value in latents.items()},
            "conditional_residuals": layer_rows,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", choices=("HV", "LF"), default=("HV", "LF"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("dirty analysis worktree")
    torch.set_num_threads(4)
    plan = read(args.config)
    preflight = Path(plan["ablation"]["preflight_path"])
    ps = digest(preflight / "manifest.json")
    cs = digest(args.config)
    if ps != plan["ablation"]["preflight_sha256"]:
        raise ValueError("preflight changed")
    cells, manifests = {}, {}
    reference = None
    for cell in args.cells:
        run_dir = args.root / "pilot/porto/42" / cell / "attempt_001"
        manifest, count = verify_completed(run_dir, reference, cs, ps, cell)
        if reference is None:
            reference = manifest
        backend = Backend(args.config, preflight, cell, "pilot", 42, time.monotonic() + 7200)
        cells[cell] = run_cell(backend, run_dir)
        manifests[cell] = {"sha256": digest(run_dir / "run_manifest.json"), "revision": manifest["revision"], "verified_artifacts": count}
    result = {
        "status": "completed_frozen_geometry",
        "analysis_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv,
        "config_sha256": cs,
        "preflight_sha256": ps,
        "method": "FP32 eval, full train/validation event population, covariance entropy effective rank, no optimizer",
        "source_manifests": manifests,
        "cells": cells,
    }
    atomic_json(args.output, result)
    print(json.dumps({cell: {split: {"trajectories": data["trajectories"], "events": data["events"], "ranks": {name: row["effective_rank"] for name, row in data["latent_geometry"].items()}, "quantization_mse": [row["residual_mse_per_dimension"] for row in data["conditional_residuals"]]} for split, data in splits.items()} for cell, splits in cells.items()}))


if __name__ == "__main__":
    main()
