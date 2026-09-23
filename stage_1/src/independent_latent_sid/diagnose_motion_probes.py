"""Frozen event-level motion probes with a matched position-only control."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from hierarchical_rqvae.analyze import digest, read
from hierarchical_rqvae.preflight import atomic_json
from independent_latent_sid.analyze_results import verify_completed


def labels(cache: dict, scale_km: tuple[float, float]) -> dict:
    mask = cache["mask"].bool()
    count = mask.sum(1)
    last = (count - 1).clamp(min=0)
    rows = torch.arange(len(mask))
    gps = cache["targets"]["gps"].float()
    roads = cache["targets"]["road"].long()
    start = gps[:, 0]
    finish = gps[rows, last]
    displacement = (finish - start) * torch.tensor(scale_km)
    transition = (roads[rows, last] != roads[:, 0]).float()
    valid_motion = count >= 2
    # Aligned road IDs are zero-based; ID 0 is a real road, not a missing sentinel.
    valid_road = valid_motion
    return {
        "displacement": displacement,
        "transition": transition,
        "valid_motion": valid_motion,
        "valid_road": valid_road,
        "position": torch.cat([start, F.one_hot(last, 8).float()], dim=1),
    }


def run_probe(train_x, val_x, train_y, val_y, train_mask, val_mask, task, mean, std, device):
    torch.manual_seed(42)
    head = nn.Sequential(nn.Linear(train_x.shape[1], 64), nn.SiLU(), nn.Linear(64, 2 if task == "motion" else 1)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(42)
    tx, ty, tm = train_x.to(device), train_y.to(device), train_mask.to(device)
    vx, vy, vm = val_x.to(device), val_y.to(device), val_mask.to(device)
    for _ in range(10):
        head.train()
        for indices in torch.randperm(len(tx), generator=generator).split(256):
            idx = indices.to(device)
            chosen = tm[idx]
            if not chosen.any():
                continue
            prediction = head(tx[idx][chosen])
            target = ty[idx][chosen]
            loss = F.mse_loss(prediction, (target - mean) / std) if task == "motion" else F.binary_cross_entropy_with_logits(prediction[:, 0], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    head.eval()
    with torch.no_grad():
        prediction = torch.cat([head(batch) for batch in vx.split(256)])[vm]
    target = vy[vm]
    if task == "motion":
        actual = prediction * std + mean
        error = actual - target
        return {"events": len(target), "mae_east_km": float(error[:, 0].abs().mean()), "mae_north_km": float(error[:, 1].abs().mean()), "mae_vector_km": float(error.norm(dim=1).mean()), "rmse_vector_km": float(error.square().sum(dim=1).mean().sqrt())}
    probs = prediction[:, 0].sigmoid().cpu().numpy()
    truth = target.cpu().numpy()
    return {"events": len(truth), "positive_rate": float(truth.mean()), "accuracy": float(((probs >= 0.5) == truth).mean()), "auroc": float(roc_auc_score(truth, probs)) if len(np.unique(truth)) == 2 else None}


def run_cell(run_dir: Path, scale_km, device):
    cached = torch.load(run_dir / "diagnostics/frozen_features.pt", map_location="cpu", weights_only=True)
    train, val = cached["train"], cached["validation"]
    train_label, val_label = labels(train, scale_km), labels(val, scale_km)
    result = {"population": {split: {"events": len(cached[split]["traj_id"]), "trajectories": len(torch.unique(cached[split]["traj_id"])), "motion_valid": int(lab["valid_motion"].sum()), "road_valid": int(lab["valid_road"].sum())} for split, lab in (("train", train_label), ("validation", val_label))}}
    result["probes"] = []
    for task, target, valid in (("motion", "displacement", "valid_motion"), ("road_transition", "transition", "valid_road")):
        train_y, val_y = train_label[target], val_label[target]
        train_mask, val_mask = train_label[valid], val_label[valid]
        mean = train_y[train_mask].mean(0) if task == "motion" else torch.tensor(0.)
        std = train_y[train_mask].std(0).clamp_min(1e-6) if task == "motion" else torch.tensor(1.)
        mean, std = mean.to(device), std.to(device)
        if task == "motion":
            error = val_y[val_mask] - mean.cpu()
            constant = {"events": int(val_mask.sum()), "mae_east_km": float(error[:, 0].abs().mean()), "mae_north_km": float(error[:, 1].abs().mean()), "mae_vector_km": float(error.norm(dim=1).mean())}
        else:
            truth = val_y[val_mask]
            majority = float(train_y[train_mask].mean() >= 0.5)
            constant = {"events": int(val_mask.sum()), "accuracy": float((truth == majority).float().mean()), "auroc": 0.5, "train_positive_rate": float(train_y[train_mask].mean())}
        result[task] = {"train_mean": mean.cpu().tolist(), "train_std": std.cpu().tolist(), "constant": constant}
        for representation in ("prefixes", "layers"):
            for depth in range(4):
                train_x = torch.cat([train[representation][:, depth].float(), train_label["position"][:, 2:]], dim=1)
                val_x = torch.cat([val[representation][:, depth].float(), val_label["position"][:, 2:]], dim=1)
                score = run_probe(train_x, val_x, train_y, val_y, train_mask, val_mask, task, mean, std, device)
                result["probes"].append({"task": task, "representation": representation, "depth": depth + 1, **score})
        # Position-only MLP has the same width, optimizer, seed, epochs and event order.
        score = run_probe(train_label["position"], val_label["position"], train_y, val_y, train_mask, val_mask, task, mean, std, device)
        result[task]["position_only"] = score
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", choices=("HV", "LF"), default=("HV", "LF"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("output exists or dirty worktree")
    torch.set_num_threads(4)
    plan = read(args.config)
    preflight = Path(plan["ablation"]["preflight_path"])
    ps, cs = digest(preflight / "manifest.json"), digest(args.config)
    if ps != plan["ablation"]["preflight_sha256"]:
        raise ValueError("preflight changed")
    metadata = read(preflight / "aligned" / "metadata.json") if (preflight / "aligned" / "metadata.json").exists() else None
    if metadata is None:
        from independent_latent_sid.backend import Backend
        backend = Backend(args.config, preflight, args.cells[0], "pilot", 42, float("inf"))
        metadata = backend._dataset("test").metadata
    mbr = metadata["mbr"]
    latitude = math.radians((float(mbr["min_lat"]) + float(mbr["max_lat"])) / 2)
    scale = ((float(mbr["max_lon"]) - float(mbr["min_lon"])) * 111.32 * math.cos(latitude), (float(mbr["max_lat"]) - float(mbr["min_lat"])) * 110.54)
    reference, cells, sources = None, {}, {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for cell in args.cells:
        run_dir = args.root / "pilot/porto/42" / cell / "attempt_001"
        manifest, count = verify_completed(run_dir, reference, cs, ps, cell)
        if reference is None:
            reference = manifest
        cells[cell] = run_cell(run_dir, scale, device)
        sources[cell] = {"manifest_sha256": digest(run_dir / "run_manifest.json"), "revision": manifest["revision"], "verified_artifacts": count}
    atomic_json(args.output, {"status": "complete_frozen_motion_probes", "analysis_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "command": sys.argv, "config_sha256": cs, "preflight_sha256": ps, "projection_scale_km_per_normalized_coordinate": scale, "method": "native event first to last valid point; frozen latent, 10 epoch AdamW probe, train-only target normalization; endpoint road ID change; event-level evaluation", "source_manifests": sources, "cells": cells})
    print(json.dumps({cell: {"population": value["population"], "motion_constant": value["motion"]["constant"], "motion_position": value["motion"]["position_only"], "road_constant": value["road_transition"]["constant"], "road_position": value["road_transition"]["position_only"]} for cell, value in cells.items()}))


if __name__ == "__main__":
    main()
