"""Read-only snapshots during an unchanged independent-SID Stage 1 training loop.

Run only from a committed remote HTP worktree. Large output remains on the GPU host.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from hierarchical_rqvae.preflight import atomic_json, sha256
from hierarchical_rqvae.run import verify_preflight
from independent_latent_sid.backend import Backend
from independent_latent_sid.validation import validate_plan
from staged_attribution.token_protocol import codebook_protocol
from staged_attribution.train_stage1 import build_cosine_warmup_scheduler, train_step


AUDIT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 1024, 1280)
SAVE_STEPS = (0, 32, 128, 640, 1280)


def effective_rank(values: torch.Tensor) -> dict[str, float]:
    x = values.float()
    x = x - x.mean(0, keepdim=True)
    covariance = x.T @ x / max(len(x) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if total <= 0:
        return {"effective_rank": 0.0, "centered_rms": 0.0}
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return {
        "effective_rank": float(entropy.exp()),
        "centered_rms": float(x.square().mean().sqrt()),
    }


@torch.no_grad()
def audit(model: torch.nn.Module, batch: dict, device: torch.device) -> dict:
    previous_mode = model.training
    hit_before = [book.codebook_hit.clone() for book in model.rq_quant.codebooks]
    model.eval()
    try:
        moved = {name: value.to(device) for name, value in batch.items()}
        output = model.encode_views(moved)
        mask = output["latent_mask"]
        z = output["prequant_latent"][mask].float()
        latents = output.get("view_latents", {"fused": z})
        layers = output["layer_vectors"].float()
        codes = output["codes"][mask]
        if len(z) != len(codes) or len(z) != len(layers):
            raise ValueError("event population differs across latent, codes, and vectors")
        result = {
            "events": len(z),
            "latent": {name: effective_rank(value) for name, value in latents.items()},
            "layers": [],
        }
        aggregate = torch.zeros_like(z)
        construction = getattr(model, "construction", None)
        for layer_index, (book, q) in enumerate(zip(model.rq_quant.codebooks, layers.unbind(1))):
            if construction == "conditional":
                view = ("grid", "road", "gps", "gps")[layer_index]
                target = latents[view] - aggregate
            else:
                target = z - aggregate
            ids = codes[:, layer_index]
            histogram = torch.bincount(ids, minlength=book.n_embed).float()
            if int(histogram.sum()) != len(z):
                raise ValueError("invalid code population")
            p = histogram / histogram.sum()
            perplexity = (-(p * p.clamp_min(1e-30).log()).sum()).exp()
            target_shape = effective_rank(target)
            codebook_shape = effective_rank(book.codebook.weight.detach())
            quant_mse = (target - q).square().mean()
            target_variance = target.float().var(dim=0, unbiased=False).mean()
            result["layers"].append({
                "k": book.n_embed,
                "active": int((histogram > 0).sum()),
                "perplexity": float(perplexity),
                "max_share": float(p.max()),
                "target_norm": float(target.norm(dim=1).mean()),
                "code_norm": float(q.norm(dim=1).mean()),
                "quant_mse": float(quant_mse),
                "quant_mse_over_target_variance": float(quant_mse / target_variance.clamp_min(1e-12)),
                "target_effective_rank": target_shape["effective_rank"],
                "target_centered_rms": target_shape["centered_rms"],
                "codebook_effective_rank": codebook_shape["effective_rank"],
                "codebook_centered_rms": codebook_shape["centered_rms"],
                "target_codebook_mean_gap": float((target.mean(0) - book.codebook.weight.mean(0)).norm()),
            })
            aggregate = aggregate + q
        result["final_latent_mse"] = float((z - aggregate).square().mean())
        return result
    finally:
        for book, old in zip(model.rq_quant.codebooks, hit_before):
            book.codebook_hit.copy_(old)
        model.train(previous_mode)


def hit_window(model: torch.nn.Module, previous: list[torch.Tensor]) -> tuple[list[dict], list[torch.Tensor]]:
    rows = []
    current = []
    for book, old in zip(model.rq_quant.codebooks, previous):
        now = book.codebook_hit.detach().clone()
        delta = (now - old).float()
        if (delta < 0).any():
            raise ValueError("codebook hit counter decreased")
        total = float(delta.sum())
        rows.append({
            "hits": total,
            "active": int((delta > 0).sum()),
            "max_share": float(delta.max() / total) if total else None,
        })
        current.append(now)
    return rows, current


def snapshot(model, fixed_batches, device, step, last_hits):
    windows, next_hits = hit_window(model, last_hits)
    row = {"step": step, "train_hit_window": windows}
    for name, batch in fixed_batches.items():
        row[name] = audit(model, batch, device)
    return row, next_hits


def run_cell(config, preflight, output_root, cell, seed, revision):
    cell_dir = output_root / cell
    cell_dir.mkdir(parents=True, exist_ok=False)
    backend = Backend(config, preflight, cell, "pilot", seed, time.monotonic() + 24 * 3600)
    loader = backend._loader("train", shuffle=True)
    model = backend._new_model().to(backend.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = build_cosine_warmup_scheduler(
        optimizer, len(loader), backend.budget.stage1_epochs,
        warmup_epochs=10, warmup_ratio=.01, minimum_ratio=.1,
    )
    speed, turn, meters = backend._auxiliary_settings(backend._dataset("train"))
    fixed_batches = {
        "fixed_train": next(iter(backend._loader("train", shuffle=False))),
        "fixed_validation": next(iter(backend._loader("test", shuffle=False))),
    }
    fixed_ids = {name: batch["traj_id"].tolist() for name, batch in fixed_batches.items()}
    if len(loader) * backend.budget.stage1_epochs != 1280:
        raise ValueError("pilot step budget differs from original")
    record = {
        "status": "running", "cell": cell, "seed": seed, "training_revision": revision,
        "config_sha256": sha256(config), "preflight_sha256": sha256(preflight / "manifest.json"),
        "fixed_batch_trajectories": 128, "audit_steps": AUDIT_STEPS,
        "fixed_traj_ids": fixed_ids,
        "method": "original Stage1 train_step/optimizer/scheduler; FP32 eval audits with hit counter restoration",
        "snapshots": [],
    }
    last_hits = [book.codebook_hit.detach().clone() for book in model.rq_quant.codebooks]
    started = time.monotonic()
    try:
        row, last_hits = snapshot(model, fixed_batches, backend.device, 0, last_hits)
        record["snapshots"].append(row)
        torch.save(model.state_dict(), cell_dir / "model_step_0000.pt")
        atomic_json(cell_dir / "timeline.json", record)
        step = 0
        for _epoch in range(backend.budget.stage1_epochs):
            for batch in loader:
                values = train_step(model, batch, optimizer, backend.device, seed + step, speed, turn, meters)
                if values["sampling_enabled"]:
                    raise ValueError("sampling consistency unexpectedly enabled")
                scheduler.step()
                step += 1
                if step in AUDIT_STEPS:
                    row, last_hits = snapshot(model, fixed_batches, backend.device, step, last_hits)
                    row["loss"] = values["loss"]
                    row["gradient_norm"] = values["gradient_norm"]
                    row["lr"] = optimizer.param_groups[0]["lr"]
                    record["snapshots"].append(row)
                    record["elapsed_seconds"] = time.monotonic() - started
                    atomic_json(cell_dir / "timeline.json", record)
                    print(f"{cell} step {step}/1280 elapsed {record['elapsed_seconds']:.0f}s", flush=True)
                if step in SAVE_STEPS:
                    torch.save(model.state_dict(), cell_dir / f"model_step_{step:04d}.pt")
        if step != 1280:
            raise ValueError("training ended before 1280 steps")
        record["status"] = "completed"
        record["elapsed_seconds"] = time.monotonic() - started
        atomic_json(cell_dir / "timeline.json", record)
    except Exception as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        record["elapsed_seconds"] = time.monotonic() - started
        atomic_json(cell_dir / "timeline.json", record)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", choices=("B2", "H1", "LF", "HV"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plan = json.loads(args.config.read_text())
    validate_plan(plan)
    manifest = verify_preflight(args.preflight)
    if sha256(args.preflight / "manifest.json") != plan["ablation"]["preflight_sha256"]:
        raise ValueError("frozen preflight changed")
    if len(manifest["train_ids"]) < 8192 or len(manifest["validation_ids"]) < 1024:
        raise ValueError("incomplete frozen pilot cohort")
    if subprocess.check_output(("git", "status", "--porcelain"), text=True).strip():
        raise RuntimeError("commit experiment code before running")
    revision = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output_root / "run_identity.json", {
        "revision": revision, "config": str(args.config), "config_sha256": sha256(args.config),
        "preflight": str(args.preflight), "preflight_sha256": sha256(args.preflight / "manifest.json"),
        "cells": args.cells, "seed": args.seed, "gpu": torch.cuda.get_device_name(),
    })
    with codebook_protocol(plan["model"]["vocab_sizes"]):
        for cell in args.cells:
            run_cell(args.config, args.preflight, args.output_root, cell, args.seed, revision)


if __name__ == "__main__":
    main()
