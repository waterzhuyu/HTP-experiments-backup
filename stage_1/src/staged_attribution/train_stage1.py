from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import torch

from .losses import derive_auxiliary_targets
from .model import AttributionRQVAE
from .views import make_sampling_view


def _to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


def train_step(
    model: AttributionRQVAE,
    batch: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sampling_seed: int,
    speed_edges: torch.Tensor,
    turn_edges: torch.Tensor,
    meters_per_coordinate: tuple[float, float],
    gradient_clip_norm: float = 1.0,
    keep_probability: float = 0.65,
) -> dict[str, object]:
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    device_batch = _to_device(batch, device)
    model.train()
    consistency_enabled = any(
        model.q_weights[name] > 0 for name in ("invariance", "variance", "covariance")
    )
    auxiliary_enabled = model.q_weights["auxiliary"] > 0
    sampling_batch = (
        make_sampling_view(
            device_batch,
            seed=sampling_seed,
            keep_probability=keep_probability,
        )
        if consistency_enabled
        else None
    )
    auxiliary_targets = (
        derive_auxiliary_targets(
            device_batch,
            speed_edges=speed_edges,
            turn_edges=turn_edges,
            meters_per_coordinate=meters_per_coordinate,
        )
        if auxiliary_enabled
        else None
    )

    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        output = model(
            device_batch,
            sampling_batch=sampling_batch,
            auxiliary_targets=auxiliary_targets,
        )
    loss = output["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite Stage1 loss: {float(loss.detach())}")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("non-finite Stage1 gradient norm")
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "base_loss": float(output["base_loss"].detach()),
        "reconstruction_loss": float(output["reconstruction_loss"].detach()),
        "percent_loss": float(output["percent_loss"].detach()),
        "dxy_loss": float(output["dxy_loss"].detach()),
        "quant_loss": float(output["quant_loss"].detach()),
        "q_loss_components": {
            name: float(value.detach())
            for name, value in output["q_loss_components"].items()
        },
        "gradient_norm": float(gradient_norm.detach()),
        "sampling_enabled": consistency_enabled,
        "auxiliary_enabled": auxiliary_enabled,
    }


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int = 10,
    warmup_ratio: float = 0.01,
    minimum_ratio: float = 0.10,
) -> torch.optim.lr_scheduler.LambdaLR:
    if steps_per_epoch <= 0 or epochs <= 0:
        raise ValueError("steps_per_epoch and epochs must be positive")
    if warmup_epochs < 0 or not 0 < warmup_ratio <= 1 or not 0 < minimum_ratio <= 1:
        raise ValueError("invalid Stage1 scheduler ratios")
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps)

    def multiplier(step: int) -> float:
        if warmup_steps and step <= warmup_steps:
            return warmup_ratio + (1.0 - warmup_ratio) * step / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def train_epochs(
    model: AttributionRQVAE,
    loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    seed: int,
    speed_edges: torch.Tensor,
    turn_edges: torch.Tensor,
    meters_per_coordinate: tuple[float, float],
    gradient_clip_norm: float = 1.0,
    keep_probability: float = 0.65,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> list[dict[str, object]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    model = model.to(device)
    history: list[dict[str, object]] = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        accumulators: dict[str, list[float]] = defaultdict(list)
        component_accumulators: dict[str, list[float]] = defaultdict(list)
        steps = 0
        for batch in loader:
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                sampling_seed=seed + global_step,
                speed_edges=speed_edges,
                turn_edges=turn_edges,
                meters_per_coordinate=meters_per_coordinate,
                gradient_clip_norm=gradient_clip_norm,
                keep_probability=keep_probability,
            )
            global_step += 1
            steps += 1
            if scheduler is not None:
                scheduler.step()
            for name in (
                "loss",
                "base_loss",
                "reconstruction_loss",
                "percent_loss",
                "dxy_loss",
                "quant_loss",
                "gradient_norm",
            ):
                accumulators[name].append(float(metrics[name]))
            for name, value in metrics["q_loss_components"].items():
                component_accumulators[name].append(float(value))
        if steps == 0:
            raise ValueError("training loader produced zero batches")
        row: dict[str, object] = {
            "epoch": epoch,
            "steps": steps,
            "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{
                name: sum(values) / len(values)
                for name, values in accumulators.items()
            },
            "q_loss_components": {
                name: sum(values) / len(values)
                for name, values in component_accumulators.items()
            },
        }
        if not all(
            math.isfinite(float(value))
            for name, value in row.items()
            if name not in {"epoch", "steps", "q_loss_components"}
        ):
            raise FloatingPointError(f"non-finite epoch metrics: {row}")
        history.append(row)
    return history


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_stage1_artifacts(
    output_dir: str | Path,
    model: AttributionRQVAE,
    resolved_config: Mapping[str, object],
    history: list[dict[str, object]],
    metrics: Mapping[str, object],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_config.json"
    checkpoint_path = output / "stage1_model.pt"
    metrics_path = output / "stage1_metrics.json"
    _atomic_json(resolved_path, dict(resolved_config))
    _atomic_json(metrics_path, dict(metrics))
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "resolved_config": dict(resolved_config),
            "history": history,
        },
        temporary_checkpoint,
    )
    temporary_checkpoint.replace(checkpoint_path)
    return {
        "resolved_config": resolved_path,
        "checkpoint": checkpoint_path,
        "stage1_metrics": metrics_path,
    }
