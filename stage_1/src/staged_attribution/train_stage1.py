from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sized
from datetime import datetime, timezone
from pathlib import Path

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Stage1TrainingProgressRecorder:
    """Persist Stage1 optimizer progress without participating in optimization."""

    def __init__(
        self,
        path: str | Path,
        epochs_total: int,
        steps_per_epoch: int,
        progress_interval: int = 100,
    ) -> None:
        if epochs_total <= 0 or steps_per_epoch <= 0:
            raise ValueError("Stage1 progress totals must be positive")
        if progress_interval <= 0:
            raise ValueError("Stage1 progress_interval must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.epochs_total = int(epochs_total)
        self.steps_per_epoch = int(steps_per_epoch)
        self.steps_total = self.epochs_total * self.steps_per_epoch
        self.progress_interval = int(progress_interval)
        self.started_at = _utc_now()
        self.started = time.monotonic()
        self.epoch_current = 0
        self.epochs_completed = 0
        self.step_in_epoch = 0
        self.steps_completed = 0
        self.last_metrics: dict[str, float] = {}
        self._write("running")

    @staticmethod
    def _finite_metrics(metrics: Mapping[str, object] | None) -> dict[str, float]:
        if metrics is None:
            return {}
        names = (
            "loss",
            "base_loss",
            "reconstruction_loss",
            "percent_loss",
            "dxy_loss",
            "quant_loss",
            "gradient_norm",
            "learning_rate",
        )
        result: dict[str, float] = {}
        for name in names:
            value = metrics.get(name)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                result[name] = numeric
        return result

    def _payload(self, status: str, error: Exception | None = None) -> dict[str, object]:
        elapsed = max(time.monotonic() - self.started, 0.0)
        rate = self.steps_completed / elapsed if elapsed > 0 else 0.0
        remaining = max(self.steps_total - self.steps_completed, 0)
        if remaining == 0:
            eta_seconds: float | None = 0.0
        elif rate > 0:
            eta_seconds = remaining / rate
        else:
            eta_seconds = None
        now = _utc_now()
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": status,
            "started_at": self.started_at,
            "updated_at": now,
            "finished_at": now if status in {"completed", "failed"} else None,
            "epochs_total": self.epochs_total,
            "epochs_completed": self.epochs_completed,
            "epoch_current": self.epoch_current,
            "steps_per_epoch": self.steps_per_epoch,
            "step_in_epoch": self.step_in_epoch,
            "steps_total": self.steps_total,
            "steps_completed": self.steps_completed,
            "elapsed_seconds": elapsed,
            "steps_per_second": rate,
            "eta_seconds": eta_seconds,
            "last_metrics": dict(self.last_metrics),
            "error": None,
        }
        if error is not None:
            payload["error"] = {"type": type(error).__name__, "message": str(error)}
        return payload

    def _write(self, status: str, error: Exception | None = None) -> None:
        _atomic_json(self.path, self._payload(status, error))

    def update(
        self,
        epoch: int,
        step_in_epoch: int,
        steps_completed: int,
        metrics: Mapping[str, object] | None = None,
        force: bool = False,
    ) -> bool:
        if steps_completed < self.steps_completed:
            raise ValueError("Stage1 training progress cannot move backwards")
        if not 1 <= epoch <= self.epochs_total:
            raise ValueError("Stage1 progress epoch is out of range")
        if not 1 <= step_in_epoch <= self.steps_per_epoch:
            raise ValueError("Stage1 progress step_in_epoch is out of range")
        expected_step = (epoch - 1) * self.steps_per_epoch + step_in_epoch
        if steps_completed != expected_step:
            raise ValueError("Stage1 global step does not match epoch progress")
        self.epoch_current = int(epoch)
        self.step_in_epoch = int(step_in_epoch)
        self.steps_completed = int(steps_completed)
        self.epochs_completed = epoch if step_in_epoch == self.steps_per_epoch else epoch - 1
        self.last_metrics.update(self._finite_metrics(metrics))
        should_write = (
            force
            or self.steps_completed == self.steps_total
            or self.steps_completed % self.progress_interval == 0
        )
        if should_write:
            self._write("running")
        return should_write

    def complete(self, history: list[dict[str, object]]) -> None:
        if self.steps_completed != self.steps_total or self.epochs_completed != self.epochs_total:
            raise ValueError("Stage1 training returned before the planned step count")
        if history:
            self.last_metrics.update(self._finite_metrics(history[-1]))
        self._write("completed")

    def fail(self, error: Exception) -> None:
        self._write("failed", error)


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
    progress_path: str | Path | None = None,
    progress_interval: int = 100,
) -> list[dict[str, object]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    progress = None
    if progress_path is not None:
        if not isinstance(loader, Sized):
            raise ValueError("Stage1 progress requires a sized training loader")
        progress = _Stage1TrainingProgressRecorder(
            progress_path,
            epochs_total=epochs,
            steps_per_epoch=len(loader),
            progress_interval=progress_interval,
        )
    history: list[dict[str, object]] = []
    global_step = 0
    try:
        model = model.to(device)
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
                if progress is not None:
                    progress.update(
                        epoch,
                        steps,
                        global_step,
                        {**metrics, "learning_rate": optimizer.param_groups[0]["lr"]},
                    )
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
            if progress is not None:
                progress.update(epoch, steps, global_step, row, force=True)
        if progress is not None:
            progress.complete(history)
    except Exception as error:
        if progress is not None:
            progress.fail(error)
        raise
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
