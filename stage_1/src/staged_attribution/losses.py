from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F


LOSS_NAMES = ("usage", "invariance", "variance", "covariance", "auxiliary")


def _valid(latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if latent.shape[:2] != mask.shape:
        raise ValueError("latent and mask axes do not align")
    values = latent[mask]
    if not len(values):
        raise ValueError("loss requires at least one valid latent event")
    return values


def invariance_loss(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("invariance latent tensors must have identical shape")
    first_valid = F.normalize(_valid(first, mask), dim=-1)
    second_valid = F.normalize(_valid(second, mask), dim=-1)
    return (1.0 - (first_valid * second_valid).sum(dim=-1)).mean()


def variance_loss(latent: torch.Tensor, mask: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    values = _valid(latent, mask)
    std = torch.sqrt(values.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(target_std - std).mean()


def covariance_loss(latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = _valid(latent, mask)
    if len(values) < 2:
        return values.sum() * 0.0
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (len(values) - 1)
    diagonal = torch.diag(torch.diag(covariance))
    return (covariance - diagonal).square().sum() / covariance.shape[0]


def _mean_speed(
    gps: torch.Tensor,
    time: torch.Tensor,
    meters_per_coordinate: tuple[float, float],
) -> torch.Tensor:
    if len(gps) < 2:
        return gps.new_tensor(0.0)
    scale = gps.new_tensor(meters_per_coordinate)
    distance = ((gps[1:] - gps[:-1]) * scale).square().sum(dim=-1).sqrt()
    delta_time = time[1:] - time[:-1]
    valid = delta_time > 1e-6
    if not torch.any(valid):
        return gps.new_tensor(0.0)
    return (distance[valid] / delta_time[valid]).mean()


def _mean_turn(gps: torch.Tensor) -> torch.Tensor:
    if len(gps) < 3:
        return gps.new_tensor(0.0)
    vectors = gps[1:] - gps[:-1]
    first = vectors[:-1]
    second = vectors[1:]
    denominator = first.norm(dim=-1) * second.norm(dim=-1)
    valid = denominator > 1e-9
    if not torch.any(valid):
        return gps.new_tensor(0.0)
    cosine = (first[valid] * second[valid]).sum(dim=-1) / denominator[valid]
    return torch.acos(cosine.clamp(-1.0, 1.0)).abs().mean()


def derive_auxiliary_targets(
    batch: Mapping[str, torch.Tensor],
    speed_edges: torch.Tensor,
    turn_edges: torch.Tensor,
    meters_per_coordinate: tuple[float, float],
    event_size: int = 8,
) -> dict[str, torch.Tensor]:
    if event_size <= 0:
        raise ValueError("event_size must be positive")
    lengths = batch["lengths"]
    latent_lengths = torch.div(lengths + event_size - 1, event_size, rounding_mode="floor")
    max_events = int(latent_lengths.max())
    shape = (len(lengths), max_events)
    device = batch["gps"].device
    targets = {
        "speed_state": torch.full(shape, -100, dtype=torch.long, device=device),
        "turn_state": torch.full(shape, -100, dtype=torch.long, device=device),
        "road_transition": torch.full(shape, -100, dtype=torch.long, device=device),
        "grid_transition": torch.full(shape, -100, dtype=torch.long, device=device),
    }
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    speed_edges = speed_edges.to(device=device, dtype=batch["gps"].dtype)
    turn_edges = turn_edges.to(device=device, dtype=batch["gps"].dtype)

    for row, length_value in enumerate(lengths.tolist()):
        length = int(length_value)
        for event in range(math.ceil(length / event_size)):
            start = event * event_size
            end = min(start + event_size, length)
            gps = batch["gps"][row, start:end]
            time = batch["time"][row, start:end]
            grid = batch["grid"][row, start:end]
            road = batch["point_road"][row, start:end]
            speed = _mean_speed(gps, time, meters_per_coordinate)
            turn = _mean_turn(gps)
            targets["speed_state"][row, event] = torch.bucketize(speed, speed_edges)
            targets["turn_state"][row, event] = torch.bucketize(turn, turn_edges)
            targets["road_transition"][row, event] = int(
                len(road) > 1 and bool(torch.any(road[1:] != road[:-1]))
            )
            targets["grid_transition"][row, event] = int(
                len(grid) > 1 and bool(torch.any(grid[1:] != grid[:-1]))
            )
            mask[row, event] = True
    targets["mask"] = mask
    return targets


def auxiliary_classification_loss(
    latent: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    heads: Mapping[str, torch.nn.Module],
) -> torch.Tensor:
    mask = targets["mask"]
    values = _valid(latent, mask)
    losses = []
    for name in ("speed_state", "turn_state", "road_transition", "grid_transition"):
        logits = heads[name](values)
        losses.append(F.cross_entropy(logits, targets[name][mask]))
    return torch.stack(losses).mean()


def combine_q_losses(
    base_loss: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if set(components) != set(LOSS_NAMES):
        raise ValueError(f"components must be exactly {LOSS_NAMES}")
    if set(weights) != set(LOSS_NAMES):
        raise ValueError(f"weights must be exactly {LOSS_NAMES}")
    total = base_loss
    reported: dict[str, torch.Tensor] = {}
    for name in LOSS_NAMES:
        weight = float(weights[name])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid non-negative weight for {name}")
        value = components[name]
        if value.ndim != 0 or not torch.isfinite(value):
            raise ValueError(f"loss component {name} must be a finite scalar")
        reported[name] = value
        total = total + weight * value
    return total, reported
