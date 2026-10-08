from __future__ import annotations

from typing import Mapping

import torch


def _nearest_indices(source_time: torch.Tensor, target_time: torch.Tensor) -> torch.Tensor:
    right = torch.searchsorted(source_time.contiguous(), target_time.contiguous())
    right = right.clamp(max=len(source_time) - 1)
    left = (right - 1).clamp_min(0)
    choose_right = (source_time[right] - target_time).abs() < (
        target_time - source_time[left]
    ).abs()
    return torch.where(choose_right, right, left)


def _linear_interpolate(
    source_time: torch.Tensor,
    source_values: torch.Tensor,
    target_time: torch.Tensor,
) -> torch.Tensor:
    right = torch.searchsorted(source_time.contiguous(), target_time.contiguous())
    right = right.clamp(max=len(source_time) - 1)
    left = (right - 1).clamp_min(0)
    left_time = source_time[left]
    right_time = source_time[right]
    denominator = right_time - left_time
    weight = torch.where(
        denominator.abs() > 1e-12,
        (target_time - left_time) / denominator.clamp_min(1e-12),
        torch.zeros_like(target_time),
    )
    return source_values[left] + weight.unsqueeze(-1) * (
        source_values[right] - source_values[left]
    )


def make_sampling_view(
    batch: Mapping[str, torch.Tensor],
    seed: int,
    keep_probability: float = 0.65,
) -> dict[str, torch.Tensor]:
    """Thin observations then restore them to the original event index support."""
    if not 0.0 < keep_probability <= 1.0:
        raise ValueError("keep_probability must be in (0, 1]")
    gps = batch["gps"]
    grid = batch["grid"]
    road = batch["point_road"]
    time = batch["time"]
    lengths = batch["lengths"]
    if gps.shape[:2] != grid.shape or grid.shape != road.shape or grid.shape != time.shape:
        raise ValueError("sampling-view tensors must share batch/time axes")
    result = {name: value for name, value in batch.items()}
    result["gps"] = gps.clone()
    result["grid"] = grid.clone()
    result["point_road"] = road.clone()
    kept_mask = torch.zeros(grid.shape, dtype=torch.bool, device=grid.device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for row, length_value in enumerate(lengths.tolist()):
        length = int(length_value)
        if length <= 0:
            raise ValueError("sampling view cannot process an empty trajectory")
        row_time = time[row, :length]
        if torch.any(row_time[1:] < row_time[:-1]):
            raise ValueError("observation time must be non-decreasing")
        keep_cpu = torch.rand(length, generator=generator) < keep_probability
        keep_cpu[0] = True
        keep_cpu[-1] = True
        keep = keep_cpu.to(grid.device)
        kept_mask[row, :length] = keep
        source_time = row_time[keep]
        target_time = row_time
        result["gps"][row, :length] = _linear_interpolate(
            source_time,
            gps[row, :length][keep],
            target_time,
        )
        nearest = _nearest_indices(source_time, target_time)
        result["grid"][row, :length] = grid[row, :length][keep][nearest]
        result["point_road"][row, :length] = road[row, :length][keep][nearest]
    result["sampling_kept_mask"] = kept_mask
    return result
