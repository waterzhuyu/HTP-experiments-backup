from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset


VIEWS = ("gps", "grid", "road")
ENCODER_FAMILIES = ("transformer", "gru")


def sequence_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    max_length = int(lengths.max()) if max_length is None else max_length
    return torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]


class MultiViewTrajectoryDataset(Dataset):
    """Point-aligned GPS/Grid/Road trajectories stored as flat arrays plus offsets."""

    def __init__(self, data_dir: Path, split: str, indices: np.ndarray | None = None):
        self.gps = np.load(data_dir / f"{split}_gps.npy", mmap_mode="r")
        self.grid = np.load(data_dir / f"{split}_grid.npy", mmap_mode="r")
        self.road = np.load(data_dir / f"{split}_road.npy", mmap_mode="r")
        self.offsets = np.load(data_dir / f"{split}_offsets.npy", mmap_mode="r")
        self.traj_ids = np.load(data_dir / f"{split}_traj_ids.npy", mmap_mode="r")
        self.indices = (
            np.arange(len(self.traj_ids), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        source_index = int(self.indices[index])
        start = int(self.offsets[source_index])
        end = int(self.offsets[source_index + 1])
        return {
            "traj_id": int(self.traj_ids[source_index]),
            "gps": np.asarray(self.gps[start:end]).copy(),
            "grid": np.asarray(self.grid[start:end]).copy(),
            "road": np.asarray(self.road[start:end]).copy(),
        }


def collate_multiview(batch: list[dict[str, np.ndarray | int]]) -> dict[str, torch.Tensor]:
    gps = [torch.as_tensor(item["gps"], dtype=torch.float32) for item in batch]
    grid = [torch.as_tensor(item["grid"], dtype=torch.long) for item in batch]
    road = [torch.as_tensor(item["road"], dtype=torch.long) for item in batch]
    lengths = torch.as_tensor([len(value) for value in gps], dtype=torch.long)
    return {
        "traj_id": torch.as_tensor([int(item["traj_id"]) for item in batch], dtype=torch.long),
        "gps": pad_sequence(gps, batch_first=True, padding_value=0.0),
        "grid": pad_sequence(grid, batch_first=True, padding_value=0),
        "road": pad_sequence(road, batch_first=True, padding_value=0),
        "lengths": lengths,
    }


class TemporalDownsampler(nn.Module):
    """Three contextualize-then-stride blocks; output length is ceil(T/8)."""

    def __init__(self, family: str, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        if family not in ENCODER_FAMILIES:
            raise ValueError(f"unknown encoder family: {family}")
        if family == "gru" and hidden_dim % 2:
            raise ValueError("GRU hidden_dim must be even for the bidirectional control")
        self.family = family
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        if family == "transformer":
            self.context_blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=4,
                        dim_feedforward=hidden_dim * 4,
                        dropout=dropout,
                        batch_first=True,
                        activation="gelu",
                        norm_first=True,
                    )
                    for _ in range(3)
                ]
            )
        else:
            self.context_blocks = nn.ModuleList(
                [
                    nn.GRU(
                        hidden_dim,
                        hidden_dim // 2,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=True,
                    )
                    for _ in range(3)
                ]
            )
        self.downsamplers = nn.ModuleList(
            [nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1) for _ in range(3)]
        )
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = values
        current_lengths = lengths
        for norm, context, downsample in zip(self.norms, self.context_blocks, self.downsamplers):
            mask = sequence_mask(current_lengths, hidden.shape[1])
            hidden = norm(hidden) * mask.unsqueeze(-1)
            if self.family == "transformer":
                hidden = context(hidden, src_key_padding_mask=~mask)
            else:
                packed = pack_padded_sequence(
                    hidden,
                    current_lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_hidden, _ = context(packed)
                hidden, _ = pad_packed_sequence(
                    packed_hidden,
                    batch_first=True,
                    total_length=hidden.shape[1],
                )
            hidden = downsample(hidden.transpose(1, 2)).transpose(1, 2)
            current_lengths = torch.div(current_lengths + 1, 2, rounding_mode="floor")
        mask = sequence_mask(current_lengths, hidden.shape[1])
        latent = self.output(hidden) * mask.unsqueeze(-1)
        return latent, current_lengths, mask


class SphericalVectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, beta: float):
        super().__init__()
        self.codebook_size = codebook_size
        self.beta = beta
        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(dim))

    def distances(self, values: torch.Tensor) -> torch.Tensor:
        values = F.normalize(values.float(), dim=-1)
        weight = F.normalize(self.embedding.weight.float(), dim=-1)
        return 2.0 - 2.0 * values @ weight.T

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = F.normalize(values.float(), dim=-1)
        distances = self.distances(normalized)
        codes = distances.argmin(dim=-1)
        quantized = F.normalize(self.embedding.weight.float(), dim=-1)[codes]
        codebook_loss = F.mse_loss(quantized, normalized.detach())
        commitment_loss = self.beta * F.mse_loss(normalized, quantized.detach())
        straight_through = normalized + (quantized - normalized).detach()
        return straight_through, codes, codebook_loss + commitment_loss, distances


@dataclass
class TokenizerOutput:
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    codes: torch.Tensor
    latents: torch.Tensor
    mask: torch.Tensor


class IndependentTokenizer(nn.Module):
    """One view-specific encoder, decoder and codebook with no cross-view parameters."""

    def __init__(
        self,
        view: str,
        encoder_family: str,
        num_grids: int,
        road_features: torch.Tensor,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        codebook_size: int = 256,
        beta: float = 0.25,
        dropout: float = 0.1,
        usage_weight: float = 0.05,
        assignment_temperature: float = 0.2,
    ):
        super().__init__()
        if view not in VIEWS:
            raise ValueError(f"unknown view: {view}")
        self.view = view
        self.num_grids = num_grids
        self.usage_weight = usage_weight
        self.assignment_temperature = assignment_temperature
        if view == "gps":
            self.input = nn.Linear(2, hidden_dim)
            output_dim = 2
        elif view == "grid":
            self.input = nn.Embedding(num_grids, hidden_dim)
            output_dim = 2
        else:
            self.road_embedding = nn.Embedding.from_pretrained(road_features, freeze=True)
            self.input = nn.Linear(road_features.shape[1], hidden_dim)
            output_dim = road_features.shape[1]
        self.encoder = TemporalDownsampler(encoder_family, hidden_dim, latent_dim, dropout)
        self.quantizer = SphericalVectorQuantizer(codebook_size, latent_dim, beta)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def grid_centers(ids: torch.Tensor, n_grid_x: int, n_grid_y: int) -> torch.Tensor:
        x = (ids.remainder(n_grid_x).float() + 0.5) / n_grid_x
        y = (torch.div(ids, n_grid_x, rounding_mode="floor").float() + 0.5) / n_grid_y
        return torch.stack((x, y), dim=-1)

    @staticmethod
    def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (prediction.sub(target).square().mean(dim=-1) * mask).sum() / mask.sum().clamp_min(1)

    def _input_and_target(
        self,
        batch: dict[str, torch.Tensor],
        n_grid_x: int,
        n_grid_y: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.view == "gps":
            return self.input(batch["gps"]), batch["gps"]
        if self.view == "grid":
            return self.input(batch["grid"]), self.grid_centers(batch["grid"], n_grid_x, n_grid_y)
        road_features = self.road_embedding(batch["road"])
        return self.input(road_features), F.normalize(road_features.float(), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor], n_grid_x: int, n_grid_y: int) -> TokenizerOutput:
        values, target = self._input_and_target(batch, n_grid_x, n_grid_y)
        latent, _, mask = self.encoder(values, batch["lengths"])
        valid_latent = latent[mask]
        quantized, valid_codes, vq_loss, distances = self.quantizer(valid_latent)
        restored = torch.zeros(latent.shape, dtype=quantized.dtype, device=latent.device)
        restored[mask] = quantized
        codes = torch.full(latent.shape[:2], -1, dtype=torch.long, device=latent.device)
        codes[mask] = valid_codes
        expanded = restored.repeat_interleave(8, dim=1)[:, : target.shape[1]]
        prediction = self.decoder(expanded)
        point_mask = sequence_mask(batch["lengths"], target.shape[1])
        reconstruction = self.masked_mse(prediction, target, point_mask)
        marginal = F.softmax(-distances / self.assignment_temperature, dim=-1).mean(dim=0)
        entropy = -(marginal * marginal.clamp_min(1e-9).log()).sum()
        usage = 1.0 - entropy / math.log(self.quantizer.codebook_size)
        loss = reconstruction + vq_loss + self.usage_weight * usage
        return TokenizerOutput(
            loss=loss,
            losses={
                "reconstruction": reconstruction,
                "vq": vq_loss,
                "usage_regularization": usage,
                "total": loss,
            },
            codes=codes,
            latents=F.normalize(latent.float(), dim=-1),
            mask=mask,
        )


def codebook_statistics(codes: np.ndarray, codebook_size: int) -> dict[str, float | int]:
    counts = np.bincount(codes, minlength=codebook_size).astype(np.float64)
    probability = counts[counts > 0] / max(counts.sum(), 1.0)
    return {
        "active_codes": int(np.count_nonzero(counts)),
        "active_fraction": float(np.count_nonzero(counts) / codebook_size),
        "perplexity": float(np.exp(-(probability * np.log(probability)).sum())),
        "max_token_share": float(counts.max() / max(counts.sum(), 1.0)),
    }
