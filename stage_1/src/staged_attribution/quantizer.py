from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rq_quant import VQEmbedding


@dataclass
class AttributionQuantizerOutput:
    quantized: torch.Tensor
    straight_through: torch.Tensor
    codes: torch.Tensor
    quant_loss: torch.Tensor
    usage_loss: torch.Tensor
    soft_marginals: tuple[torch.Tensor, ...]
    layer_info: list[dict[str, float]]


class AttributionRQBottleneck(nn.Module):
    """State-compatible HTP residual VQ with differentiable usage diagnostics."""

    def __init__(
        self,
        base_n_embed: int,
        codebook_dim: int,
        rq_layer: int,
        rq_multi: list[int],
        beta: float = 0.25,
        assignment_temperature: float = 0.2,
    ) -> None:
        super().__init__()
        if rq_layer != len(rq_multi):
            raise ValueError("rq_layer must equal len(rq_multi)")
        if assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be positive")
        self.base_n_embed = base_n_embed
        self.rq_multi = list(rq_multi)
        self.rq_layer = rq_layer
        self.n_embed = [base_n_embed // value for value in rq_multi]
        self.codebooks = nn.ModuleList(
            [VQEmbedding(size, codebook_dim) for size in self.n_embed]
        )
        self.beta = beta
        self.assignment_temperature = assignment_temperature

    @staticmethod
    def _distances(values: torch.Tensor, codebook: VQEmbedding) -> torch.Tensor:
        return (
            values.square().sum(dim=-1, keepdim=True)
            + codebook.codebook.weight.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * values @ codebook.codebook.weight.T
        )

    def reset_codebook_hit(self) -> None:
        for codebook in self.codebooks:
            codebook.codebook_hit.zero_()

    def forward(self, values: torch.Tensor) -> AttributionQuantizerOutput:
        if values.ndim != 2 or not len(values):
            raise ValueError("quantizer input must be a non-empty [events, dimension] tensor")
        residual = values.detach().clone()
        quantized_layers: list[torch.Tensor] = []
        codes: list[torch.Tensor] = []
        cumulative: list[torch.Tensor] = []
        marginals: list[torch.Tensor] = []
        info: list[dict[str, float]] = []
        aggregate = torch.zeros_like(values)

        for codebook in self.codebooks:
            distances = self._distances(residual, codebook)
            code = distances.argmin(dim=-1)
            quantized = codebook.codebook(code)
            with torch.no_grad():
                hits = code.bincount(minlength=codebook.n_embed).float()
                codebook.codebook_hit.add_(hits)
                probabilities = hits / hits.sum().clamp_min(1.0)
                perplexity = torch.exp(
                    -(probabilities * probabilities.clamp_min(1e-9).log()).sum()
                )
                layer_info = {
                    "avg_usage": float((codebook.codebook_hit > 0).float().mean()),
                    "ppl": float(perplexity / codebook.n_embed),
                }
            soft = F.softmax(-distances / self.assignment_temperature, dim=-1)
            marginal = soft.mean(dim=0)
            marginals.append(marginal)
            residual = residual - quantized
            aggregate = aggregate + quantized
            cumulative.append(aggregate)
            quantized_layers.append(quantized)
            codes.append(code)
            info.append(layer_info)

        partial_losses: list[torch.Tensor] = []
        for index, partial in enumerate(cumulative):
            codebook_loss = F.mse_loss(values.detach(), partial)
            commitment_loss = self.beta * F.mse_loss(values, partial.detach())
            layer_loss = codebook_loss + commitment_loss
            partial_losses.append(layer_loss)
            info[index].update(
                {
                    "loss": float(layer_loss.detach()),
                    "quant_loss": float(codebook_loss.detach()),
                    "commitment_loss": float(commitment_loss.detach()),
                }
            )
        quant_loss = torch.stack(partial_losses).mean()
        usage_losses = []
        for marginal, size in zip(marginals, self.n_embed):
            entropy = -(marginal * marginal.clamp_min(1e-9).log()).sum()
            usage_losses.append(1.0 - entropy / math.log(size))
        usage_loss = torch.stack(usage_losses).mean()
        actual_quantized = torch.stack(quantized_layers, dim=-1).sum(dim=-1)
        straight_through = (
            values + (actual_quantized - values).detach()
            if self.training
            else actual_quantized
        )
        return AttributionQuantizerOutput(
            quantized=actual_quantized,
            straight_through=straight_through,
            codes=torch.stack(codes, dim=-1),
            quant_loss=quant_loss,
            usage_loss=usage_loss,
            soft_marginals=tuple(marginals),
            layer_info=info,
        )

    @torch.no_grad()
    def encode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 2 or codes.shape[0] != self.rq_layer:
            raise ValueError(f"codes must have shape [{self.rq_layer}, events]")
        return torch.stack(
            [codebook.encode(codes[index]) for index, codebook in enumerate(self.codebooks)],
            dim=-1,
        ).sum(dim=-1)
