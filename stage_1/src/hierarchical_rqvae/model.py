from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pad_sequence

from staged_attribution.model import AttributionRQVAE
from staged_attribution.quantizer import AttributionRQBottleneck


class Float32RQ(AttributionRQBottleneck):
    def forward(self, values):
        with torch.autocast(device_type=values.device.type, enabled=False):
            return super().forward(values.float())


class HierarchicalRQVAE(AttributionRQVAE):
    """Standard RQ with specified prefix reconstruction heads; no latent bypass."""

    def __init__(self, config, road_features, num_grids, active_views,
                 prefix_depths: Mapping[str, int], aux_weights: Mapping[str, float],
                 observed_vocab: Mapping[str, int], head_seed=42, chunk_positions=256):
        super().__init__(config, road_features, num_grids, frozenset(active_views))
        if set(prefix_depths) != {"gps", "grid", "road"} or set(aux_weights) != set(prefix_depths):
            raise ValueError("exactly gps/grid/road routing and weights required")
        if any(type(v) is not int or not 1 <= v <= 4 for v in prefix_depths.values()):
            raise ValueError("prefix depth must be in 1..4")
        if any(not math.isfinite(v) or v < 0 for v in aux_weights.values()):
            raise ValueError("invalid auxiliary weight")
        if chunk_positions < 1:
            raise ValueError("chunk_positions must be positive")
        self.prefix_depths = dict(prefix_depths)
        self.view_weights = dict(aux_weights)
        self.chunk_positions = chunk_positions
        self.ce_scales = {k: math.log(max(int(observed_vocab[k]), 2)) for k in ("grid", "road")}
        # Preserve inherited state names and initialization, including codebook RNG.
        self.rq_quant.__class__ = Float32RQ
        dimension = int(config["RQ_quant"]["d_model"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.view_heads = nn.ModuleDict({
                name: nn.Sequential(nn.Linear(dimension + 8, 64), nn.SiLU(), nn.Linear(64, size))
                for name, size in (("grid", num_grids), ("road", len(road_features)), ("gps", 2))
            })

    def encode_views(self, batch):
        output = super().encode_views(batch)
        mask = output["latent_mask"]
        codes = output["codes"][mask]
        layers = torch.stack([
            book.codebook(codes[:, layer])
            for layer, book in enumerate(self.rq_quant.codebooks)
        ], dim=1)
        output["layer_vectors"] = layers
        output["prefix_vectors"] = layers.cumsum(dim=1)
        return output

    def view_losses(self, output, batch):
        # Map valid observations to valid packed events; last partial block is masked.
        lengths = batch["lengths"]
        b, t = torch.where(torch.arange(batch["gps"].shape[1], device=lengths.device)[None] < lengths[:, None])
        event_offsets = output["latent_lengths"].cumsum(0) - output["latent_lengths"]
        event = event_offsets[b] + t // 8
        offsets = F.one_hot(t % 8, 8).float()
        weights = 1.0 / (lengths[b].float() * len(lengths))
        z = output["prequant_latent"][output["latent_mask"]].float()
        losses = {}
        for name, depth in self.prefix_depths.items():
            actual = output["prefix_vectors"][:, depth - 1]
            prefix = z + (actual - z).detach() if self.training else actual
            target = batch[{"gps": "gps", "grid": "grid", "road": "point_road"}[name]][b, t]
            total = z.sum() * 0.0
            head = self.view_heads[name]
            scale = self.ce_scales.get(name, 1.0)
            def chunk_loss(x, y, w, head=head, name=name, scale=scale):
                with torch.autocast(device_type=x.device.type, enabled=False):
                    prediction = head(x.float())
                    point_loss = ((prediction - y.float()).square().mean(-1) if name == "gps"
                                  else F.cross_entropy(prediction, y.long(), reduction="none") / scale)
                    return (point_loss * w).sum()
            for start in range(0, len(b), self.chunk_positions):
                sl = slice(start, start + self.chunk_positions)
                inputs = torch.cat([prefix[event[sl]], offsets[sl]], dim=-1)
                if self.training and torch.is_grad_enabled():
                    # Recompute head logits on backward; merely splitting CE would retain all logits.
                    value = checkpoint(chunk_loss, inputs, target[sl], weights[sl], use_reentrant=False)
                else:
                    value = chunk_loss(inputs, target[sl], weights[sl])
                total = total + value
            losses[name] = total
        return losses

    def forward(self, batch, sampling_batch=None, auxiliary_targets=None):
        if sampling_batch is not None or auxiliary_targets is not None:
            raise ValueError("v1 does not enable sampling or inherited auxiliary losses")
        output = super().forward(batch)
        if any(self.view_weights.values()):
            losses = self.view_losses(output, batch)
            output["loss"] = output["loss"] + sum(self.view_weights[k] * v for k, v in losses.items())
        else:
            losses = {k: output["loss"] * 0 for k in self.view_weights}
        output["view_losses"] = losses
        return output

    @torch.no_grad()
    def decode_quantized(self, values, code_lengths, lens_type, batch):
        """Diagnostic decoder for real cumulative prefixes; omitted residuals are zero."""
        if values.ndim != 2 or len(values) != int(code_lengths.sum()):
            raise ValueError('quantized values and code_lengths disagree')
        projected = self.rq_post_quant_linear(values)
        sequences = torch.split(projected, code_lengths.tolist())
        decoder_latent = pad_sequence(sequences, batch_first=True, padding_value=0.)
        route_embedding, route_mask = self._route_context(batch)
        decoded, mask = self.decoder(decoder_latent,code_lengths,lens_type.bool().T,route_embedding,route_mask)
        percent = torch.zeros(mask.shape,device=values.device)
        dxy = torch.zeros(*mask.shape,2,device=values.device)
        percent[mask] = self.percent_head(decoded[mask]).squeeze(-1)
        dxy[mask] = self.dxy_head(decoded[mask])
        return percent,dxy,mask.sum(-1)
