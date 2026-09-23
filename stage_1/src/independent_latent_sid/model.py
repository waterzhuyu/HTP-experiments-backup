"""Independent view encoders and ordered conditional residual SID exploration."""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hierarchical_rqvae.model import HierarchicalRQVAE
from staged_attribution.quantizer import AttributionQuantizerOutput


class IndependentLatentSID(HierarchicalRQVAE):
    def __init__(self, *args, construction="conditional", reconstruction_route="gps_only", reconstruction_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if construction not in {"late_fusion", "conditional"}:
            raise ValueError("unknown SID construction")
        if self.prefix_depths != {"grid": 1, "road": 2, "gps": 4}:
            raise ValueError("independent SID requires Grid/Road/GPS depths 1/2/4")
        self.construction = construction
        if reconstruction_route not in {"gps_only", "all_views"}:
            raise ValueError("unknown reconstruction route")
        if construction != "conditional" and reconstruction_route != "gps_only":
            raise ValueError("route ablation applies only to conditional SID")
        self.reconstruction_route = reconstruction_route
        if not math.isfinite(reconstruction_scale) or reconstruction_scale <= 0:
            raise ValueError("reconstruction_scale must be finite and positive")
        self.reconstruction_scale = float(reconstruction_scale)
        # Equal initial weights, independent parameter storage; no RNG shift between cells.
        self.view_encoders = nn.ModuleDict({v: copy.deepcopy(self.encoder) for v in ("grid", "road", "gps")})
        self.view_projections = nn.ModuleDict({v: copy.deepcopy(self.rq_pre_quant_linear) for v in self.view_encoders})
        del self.encoder
        del self.rq_pre_quant_linear

    def independent_latents(self, batch):
        f = self.fusion
        mask = torch.arange(batch["gps"].shape[1], device=batch["gps"].device)[None] < batch["lengths"][:, None]
        # Sanitize padding before embedding lookup; padded IDs are not observations.
        road = torch.where(mask, batch["point_road"] + 1, 0)
        grid = torch.where(mask, batch["grid"] + 1, 0)
        gps = torch.where(mask[..., None], batch["gps"], 0.)
        features = {"gps": f.norms["gps"](f.gps_adapter(gps)),
                    "road": f.norms["road"](f.road_adapter(f.point_road_embedding(road))),
                    "grid": f.norms["grid"](f.grid_embedding(grid))}
        latents = {}
        reference = None
        for view in self.view_encoders:
            encoded, lengths, types, valid = self.view_encoders[view](features[view] * mask[..., None], batch["lengths"])
            if reference is not None and not torch.equal(valid, reference):
                raise ValueError("view event alignment differs")
            reference = valid
            latents[view] = self.view_projections[view](encoded[valid]).float()
        return latents, encoded, lengths, torch.stack(types).T, valid

    def conditional_quantize(self, latents):
        # q1=Q(z_grid), q2=Q(z_road-s1), q3/q4 refine z_gps.
        # Stop gradients through previous codewords in each local residual objective.
        with torch.autocast(device_type=latents["gps"].device.type, enabled=False):
            aggregate = torch.zeros_like(latents["gps"])
            codes, losses, info = [], [], []
            for view, book in zip(("grid", "road", "gps", "gps"), self.rq_quant.codebooks):
                target = latents[view].float() - aggregate.detach()
                distances = self.rq_quant._distances(target.detach(), book)
                ids = distances.argmin(-1)
                q = book.codebook(ids)
                cb = F.mse_loss(q, target.detach())
                commitment = self.rq_quant.beta * F.mse_loss(target, q.detach())
                loss = cb + commitment
                losses.append(loss)
                with torch.no_grad():
                    hits = ids.bincount(minlength=book.n_embed).float()
                    book.codebook_hit.add_(hits)
                    p = hits / hits.sum()
                    ppl = (-(p * p.clamp_min(1e-9).log()).sum()).exp()
                info.append({"avg_usage": float((book.codebook_hit > 0).float().mean()),
                             "ppl": float(ppl / book.n_embed), "loss": float(loss.detach()),
                             "quant_loss": float(cb.detach()), "commitment_loss": float(commitment.detach())})
                aggregate = aggregate + q
                codes.append(ids)
            z = latents["gps"]
            if self.reconstruction_route == "all_views":
                z = z + (latents["grid"] - latents["grid"].detach()) + (latents["road"] - latents["road"].detach())
            else:
                z = z + (self.reconstruction_scale - 1.0) * (z - z.detach())
            st = z + (aggregate - z).detach() if self.training else aggregate
            return AttributionQuantizerOutput(aggregate, st, torch.stack(codes, -1),
                torch.stack(losses).mean(), z.sum() * 0, (), info)

    def encode_views(self, batch):
        latents, encoded, lengths, types, mask = self.independent_latents(batch)
        z = sum(latents.values()) / math.sqrt(3) if self.construction == "late_fusion" else latents["gps"]
        quant = self.rq_quant(z) if self.construction == "late_fusion" else self.conditional_quantize(latents)
        decoder = torch.zeros_like(encoded)
        decoder[mask] = self.rq_post_quant_linear(quant.straight_through).to(decoder.dtype)
        prequant = z.new_zeros(*mask.shape, z.shape[-1]); prequant[mask] = z
        codes = torch.full((*mask.shape, 4), -1, device=z.device, dtype=torch.long); codes[mask] = quant.codes
        layers = torch.stack([b.codebook(quant.codes[:, i]) for i, b in enumerate(self.rq_quant.codebooks)], 1)
        return {"fused": None, "decoder_latent": decoder, "prequant_latent": prequant,
                "codes": codes, "latent_lengths": lengths, "latent_mask": mask, "lens_type": types,
                "quant_loss": quant.quant_loss, "usage_loss": quant.usage_loss, "rq_info": quant.layer_info,
                "layer_vectors": layers, "prefix_vectors": layers.cumsum(1), "view_latents": latents}

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
            if self.construction == "conditional":
                z = output["view_latents"][name]
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
