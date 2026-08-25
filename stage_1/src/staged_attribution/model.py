from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from model.unet import Decoder, Encoder, get_padding_mask
from staged_attribution.losses import (
    LOSS_NAMES,
    auxiliary_classification_loss,
    combine_q_losses,
    covariance_loss,
    invariance_loss,
    variance_loss,
)
from staged_attribution.quantizer import AttributionRQBottleneck


ALLOWED_VIEWS = frozenset({"gps", "road", "grid"})


class PositionEmbedding(nn.Module):
    def __init__(self, dimension: int, max_length: int = 512) -> None:
        super().__init__()
        values = torch.zeros(max_length, dimension, dtype=torch.float32)
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = (
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / dimension)
        ).exp()
        values[:, 0::2] = torch.sin(position * scale)
        values[:, 1::2] = torch.cos(position * scale[: values[:, 1::2].shape[1]])
        self.register_buffer("values", values.unsqueeze(0), persistent=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2] > self.values.shape[1]:
            raise ValueError("route length exceeds positional embedding support")
        return self.values[:, : inputs.shape[-2]]


class ViewFusion(nn.Module):
    """Equal-parameter GPS/Road/Grid adapters with a frozen active-view mask."""

    VIEW_ORDER = ("gps", "road", "grid")

    def __init__(
        self,
        hidden_dim: int,
        road_features: torch.Tensor,
        num_grids: int,
        active_views: frozenset[str],
    ) -> None:
        super().__init__()
        unsupported = active_views - ALLOWED_VIEWS
        if unsupported or not active_views:
            raise ValueError(f"active_views must be a non-empty GPS/Road/Grid subset: {unsupported}")
        if num_grids <= 0:
            raise ValueError("num_grids must be positive")
        self.active_views = frozenset(active_views)
        self.gps_adapter = nn.Linear(2, hidden_dim)
        point_road_weight = torch.cat(
            [torch.zeros(1, road_features.shape[1]), road_features.detach().clone()], dim=0
        )
        self.point_road_embedding = nn.Embedding.from_pretrained(
            point_road_weight, freeze=False, padding_idx=0
        )
        self.road_adapter = nn.Linear(road_features.shape[1], hidden_dim)
        self.grid_embedding = nn.Embedding(num_grids + 1, hidden_dim, padding_idx=0)
        self.norms = nn.ModuleDict(
            {view: nn.LayerNorm(hidden_dim) for view in self.VIEW_ORDER}
        )
        self.null_vectors = nn.Parameter(torch.zeros(len(self.VIEW_ORDER), hidden_dim))
        nn.init.normal_(self.null_vectors, std=1.0 / math.sqrt(hidden_dim))

    @staticmethod
    def _point_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        return torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        gps = batch["gps"]
        grid = batch["grid"]
        road = batch["point_road"]
        lengths = batch["lengths"]
        if gps.ndim != 3 or gps.shape[-1] != 2:
            raise ValueError("gps must have shape [batch, time, 2]")
        if grid.shape != gps.shape[:2] or road.shape != gps.shape[:2]:
            raise ValueError("GPS/Grid/Road point tensors must be aligned")
        mask = self._point_mask(lengths, gps.shape[1])
        if torch.any(grid[mask] < 0) or torch.any(grid[mask] >= self.grid_embedding.num_embeddings - 1):
            raise ValueError("valid Grid IDs outside embedding range")
        if torch.any(road[mask] < 0) or torch.any(
            road[mask] >= self.point_road_embedding.num_embeddings - 1
        ):
            raise ValueError("valid Road IDs outside embedding range")

        representations = {
            "gps": self.norms["gps"](self.gps_adapter(gps)),
            "road": self.norms["road"](
                self.road_adapter(self.point_road_embedding((road + 1).clamp_min(0)))
            ),
            "grid": self.norms["grid"](self.grid_embedding((grid + 1).clamp_min(0))),
        }
        selected: list[torch.Tensor] = []
        for index, view in enumerate(self.VIEW_ORDER):
            if view in self.active_views:
                selected.append(representations[view])
            else:
                selected.append(
                    self.null_vectors[index].view(1, 1, -1).expand_as(representations[view])
                )
        fused = torch.stack(selected).sum(dim=0) / math.sqrt(len(self.active_views))
        return fused * mask.unsqueeze(-1)


class AttributionRQVAE(nn.Module):
    """HTP-compatible RQ-VAE with a controlled point-view encoder intervention."""

    def __init__(
        self,
        config: Mapping[str, object],
        road_features: torch.Tensor,
        num_grids: int,
        active_views: frozenset[str],
        q_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        unet = config["Unet"]
        rq = config["RQ_quant"]
        if not isinstance(unet, Mapping) or not isinstance(rq, Mapping):
            raise ValueError("config must contain Unet and RQ_quant mappings")
        hidden = int(unet["hidden_chs"])
        road_dim = int(unet["road_dim"])
        if road_features.ndim != 2 or road_features.shape[1] != road_dim:
            raise ValueError(
                f"road feature dimension {road_features.shape[-1]} must equal road_dim {road_dim}"
            )
        vocab_sizes = tuple(
            int(rq["base_vocab_size"]) // int(value) for value in rq["vocab_multi"]
        )
        if vocab_sizes != (32, 64, 128, 256) or int(rq["n_codebooks"]) != 4:
            raise ValueError("HTP RQ protocol must remain [32, 64, 128, 256]")

        self.active_views = frozenset(active_views)
        self.q_weights = {
            name: float((q_weights or {}).get(name, 0.0)) for name in LOSS_NAMES
        }
        if q_weights is not None and set(q_weights) != set(LOSS_NAMES):
            raise ValueError(f"q_weights must be exactly {LOSS_NAMES}")
        if any(not math.isfinite(value) or value < 0 for value in self.q_weights.values()):
            raise ValueError("q_weights must be finite and non-negative")
        self.fusion = ViewFusion(hidden, road_features, num_grids, self.active_views)
        self.encoder = Encoder(
            ch=hidden,
            ch_multi=list(unet["ch_multi"]),
            num_res_blocks=int(unet["num_res_blocks"]),
            dropout=float(config["dropout"]),
        )
        # ViewFusion already produces the original encoder's hidden width.
        self.encoder.conv_in = nn.Identity()
        encoder_output_dim = hidden * int(list(unet["ch_multi"])[-1])
        self.rq_pre_quant_linear = nn.Linear(encoder_output_dim, int(rq["d_model"]))
        self.rq_quant = AttributionRQBottleneck(
            base_n_embed=int(rq["base_vocab_size"]),
            codebook_dim=int(rq["d_model"]),
            rq_layer=int(rq["n_codebooks"]),
            rq_multi=list(rq["vocab_multi"]),
            beta=float(rq["beta"]),
            assignment_temperature=0.2,
        )
        self.rq_post_quant_linear = nn.Linear(int(rq["d_model"]), encoder_output_dim)
        self.decoder = Decoder(
            ch=hidden,
            ch_multi=list(unet["ch_multi"]),
            num_res_blocks=int(unet["num_res_blocks"]),
            dropout=float(config["dropout"]),
            road_dim=road_dim,
        )

        route_weight = torch.cat(
            [torch.zeros(1, road_features.shape[1]), road_features.detach().clone()], dim=0
        )
        self.road_emb = nn.Embedding.from_pretrained(route_weight, freeze=False, padding_idx=0)
        self.road_gps_lin = nn.Linear(2, road_dim)
        self.pos_emb = PositionEmbedding(road_dim)
        self.road_trm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                road_dim,
                4,
                4 * road_dim,
                float(config["dropout"]),
                batch_first=True,
                activation=F.silu,
            ),
            int(unet["road_layers"]),
        )
        self.percent_head = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, 1)
        )
        self.dxy_head = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.ReLU(), nn.Linear(4 * hidden, 2)
        )
        latent_dim = int(rq["d_model"])
        self.auxiliary_heads = nn.ModuleDict(
            {
                "speed_state": nn.Linear(latent_dim, 4),
                "turn_state": nn.Linear(latent_dim, 3),
                "road_transition": nn.Linear(latent_dim, 2),
                "grid_transition": nn.Linear(latent_dim, 2),
            }
        )

    def fuse_views(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.fusion(batch)

    def encode_views(self, batch: Mapping[str, torch.Tensor]) -> dict[str, object]:
        fused = self.fuse_views(batch)
        encoded, latent_lengths, lens_types, latent_mask = self.encoder(fused, batch["lengths"])
        valid_prequant = self.rq_pre_quant_linear(encoded[latent_mask])
        quantizer_output = self.rq_quant(valid_prequant)
        valid_decoder_latent = self.rq_post_quant_linear(quantizer_output.straight_through)

        decoder_latent = torch.zeros_like(encoded)
        decoder_latent[latent_mask] = valid_decoder_latent
        prequant_latent = torch.zeros(
            *latent_mask.shape,
            valid_prequant.shape[-1],
            dtype=valid_prequant.dtype,
            device=valid_prequant.device,
        )
        prequant_latent[latent_mask] = valid_prequant
        codes = torch.full(
            (*latent_mask.shape, quantizer_output.codes.shape[-1]),
            -1,
            dtype=torch.long,
            device=quantizer_output.codes.device,
        )
        codes[latent_mask] = quantizer_output.codes
        return {
            "fused": fused,
            "decoder_latent": decoder_latent,
            "prequant_latent": prequant_latent,
            "codes": codes,
            "latent_lengths": latent_lengths,
            "latent_mask": latent_mask,
            "lens_type": torch.stack(lens_types).T,
            "quant_loss": quantizer_output.quant_loss,
            "usage_loss": quantizer_output.usage_loss,
            "rq_info": quantizer_output.layer_info,
        }

    def _route_context(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        route = batch["route"]
        route_lengths = batch["route_lengths"]
        geometry = batch["route_geometry"]
        geometry_lengths = batch["route_geometry_lengths"]
        if int(route_lengths.sum()) != len(geometry_lengths):
            raise ValueError("route geometry population does not match route lengths")
        route_embedding = self.road_emb((route + 1).clamp_min(0))
        geometry_mask = get_padding_mask(geometry_lengths, geometry.device).squeeze(1)
        geometry_embedding = self.road_gps_lin(geometry)
        geometry_embedding = (
            geometry_embedding * geometry_mask.unsqueeze(-1)
        ).sum(dim=1) / geometry_lengths.unsqueeze(1)
        geometry_sequences = torch.split(geometry_embedding, route_lengths.tolist())
        geometry_embedding = pad_sequence(geometry_sequences, batch_first=True, padding_value=0.0)
        route_embedding = route_embedding + geometry_embedding + self.pos_emb(route_embedding)
        route_mask = get_padding_mask(route_lengths, route.device)
        route_embedding = self.road_trm(
            route_embedding, src_key_padding_mask=~route_mask.squeeze(1)
        )
        return route_embedding, route_mask

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        sampling_batch: Mapping[str, torch.Tensor] | None = None,
        auxiliary_targets: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, object]:
        output = self.encode_views(batch)
        route_embedding, route_mask = self._route_context(batch)
        lens_types = [output["lens_type"][:, index].bool() for index in range(3)]
        decoded, point_mask = self.decoder(
            output["decoder_latent"],
            output["latent_lengths"],
            lens_types,
            route_embedding,
            route_mask,
        )
        decoded_valid = decoded[point_mask]
        percent_prediction = self.percent_head(decoded_valid)
        dxy_prediction = self.dxy_head(decoded_valid)
        percent_loss = F.mse_loss(
            percent_prediction, batch["road_percent"][point_mask].unsqueeze(-1)
        )
        dxy_loss = F.mse_loss(dxy_prediction, batch["dxy"][point_mask])
        reconstruction = percent_loss + dxy_loss
        base_loss = reconstruction + output["quant_loss"]
        zero = base_loss * 0.0
        consistency_required = any(
            self.q_weights[name] > 0 for name in ("invariance", "variance", "covariance")
        )
        if consistency_required and sampling_batch is None:
            raise ValueError("sampling_batch is required when Q2/Q3 consistency weights are active")
        if self.q_weights["auxiliary"] > 0 and auxiliary_targets is None:
            raise ValueError("auxiliary_targets are required when the auxiliary weight is active")
        components = {
            "usage": output["usage_loss"],
            "invariance": zero,
            "variance": zero,
            "covariance": zero,
            "auxiliary": zero,
        }
        if sampling_batch is not None:
            sampling_output = self.encode_views(sampling_batch)
            if not torch.equal(output["latent_mask"], sampling_output["latent_mask"]):
                raise ValueError("sampling view changed latent event alignment")
            mask = output["latent_mask"]
            components["invariance"] = invariance_loss(
                output["prequant_latent"], sampling_output["prequant_latent"], mask
            )
            components["variance"] = 0.5 * (
                variance_loss(output["prequant_latent"], mask)
                + variance_loss(sampling_output["prequant_latent"], mask)
            )
            components["covariance"] = 0.5 * (
                covariance_loss(output["prequant_latent"], mask)
                + covariance_loss(sampling_output["prequant_latent"], mask)
            )
            output["sampling_output"] = sampling_output
        if auxiliary_targets is not None:
            components["auxiliary"] = auxiliary_classification_loss(
                output["prequant_latent"], auxiliary_targets, self.auxiliary_heads
            )
        total_loss, reported = combine_q_losses(base_loss, components, self.q_weights)
        output.update(
            {
                "loss": total_loss,
                "base_loss": base_loss,
                "q_loss_components": reported,
                "reconstruction_loss": reconstruction,
                "percent_loss": percent_loss,
                "dxy_loss": dxy_loss,
                "point_mask": point_mask,
            }
        )
        return output

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        code_lengths: torch.Tensor,
        lens_type: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if codes.ndim != 2 or codes.shape[1] != 4:
            raise ValueError("codes must have shape [events, 4]")
        valid_quantized = self.rq_quant.encode(codes.T)
        valid_decoder_latent = self.rq_post_quant_linear(valid_quantized)
        sequences = torch.split(valid_decoder_latent, code_lengths.tolist())
        decoder_latent = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        route_embedding, route_mask = self._route_context(batch)
        decoded, point_mask = self.decoder(
            decoder_latent,
            code_lengths,
            lens_type.bool().T,
            route_embedding,
            route_mask,
        )
        percent_valid = self.percent_head(decoded[point_mask]).squeeze(-1)
        dxy_valid = self.dxy_head(decoded[point_mask])
        percent = torch.zeros(point_mask.shape, dtype=percent_valid.dtype, device=decoded.device)
        dxy = torch.zeros(*point_mask.shape, 2, dtype=dxy_valid.dtype, device=decoded.device)
        percent[point_mask] = percent_valid
        dxy[point_mask] = dxy_valid
        return percent, dxy, point_mask.sum(dim=-1)

    def load_htp_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> tuple[list[str], list[str]]:
        """Map the original GPS encoder input projection into the GPS adapter."""
        mapped = dict(state_dict)
        for suffix in ("weight", "bias"):
            original = f"encoder.conv_in.{suffix}"
            if original in mapped:
                mapped[f"fusion.gps_adapter.{suffix}"] = mapped.pop(original)
        if "pos_emb.pe" in mapped:
            mapped["pos_emb.values"] = mapped.pop("pos_emb.pe")
        incompatible = self.load_state_dict(mapped, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
