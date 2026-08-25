from __future__ import annotations

import math

import pytest
import torch

from model.rqvae import RQVAE
from staged_attribution.losses import derive_auxiliary_targets
from staged_attribution.model import AttributionRQVAE
from staged_attribution.views import make_sampling_view


VIEW_MASKS = (
    frozenset({"gps"}),
    frozenset({"road"}),
    frozenset({"grid"}),
    frozenset({"gps", "road"}),
    frozenset({"gps", "grid"}),
    frozenset({"road", "grid"}),
    frozenset({"gps", "road", "grid"}),
)


def tiny_config() -> dict[str, object]:
    return {
        "dropout": 0.0,
        "Unet": {
            "ch_multi": [1, 2, 2, 4],
            "hidden_chs": 8,
            "num_res_blocks": 1,
            "road_dim": 8,
            "road_layers": 1,
        },
        "RQ_quant": {
            "n_codebooks": 4,
            "base_vocab_size": 256,
            "vocab_multi": [8, 4, 2, 1],
            "d_model": 8,
            "beta": 0.25,
        },
    }


def point_batch() -> dict[str, torch.Tensor]:
    return {
        "gps": torch.tensor(
            [
                [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6],
                 [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0]],
                [[0.9, 0.8], [0.8, 0.7], [0.7, 0.6], [0.6, 0.5], [0.5, 0.4],
                 [0.4, 0.3], [0.3, 0.2], [0.2, 0.1], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "grid": torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3, 4], [4, 4, 3, 3, 2, 2, 1, 1, -1]],
            dtype=torch.long,
        ),
        "point_road": torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3, 4], [4, 4, 3, 3, 2, 2, 1, 1, -1]],
            dtype=torch.long,
        ),
        "lengths": torch.tensor([9, 8], dtype=torch.long),
    }


def make_model(views: frozenset[str]) -> AttributionRQVAE:
    torch.manual_seed(7)
    road_features = torch.randn(6, 8)
    return AttributionRQVAE(
        config=tiny_config(),
        road_features=road_features,
        num_grids=10,
        active_views=views,
    ).eval()


@pytest.mark.parametrize("views", VIEW_MASKS)
def test_all_view_masks_keep_htp_code_protocol(views: frozenset[str]) -> None:
    model = make_model(views)

    output = model.encode_views(point_batch())

    assert output["codes"].shape == (2, 2, 4)
    assert output["latent_lengths"].tolist() == [math.ceil(9 / 8), math.ceil(8 / 8)]
    assert output["latent_mask"].tolist() == [[True, True], [True, False]]
    for layer, size in enumerate([32, 64, 128, 256]):
        valid = output["codes"][..., layer][output["latent_mask"]]
        assert int(valid.min()) >= 0
        assert int(valid.max()) < size
    assert (output["codes"][~output["latent_mask"]] == -1).all()


def test_parameter_budget_is_identical_across_view_masks() -> None:
    counts = []
    for views in VIEW_MASKS:
        model = make_model(views)
        counts.append(
            (
                sum(parameter.numel() for parameter in model.parameters()),
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            )
        )

    assert len(set(counts)) == 1


def test_disabled_view_perturbation_has_no_effect_on_fusion() -> None:
    model = make_model(frozenset({"gps"}))
    batch = point_batch()
    perturbed = {name: value.clone() for name, value in batch.items()}
    perturbed["grid"][:, :-1] = (perturbed["grid"][:, :-1] + 3) % 10
    perturbed["point_road"][:, :-1] = (perturbed["point_road"][:, :-1] + 2) % 6

    actual = model.fuse_views(batch)
    changed = model.fuse_views(perturbed)

    assert torch.equal(actual, changed)


def test_enabled_view_perturbation_changes_fusion() -> None:
    model = make_model(frozenset({"gps", "road", "grid"}))
    batch = point_batch()
    perturbed = {name: value.clone() for name, value in batch.items()}
    perturbed["point_road"][:, :-1] = (perturbed["point_road"][:, :-1] + 2) % 6

    assert not torch.allclose(model.fuse_views(batch), model.fuse_views(perturbed))


def test_forward_and_decode_preserve_original_lengths() -> None:
    model = make_model(frozenset({"gps", "road", "grid"}))
    batch = point_batch()
    batch.update(
        {
            "route": torch.tensor([[0, 1], [4, -1]], dtype=torch.long),
            "route_lengths": torch.tensor([2, 1], dtype=torch.long),
            "route_geometry": torch.tensor(
                [
                    [[0.0, 0.0], [0.1, 0.1]],
                    [[0.1, 0.1], [0.2, 0.2]],
                    [[0.8, 0.8], [0.9, 0.9]],
                ],
                dtype=torch.float32,
            ),
            "route_geometry_lengths": torch.tensor([2, 2, 2], dtype=torch.long),
            "dxy": torch.zeros(2, 9, 2),
            "road_percent": torch.zeros(2, 9),
        }
    )

    output = model(batch)
    valid_codes = output["codes"][output["latent_mask"]]
    percent, dxy, decoded_lengths = model.decode(
        valid_codes,
        output["latent_lengths"],
        output["lens_type"],
        batch,
    )

    assert torch.isfinite(output["loss"])
    assert output["lens_type"].shape == (2, 3)
    assert decoded_lengths.tolist() == [9, 8]
    assert percent.shape == (2, 9)
    assert dxy.shape == (2, 9, 2)


def test_original_htp_checkpoint_maps_without_unexpected_keys() -> None:
    torch.manual_seed(11)
    road_features = torch.randn(6, 8)
    baseline = RQVAE(tiny_config(), road_features)
    model = AttributionRQVAE(
        tiny_config(),
        road_features,
        num_grids=10,
        active_views=frozenset({"gps"}),
    )

    missing, unexpected = model.load_htp_state_dict(baseline.state_dict())

    assert unexpected == []
    assert missing
    assert all(
        key.startswith(("fusion.", "auxiliary_heads."))
        for key in missing
    )
    assert torch.equal(
        model.fusion.gps_adapter.weight,
        baseline.encoder.conv_in.weight,
    )
    assert torch.equal(
        model.decoder.conv_in.weight,
        baseline.decoder.conv_in.weight,
    )


def complete_batch() -> dict[str, torch.Tensor]:
    batch = point_batch()
    batch.update(
        {
            "time": torch.arange(9).float().repeat(2, 1),
            "route": torch.tensor([[0, 1], [4, -1]], dtype=torch.long),
            "route_lengths": torch.tensor([2, 1], dtype=torch.long),
            "route_geometry": torch.tensor(
                [
                    [[0.0, 0.0], [0.1, 0.1]],
                    [[0.1, 0.1], [0.2, 0.2]],
                    [[0.8, 0.8], [0.9, 0.9]],
                ],
                dtype=torch.float32,
            ),
            "route_geometry_lengths": torch.tensor([2, 2, 2], dtype=torch.long),
            "dxy": torch.zeros(2, 9, 2),
            "road_percent": torch.zeros(2, 9),
        }
    )
    return batch


def q3_weights() -> dict[str, float]:
    return {
        "usage": 0.05,
        "invariance": 0.05,
        "variance": 0.05,
        "covariance": 0.005,
        "auxiliary": 0.1,
    }


def test_q0_reports_components_without_changing_base_loss() -> None:
    model = make_model(frozenset({"gps", "road", "grid"}))

    output = model(complete_batch())

    assert set(output["q_loss_components"]) == set(q3_weights())
    assert torch.allclose(output["loss"], output["base_loss"])
    assert torch.allclose(
        output["base_loss"],
        output["reconstruction_loss"] + output["quant_loss"],
    )


def test_q3_requires_sampling_view_and_auxiliary_targets() -> None:
    torch.manual_seed(7)
    model = AttributionRQVAE(
        tiny_config(),
        torch.randn(6, 8),
        num_grids=10,
        active_views=frozenset({"gps", "road", "grid"}),
        q_weights=q3_weights(),
    ).eval()

    with pytest.raises(ValueError, match="sampling_batch"):
        model(complete_batch())


def test_q3_total_matches_frozen_weighted_components() -> None:
    torch.manual_seed(7)
    model = AttributionRQVAE(
        tiny_config(),
        torch.randn(6, 8),
        num_grids=10,
        active_views=frozenset({"gps", "road", "grid"}),
        q_weights=q3_weights(),
    ).eval()
    batch = complete_batch()
    sampling = make_sampling_view(batch, seed=99, keep_probability=0.5)
    targets = derive_auxiliary_targets(
        batch,
        speed_edges=torch.tensor([0.01, 0.03, 0.1]),
        turn_edges=torch.tensor([0.1, 0.5]),
        meters_per_coordinate=(1.0, 1.0),
    )

    output = model(batch, sampling_batch=sampling, auxiliary_targets=targets)
    expected = output["base_loss"]
    for name, weight in q3_weights().items():
        expected = expected + weight * output["q_loss_components"][name]

    assert torch.allclose(output["loss"], expected)
    assert all(torch.isfinite(value) for value in output["q_loss_components"].values())
