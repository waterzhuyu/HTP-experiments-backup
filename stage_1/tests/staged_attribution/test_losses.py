from __future__ import annotations

import torch

from staged_attribution.losses import (
    combine_q_losses,
    covariance_loss,
    derive_auxiliary_targets,
    invariance_loss,
    variance_loss,
)


def test_invariance_is_zero_for_identical_latents() -> None:
    latent = torch.randn(2, 3, 5)
    mask = torch.tensor([[True, True, True], [True, True, False]])

    assert torch.allclose(invariance_loss(latent, latent, mask), torch.tensor(0.0), atol=1e-6)


def test_variance_penalizes_collapse() -> None:
    mask = torch.ones(2, 4, dtype=torch.bool)
    collapsed = torch.zeros(2, 4, 3)
    spread = torch.tensor(
        [
            [[-2.0, 0.0, 2.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [2.0, 0.0, -2.0]],
            [[-2.0, 1.0, -2.0], [-1.0, -1.0, 1.0], [1.0, 0.0, -1.0], [2.0, 1.0, 2.0]],
        ]
    )

    assert variance_loss(collapsed, mask) > variance_loss(spread, mask)


def test_covariance_loss_is_finite_and_nonnegative() -> None:
    latent = torch.randn(3, 4, 5)
    mask = torch.ones(3, 4, dtype=torch.bool)

    value = covariance_loss(latent, mask)

    assert torch.isfinite(value)
    assert value >= 0


def test_auxiliary_targets_use_eight_point_event_support() -> None:
    batch = {
        "gps": torch.stack((torch.arange(8), torch.zeros(8)), dim=-1).float().unsqueeze(0),
        "time": torch.arange(8).float().unsqueeze(0),
        "grid": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]]),
        "point_road": torch.tensor([[2, 2, 2, 2, 3, 3, 3, 3]]),
        "lengths": torch.tensor([8]),
    }

    targets = derive_auxiliary_targets(
        batch,
        speed_edges=torch.tensor([0.5, 1.5, 3.0]),
        turn_edges=torch.tensor([0.1, 0.5]),
        meters_per_coordinate=(1.0, 1.0),
    )

    assert targets["mask"].tolist() == [[True]]
    assert targets["speed_state"].tolist() == [[1]]
    assert targets["turn_state"].tolist() == [[0]]
    assert targets["road_transition"].tolist() == [[1]]
    assert targets["grid_transition"].tolist() == [[1]]


def test_auxiliary_targets_mark_padded_events_ignore_index() -> None:
    batch = {
        "gps": torch.zeros(2, 9, 2),
        "time": torch.arange(9).float().repeat(2, 1),
        "grid": torch.zeros(2, 9, dtype=torch.long),
        "point_road": torch.zeros(2, 9, dtype=torch.long),
        "lengths": torch.tensor([9, 8]),
    }

    targets = derive_auxiliary_targets(
        batch,
        speed_edges=torch.tensor([0.5, 1.5, 3.0]),
        turn_edges=torch.tensor([0.1, 0.5]),
        meters_per_coordinate=(1.0, 1.0),
    )

    assert targets["mask"].tolist() == [[True, True], [True, False]]
    for name in ("speed_state", "turn_state", "road_transition", "grid_transition"):
        assert targets[name][1, 1].item() == -100


def test_combine_q_losses_reports_every_component_and_exact_weighting() -> None:
    base = torch.tensor(2.0)
    components = {
        "usage": torch.tensor(1.0),
        "invariance": torch.tensor(2.0),
        "variance": torch.tensor(3.0),
        "covariance": torch.tensor(4.0),
        "auxiliary": torch.tensor(5.0),
    }
    weights = {
        "usage": 0.05,
        "invariance": 0.05,
        "variance": 0.05,
        "covariance": 0.005,
        "auxiliary": 0.1,
    }

    total, reported = combine_q_losses(base, components, weights)

    assert torch.allclose(total, torch.tensor(2.82))
    assert set(reported) == set(components)
    assert all(torch.equal(reported[name], components[name]) for name in components)
