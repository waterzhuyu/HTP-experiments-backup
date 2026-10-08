from __future__ import annotations

import torch

from staged_attribution.views import make_sampling_view


def batch() -> dict[str, torch.Tensor]:
    return {
        "gps": torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
                [[9.0, 0.0], [8.0, 1.0], [7.0, 2.0], [6.0, 3.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        "grid": torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, -1, -1]]),
        "point_road": torch.tensor([[0, 0, 1, 1, 2, 2], [3, 3, 4, 4, -1, -1]]),
        "time": torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 2.0, 4.0, 6.0, 0.0, 0.0]]),
        "lengths": torch.tensor([6, 4]),
    }


def test_sampling_view_is_deterministic_and_keeps_alignment() -> None:
    original = batch()

    first = make_sampling_view(original, seed=123, keep_probability=0.45)
    second = make_sampling_view(original, seed=123, keep_probability=0.45)

    for name in ("gps", "grid", "point_road", "sampling_kept_mask"):
        assert torch.equal(first[name], second[name])
    assert torch.equal(first["time"], original["time"])
    assert torch.equal(first["lengths"], original["lengths"])
    assert first["gps"].shape == original["gps"].shape
    assert first["grid"].shape == original["grid"].shape


def test_sampling_view_forces_endpoints_and_preserves_padding() -> None:
    original = batch()
    sampled = make_sampling_view(original, seed=7, keep_probability=0.1)

    for row, length in enumerate(original["lengths"].tolist()):
        assert sampled["sampling_kept_mask"][row, 0]
        assert sampled["sampling_kept_mask"][row, length - 1]
        assert torch.equal(sampled["gps"][row, 0], original["gps"][row, 0])
        assert torch.equal(sampled["gps"][row, length - 1], original["gps"][row, length - 1])
    assert sampled["grid"][1, 4:].tolist() == [-1, -1]
    assert sampled["point_road"][1, 4:].tolist() == [-1, -1]
    assert sampled["gps"][1, 4:].eq(0).all()


def test_discrete_interpolation_only_uses_kept_source_values() -> None:
    original = batch()
    sampled = make_sampling_view(original, seed=99, keep_probability=0.35)

    for row, length in enumerate(original["lengths"].tolist()):
        kept = sampled["sampling_kept_mask"][row, :length]
        allowed_grid = set(original["grid"][row, :length][kept].tolist())
        allowed_road = set(original["point_road"][row, :length][kept].tolist())
        assert set(sampled["grid"][row, :length].tolist()) <= allowed_grid
        assert set(sampled["point_road"][row, :length].tolist()) <= allowed_road
