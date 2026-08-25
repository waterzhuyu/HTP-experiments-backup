from __future__ import annotations

import numpy as np
import pytest
import torch

from staged_attribution.decode_stage3 import decode_variant_records, reconstruct_gps


class FakeDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def decode(self, codes, code_lengths, lens_type, batch):
        self.calls += 1
        assert codes.tolist() == [[1, 2, 3, 4]]
        assert code_lengths.tolist() == [1]
        assert lens_type.tolist() == [[0, 0, 0]]
        assert batch["route"].tolist() == [[2]]
        return (
            torch.tensor([[0.25, 0.25]], device=codes.device),
            torch.zeros(1, 2, 2, device=codes.device),
            torch.tensor([2], device=codes.device),
        )


def context() -> dict[str, torch.Tensor]:
    return {
        "route": torch.tensor([2]),
        "route_lengths": torch.tensor(1),
        "route_geometry": torch.tensor([[[0.0, 0.0], [2.0, 0.0]]]),
        "route_geometry_lengths": torch.tensor([2]),
    }


def test_decode_preserves_failures_and_never_changes_oracle_route() -> None:
    model = FakeDecoder()
    records = [
        {
            "traj_id": 3,
            "variant": "g00",
            "status": "ready",
            "codes": [[1, 2, 3, 4]],
            "lens_type": [0, 0, 0],
            "road_seq": [2],
        },
        {
            "traj_id": 4,
            "variant": "g11",
            "status": "failed",
            "road_seq": [5],
            "failure": {"reason": "generation_truncated", "message": "x"},
        },
    ]

    decoded = decode_variant_records(model, records, {3: context()}, torch.device("cpu"))

    assert model.calls == 1
    assert decoded[0]["status"] == "decoded"
    assert decoded[0]["road_seq"] == [2]
    assert decoded[0]["pred_percent"] == pytest.approx([0.25, 0.25])
    assert decoded[0]["pred_dxy"] == [[0.0, 0.0], [0.0, 0.0]]
    assert decoded[1]["status"] == "failed"
    assert decoded[1]["failure"]["reason"] == "generation_truncated"


def test_reconstruct_gps_matches_original_percent_and_dxy_semantics() -> None:
    decoded = {
        "traj_id": 3,
        "status": "decoded",
        "road_seq": [2],
        "pred_percent": [0.25, 0.25],
        "pred_dxy": [[0.0, 0.0], [0.0, 0.0]],
    }
    z_score = {
        "mean_percent_dist": 0.0,
        "std_percent_dist": 1.0,
        "mean_dx": 0.0,
        "std_dx": 1.0,
        "mean_dy": 0.0,
        "std_dy": 1.0,
    }

    result = reconstruct_gps(decoded, {2: np.asarray([[0.0, 0.0], [2.0, 0.0]])}, z_score)

    assert result["status"] == "reconstructed"
    np.testing.assert_allclose(result["gps_traj"], [[0.5, 0.0], [1.0, 0.0]])
    assert result["road_traj"] == [2, 2]


def test_reconstruct_failure_stays_in_population() -> None:
    failed = {"traj_id": 8, "status": "failed", "failure": {"reason": "x"}}

    assert reconstruct_gps(failed, {}, {}) == failed
