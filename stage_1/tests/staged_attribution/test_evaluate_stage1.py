from __future__ import annotations

import math

import torch

from staged_attribution.evaluate_stage1 import (
    codebook_statistics,
    prefix_conditional_entropy,
    sampling_robustness,
    semantic_collision_rates,
    validity_gate,
)


VOCAB = (2, 4, 4, 4)


def test_codebook_statistics_match_balanced_assignments() -> None:
    codes = torch.tensor(
        [
            [[0, 0, 0, 0], [1, 1, 1, 1]],
            [[0, 2, 2, 2], [1, 3, 3, 3]],
        ]
    )
    mask = torch.ones(2, 2, dtype=torch.bool)

    statistics = codebook_statistics(codes, mask, VOCAB)

    assert statistics[0]["active_codes"] == 2
    assert statistics[0]["active_fraction"] == 1.0
    assert math.isclose(statistics[0]["perplexity"], 2.0, rel_tol=1e-6)
    assert statistics[0]["max_share"] == 0.5
    assert statistics[1]["active_codes"] == 4
    assert statistics[1]["active_fraction"] == 1.0


def test_prefix_conditional_entropy_is_zero_for_deterministic_child() -> None:
    codes = torch.tensor(
        [[[0, 2, 0, 0], [0, 2, 1, 1], [1, 3, 0, 0], [1, 3, 1, 1]]]
    )
    mask = torch.ones(1, 4, dtype=torch.bool)

    entropy = prefix_conditional_entropy(codes, mask, VOCAB)

    assert math.isclose(entropy[0], 1.0, rel_tol=1e-6)
    assert math.isclose(entropy[1], 0.0, abs_tol=1e-9)


def test_semantic_collision_rate_counts_minority_labels_within_tuple() -> None:
    codes = torch.tensor(
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]]
    )
    mask = torch.ones(1, 4, dtype=torch.bool)
    labels = {
        "speed_state": torch.tensor([[0, 0, 1, 1]]),
        "road_transition": torch.tensor([[0, 0, 0, 1]]),
    }

    rates = semantic_collision_rates(codes, mask, labels)

    assert math.isclose(rates["speed_state"], 0.25)
    assert rates["road_transition"] == 0.0


def test_sampling_robustness_reports_layer_and_tuple_agreement() -> None:
    original = torch.tensor(
        [[[0, 0, 0, 0], [1, 1, 1, 1], [0, 2, 2, 2], [1, 3, 3, 3]]]
    )
    sampled = original.clone()
    sampled[0, 1, 3] = 2
    mask = torch.ones(1, 4, dtype=torch.bool)

    metrics = sampling_robustness(original, sampled, mask)

    assert metrics["layer_agreement"] == [1.0, 1.0, 1.0, 0.75]
    assert metrics["tuple_agreement"] == 0.75
    assert len(metrics["layer_ami"]) == 4
    assert all(math.isfinite(value) for value in metrics["layer_ami"])


def test_validity_gate_rejects_collapsed_layer() -> None:
    codes = torch.zeros(2, 8, 4, dtype=torch.long)
    mask = torch.ones(2, 8, dtype=torch.bool)
    statistics = codebook_statistics(codes, mask, (32, 64, 128, 256))

    gate = validity_gate(
        statistics,
        all_losses_finite=True,
        oracle_error_ratio=1.0,
    )

    assert gate["passed"] is False
    assert any("active_fraction" in reason for reason in gate["reasons"])
    assert any("max_share" in reason for reason in gate["reasons"])


def test_validity_gate_accepts_noncollapsed_layers() -> None:
    codes = torch.empty(1, 256, 4, dtype=torch.long)
    for layer, size in enumerate((32, 64, 128, 256)):
        codes[0, :, layer] = torch.arange(256) % size
    mask = torch.ones(1, 256, dtype=torch.bool)
    statistics = codebook_statistics(codes, mask, (32, 64, 128, 256))

    gate = validity_gate(
        statistics,
        all_losses_finite=True,
        oracle_error_ratio=1.1,
    )

    assert gate == {"passed": True, "reasons": []}


def test_validity_gate_rejects_nonfinite_oracle_regression() -> None:
    statistics = [
        {"active_fraction": 1.0, "perplexity_fraction": 1.0, "max_share": 0.01}
        for _ in range(4)
    ]

    gate = validity_gate(
        statistics,
        all_losses_finite=False,
        oracle_error_ratio=1.3,
    )

    assert gate["passed"] is False
    assert "non-finite loss" in gate["reasons"]
    assert "oracle decode error ratio 1.3000 exceeds 1.25" in gate["reasons"]
