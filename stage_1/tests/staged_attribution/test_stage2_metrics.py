from __future__ import annotations

import math

import pytest
import torch

from staged_attribution.evaluate_stage2 import semantic_id_metrics, teacher_forced_nll
from staged_attribution.token_protocol import serialize_answer


def answer(lens, codes) -> str:
    return serialize_answer(lens, codes)


def test_teacher_forced_nll_uses_causal_shift_and_ignores_mask() -> None:
    labels = torch.tensor([[-100, 1, 2, 3]])
    logits = torch.full((1, 4, 5), -9.0)
    logits[0, 0, 1] = 9.0
    logits[0, 1, 2] = 9.0
    logits[0, 2, 3] = 9.0

    result = teacher_forced_nll(logits, labels)

    assert result["tokens"] == 3
    assert result["nll"] < 1e-5
    assert result["perplexity"] == pytest.approx(math.exp(result["nll"]))


def test_semantic_metrics_report_layers_exact_and_first_error() -> None:
    targets = [
        answer([0, 0, 0], [[1, 2, 3, 4], [2, 3, 4, 5]]),
        answer([1, 0, 1], [[5, 6, 7, 8]]),
        answer([0, 1, 0], [[9, 10, 11, 12]]),
        answer([1, 1, 1], [[3, 4, 5, 6]]),
    ]
    predictions = [
        targets[0],
        answer([0, 0, 1], [[5, 6, 7, 8]]),
        answer([0, 1, 0], [[9, 10, 99, 12]]),
        "malformed",
    ]

    result = semantic_id_metrics(predictions, targets)

    assert result["population"] == 4
    assert result["valid_predictions"] == 3
    assert result["sampling_accuracy"] == pytest.approx(2 / 4)
    assert result["sequence_exact_accuracy"] == pytest.approx(1 / 4)
    assert result["layer_accuracy"][0] == pytest.approx(4 / 5)
    assert result["layer_accuracy"][1] == pytest.approx(4 / 5)
    assert result["layer_accuracy"][2] == pytest.approx(3 / 5)
    assert result["layer_accuracy"][3] == pytest.approx(4 / 5)
    assert result["first_error_histogram"] == {
        "none": 1,
        "sampling": 1,
        "layer_2": 1,
        "malformed": 1,
    }


def test_length_mismatch_is_a_failure_without_dropping_population() -> None:
    target = answer([0, 0, 0], [[1, 2, 3, 4], [5, 6, 7, 8]])
    short = answer([0, 0, 0], [[1, 2, 3, 4]])

    result = semantic_id_metrics([short], [target])

    assert result["population"] == 1
    assert result["sequence_exact_accuracy"] == 0.0
    assert result["first_error_histogram"] == {"length": 1}
    assert result["layer_denominators"] == [1, 1, 1, 1]


def test_metrics_reject_population_mismatch() -> None:
    with pytest.raises(ValueError, match="population"):
        semantic_id_metrics([], [answer([0, 0, 0], [[1, 2, 3, 4]])])
