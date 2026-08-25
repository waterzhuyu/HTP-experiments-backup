from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .token_protocol import TokenProtocolError, parse_answer


def teacher_forced_nll(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits and labels must have shapes [batch,time,vocab] and [batch,time]")
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    tokens = int((shifted_labels != -100).sum())
    if tokens == 0:
        raise ValueError("teacher-forced metrics require at least one supervised token")
    loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    ) / tokens
    nll = float(loss)
    return {"tokens": tokens, "nll": nll, "perplexity": math.exp(nll)}


def _first_code_error(predicted: list[list[int]], target: list[list[int]]) -> str | None:
    for predicted_event, target_event in zip(predicted, target):
        for layer, (predicted_code, target_code) in enumerate(zip(predicted_event, target_event)):
            if predicted_code != target_code:
                return f"layer_{layer}"
    return None


def semantic_id_metrics(
    predictions: Sequence[str], targets: Sequence[str]
) -> dict[str, object]:
    if len(predictions) != len(targets):
        raise ValueError("prediction and target population mismatch")
    if not targets:
        raise ValueError("semantic ID metrics require a non-empty population")
    layer_correct = [0, 0, 0, 0]
    layer_total = [0, 0, 0, 0]
    sampling_correct = 0
    exact = 0
    valid = 0
    first_errors: Counter[str] = Counter()
    for predicted_text, target_text in zip(predictions, targets):
        target = parse_answer(target_text)
        target_codes = target["codes"]
        try:
            predicted = parse_answer(predicted_text)
        except TokenProtocolError:
            first_errors["malformed"] += 1
            for event in target_codes:
                for layer in range(len(event)):
                    layer_total[layer] += 1
            continue
        valid += 1
        sampling_matches = predicted["lens_type"] == target["lens_type"]
        sampling_correct += int(sampling_matches)
        predicted_codes = predicted["codes"]
        same_length = len(predicted_codes) == len(target_codes)
        for predicted_event, target_event in zip(predicted_codes, target_codes):
            for layer, (predicted_code, target_code) in enumerate(
                zip(predicted_event, target_event)
            ):
                layer_total[layer] += 1
                layer_correct[layer] += int(predicted_code == target_code)
        sequence_matches = sampling_matches and same_length and predicted_codes == target_codes
        exact += int(sequence_matches)
        if sequence_matches:
            first_errors["none"] += 1
        elif not sampling_matches:
            first_errors["sampling"] += 1
        elif not same_length:
            first_errors["length"] += 1
        else:
            first_errors[_first_code_error(predicted_codes, target_codes) or "unknown"] += 1
    population = len(targets)
    return {
        "population": population,
        "valid_predictions": valid,
        "validity_rate": valid / population,
        "sampling_accuracy": sampling_correct / population,
        "layer_accuracy": [
            correct / total if total else float("nan")
            for correct, total in zip(layer_correct, layer_total)
        ],
        "layer_denominators": layer_total,
        "sequence_exact_accuracy": exact / population,
        "first_error_histogram": dict(first_errors),
    }
