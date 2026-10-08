from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score


def _valid_codes(codes: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    if codes.ndim != 3 or codes.shape[:2] != mask.shape:
        raise ValueError("codes must be [batch, event, layer] and align with mask")
    values = codes[mask].detach().cpu().numpy().astype(np.int64, copy=False)
    if not len(values):
        raise ValueError("metrics require at least one valid event")
    return values


def codebook_statistics(
    codes: torch.Tensor,
    mask: torch.Tensor,
    vocab_sizes: Sequence[int],
) -> list[dict[str, float | int]]:
    values = _valid_codes(codes, mask)
    if values.shape[1] != len(vocab_sizes):
        raise ValueError("vocab_sizes must match code layers")
    result: list[dict[str, float | int]] = []
    for layer, size_value in enumerate(vocab_sizes):
        size = int(size_value)
        layer_codes = values[:, layer]
        if layer_codes.min() < 0 or layer_codes.max() >= size:
            raise ValueError(f"layer {layer} contains code outside [0, {size})")
        counts = np.bincount(layer_codes, minlength=size).astype(np.float64)
        probabilities = counts[counts > 0] / counts.sum()
        active = int(np.count_nonzero(counts))
        perplexity = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
        result.append(
            {
                "layer": layer,
                "vocab_size": size,
                "events": int(len(layer_codes)),
                "active_codes": active,
                "active_fraction": active / size,
                "perplexity": perplexity,
                "perplexity_fraction": perplexity / size,
                "max_share": float(probabilities.max()),
            }
        )
    return result


def _entropy(counts: Sequence[int]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    probabilities = values[values > 0] / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def prefix_conditional_entropy(
    codes: torch.Tensor,
    mask: torch.Tensor,
    vocab_sizes: Sequence[int],
) -> list[float]:
    values = _valid_codes(codes, mask)
    if values.shape[1] != len(vocab_sizes):
        raise ValueError("vocab_sizes must match code layers")
    normalized: list[float] = []
    for layer, size_value in enumerate(vocab_sizes):
        size = int(size_value)
        if size <= 1:
            normalized.append(0.0)
            continue
        if layer == 0:
            counts = np.bincount(values[:, 0], minlength=size)
            conditional = _entropy(counts)
        else:
            groups: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
            for row in values:
                groups[tuple(int(value) for value in row[:layer])][int(row[layer])] += 1
            conditional = sum(
                (sum(counter.values()) / len(values)) * _entropy(list(counter.values()))
                for counter in groups.values()
            )
        normalized.append(float(conditional / math.log(size)))
    return normalized


def semantic_collision_rates(
    codes: torch.Tensor,
    mask: torch.Tensor,
    labels: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    values = _valid_codes(codes, mask)
    result: dict[str, float] = {}
    tuples = [tuple(int(value) for value in row) for row in values]
    for name, label_tensor in labels.items():
        if label_tensor.shape != mask.shape:
            raise ValueError(f"semantic label {name} does not align with code mask")
        valid_labels = label_tensor[mask].detach().cpu().numpy().astype(np.int64, copy=False)
        groups: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        valid_count = 0
        for code_tuple, label in zip(tuples, valid_labels):
            if int(label) == -100:
                continue
            groups[code_tuple][int(label)] += 1
            valid_count += 1
        if valid_count == 0:
            result[name] = float("nan")
            continue
        conflicts = sum(sum(counter.values()) - max(counter.values()) for counter in groups.values())
        result[name] = conflicts / valid_count
    return result


def sampling_robustness(
    original_codes: torch.Tensor,
    sampled_codes: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, object]:
    original = _valid_codes(original_codes, mask)
    sampled = _valid_codes(sampled_codes, mask)
    if original.shape != sampled.shape:
        raise ValueError("sampling code populations do not align")
    agreements = (original == sampled).mean(axis=0)
    layer_ami = [
        float(adjusted_mutual_info_score(original[:, layer], sampled[:, layer]))
        for layer in range(original.shape[1])
    ]
    return {
        "events": int(len(original)),
        "layer_agreement": [float(value) for value in agreements],
        "layer_ami": layer_ami,
        "tuple_agreement": float(np.all(original == sampled, axis=1).mean()),
    }


def validity_gate(
    layer_statistics: Sequence[Mapping[str, float | int]],
    all_losses_finite: bool,
    oracle_error_ratio: float,
    minimum_active_fraction: float = 0.10,
    minimum_perplexity_fraction: float = 0.05,
    maximum_share: float = 0.50,
    maximum_oracle_error_ratio: float = 1.25,
) -> dict[str, object]:
    reasons: list[str] = []
    if not all_losses_finite:
        reasons.append("non-finite loss")
    if not math.isfinite(oracle_error_ratio):
        reasons.append("non-finite oracle decode error ratio")
    elif oracle_error_ratio > maximum_oracle_error_ratio:
        reasons.append(
            f"oracle decode error ratio {oracle_error_ratio:.4f} exceeds {maximum_oracle_error_ratio:.2f}"
        )
    for index, statistics in enumerate(layer_statistics):
        active = float(statistics["active_fraction"])
        perplexity = float(statistics["perplexity_fraction"])
        max_share = float(statistics["max_share"])
        if active < minimum_active_fraction:
            reasons.append(
                f"layer {index} active_fraction {active:.4f} below {minimum_active_fraction:.2f}"
            )
        if perplexity < minimum_perplexity_fraction:
            reasons.append(
                f"layer {index} perplexity_fraction {perplexity:.4f} below {minimum_perplexity_fraction:.2f}"
            )
        if max_share > maximum_share:
            reasons.append(f"layer {index} max_share {max_share:.4f} exceeds {maximum_share:.2f}")
    return {"passed": not reasons, "reasons": reasons}
