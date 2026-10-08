from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import expected_mutual_information


def contingency(
    left: np.ndarray,
    right: np.ndarray,
    size: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} versus {right.shape}")
    values = np.bincount(
        left.astype(np.int64) * size + right.astype(np.int64),
        weights=weights,
        minlength=size * size,
    )
    return values.reshape(size, size).astype(np.float64, copy=False)


def entropy(probability: np.ndarray) -> float:
    positive = probability[probability > 0]
    return float(-(positive * np.log(positive)).sum())


def metrics_from_contingency(
    matrix: np.ndarray,
    compute_adjusted: bool = True,
) -> dict[str, float | int]:
    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("empty contingency matrix")
    joint = matrix / total
    left_probability = joint.sum(axis=1)
    right_probability = joint.sum(axis=0)
    h_left = entropy(left_probability)
    h_right = entropy(right_probability)
    rows, cols = np.nonzero(joint)
    mutual_information = float(
        sum(
            joint[row, col]
            * math.log(joint[row, col] / (left_probability[row] * right_probability[col]))
            for row, col in zip(rows, cols)
        )
    )
    if compute_adjusted:
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = float(expected_mutual_information(matrix, int(round(total))))
        denominator = 0.5 * (h_left + h_right) - expected
        ami = (mutual_information - expected) / denominator if abs(denominator) > 1e-12 else 1.0
    else:
        ami = float("nan")
    nmi_denominator = math.sqrt(h_left * h_right)
    nmi = mutual_information / nmi_denominator if nmi_denominator > 0 else 0.0
    row_index, col_index = linear_sum_assignment(-matrix)
    match = float(matrix[row_index, col_index].sum() / total)
    active_left = int(np.count_nonzero(matrix.sum(axis=1)))
    active_right = int(np.count_nonzero(matrix.sum(axis=0)))
    size = matrix.shape[0]
    h_right_given_left = max(h_right - mutual_information, 0.0)
    h_left_given_right = max(h_left - mutual_information, 0.0)
    return {
        "samples": int(round(total)),
        "active_left": active_left,
        "active_right": active_right,
        "optimal_match_rate": match,
        "mutual_information": mutual_information,
        "adjusted_mutual_information": float(ami),
        "normalized_mutual_information": float(nmi),
        "h_left": h_left,
        "h_right": h_right,
        "h_right_given_left": h_right_given_left,
        "h_left_given_right": h_left_given_right,
        "normalized_h_right_given_left": h_right_given_left / math.log(size),
        "normalized_h_left_given_right": h_left_given_right / math.log(size),
        "variation_of_information": h_right_given_left + h_left_given_right,
    }


def optimal_mapping(current: np.ndarray, reference: np.ndarray, size: int) -> np.ndarray:
    matrix = contingency(current, reference, size)
    rows, cols = linear_sum_assignment(-matrix)
    mapping = np.arange(size, dtype=np.int64)
    mapping[rows] = cols
    return mapping


def reindex_matrix(matrix: np.ndarray, left_mapping: np.ndarray, right_mapping: np.ndarray) -> np.ndarray:
    result = np.zeros_like(matrix)
    result[left_mapping[:, None], right_mapping[None, :]] = matrix
    return result


def distribution_similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    p = left.astype(np.float64).reshape(-1)
    q = right.astype(np.float64).reshape(-1)
    p /= max(p.sum(), 1.0)
    q /= max(q.sum(), 1.0)
    midpoint = 0.5 * (p + q)

    def kl(values: np.ndarray, target: np.ndarray) -> float:
        positive = values > 0
        return float((values[positive] * np.log(values[positive] / target[positive])).sum())

    js = 0.5 * (kl(p, midpoint) + kl(q, midpoint))
    cosine = float(p @ q / max(np.linalg.norm(p) * np.linalg.norm(q), 1e-12))
    return {"jensen_shannon_divergence": js, "cosine_similarity": cosine}


def trajectory_event_ids(offsets: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))


def bootstrap_metrics(
    left: np.ndarray,
    right: np.ndarray,
    offsets: np.ndarray,
    size: int,
    seed: int,
    replicates: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    event_trajectory = trajectory_event_ids(offsets)
    trajectory_count = len(offsets) - 1
    keys = (
        "normalized_mutual_information",
        "optimal_match_rate",
        "normalized_h_right_given_left",
        "normalized_h_left_given_right",
    )
    values = {key: [] for key in keys}
    for _ in range(replicates):
        sampled = rng.integers(0, trajectory_count, size=trajectory_count)
        weights_by_trajectory = np.bincount(sampled, minlength=trajectory_count)
        matrix = contingency(left, right, size, weights=weights_by_trajectory[event_trajectory])
        # Expected-MI adjustment is intentionally skipped inside the bootstrap:
        # computing the hypergeometric expectation for every K=256 replicate is
        # prohibitive. Formal AMI is still reported on the full sample; bootstrap
        # intervals cover NMI, optimal matching and both directional entropies.
        row = metrics_from_contingency(matrix, compute_adjusted=False)
        for key in keys:
            values[key].append(float(row[key]))
    return {
        key: {
            "mean": float(np.mean(items)),
            "lower_95": float(np.quantile(items, 0.025)),
            "upper_95": float(np.quantile(items, 0.975)),
        }
        for key, items in values.items()
    }


def _best_group_permutation(indices: np.ndarray, trajectory_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(indices) < 2:
        return indices.copy()
    best = None
    best_same = len(indices) + 1
    for _ in range(20):
        candidate = rng.permutation(indices)
        same = int(np.sum(trajectory_ids[candidate] == trajectory_ids[indices]))
        if same < best_same:
            best = candidate
            best_same = same
        if same == 0:
            break
    assert best is not None
    return best


def grouped_permutation(
    labels: np.ndarray,
    trajectory_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(labels), dtype=np.int64)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    singleton_indices: list[int] = []
    for group in range(len(unique)):
        indices = np.flatnonzero(inverse == group)
        if len(indices) < 2:
            singleton_indices.extend(indices.tolist())
            continue
        permutation[indices] = _best_group_permutation(indices, trajectory_ids, rng)
    if singleton_indices:
        indices = np.asarray(singleton_indices, dtype=np.int64)
        permutation[indices] = _best_group_permutation(indices, trajectory_ids, rng)
    same_trajectory = trajectory_ids[permutation] == trajectory_ids
    return permutation, {
        "groups": int(len(unique)),
        "singleton_events": int(np.sum(counts == 1)),
        "same_trajectory_fraction": float(same_trajectory.mean()),
    }


def global_permutation(trajectory_ids: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    labels = np.zeros(len(trajectory_ids), dtype=np.int8)
    permutation, metadata = grouped_permutation(labels, trajectory_ids, seed)
    return permutation, {"same_trajectory_fraction": float(metadata["same_trajectory_fraction"])}


def event_condition_labels(data_dir: Path, representation_offsets: np.ndarray) -> dict[str, np.ndarray]:
    point_offsets = np.load(data_dir / "test_offsets.npy", mmap_mode="r")
    grids = np.load(data_dir / "test_grid.npy", mmap_mode="r")
    if len(point_offsets) != len(representation_offsets):
        raise ValueError("point and representation trajectory counts differ")
    event_count = int(representation_offsets[-1])
    condition = np.empty(event_count, dtype=np.int64)
    trajectory_ids = trajectory_event_ids(representation_offsets)
    event_positions = np.empty(event_count, dtype=np.int8)
    cursor = 0
    for trajectory_index in range(len(point_offsets) - 1):
        point_start = int(point_offsets[trajectory_index])
        point_end = int(point_offsets[trajectory_index + 1])
        event_start = int(representation_offsets[trajectory_index])
        event_end = int(representation_offsets[trajectory_index + 1])
        event_length = event_end - event_start
        start_grid = int(grids[point_start])
        end_grid = int(grids[point_end - 1])
        start_x, start_y = start_grid % 118, start_grid // 118
        end_x, end_y = end_grid % 118, end_grid // 118
        angle = math.atan2(end_y - start_y, end_x - start_x)
        direction = int(np.floor(((angle + math.pi) / (2 * math.pi)) * 8)) % 8
        if start_grid == end_grid:
            direction = 8
        length = point_end - point_start
        length_bin = int(np.digitize(length, [32, 48, 64, 96]))
        origin_region = (start_y // 20) * 6 + start_x // 20
        for position in range(event_length):
            position_bin = min(3, int(4 * position / max(event_length, 1)))
            condition[cursor] = (((origin_region * 9 + direction) * 5 + length_bin) * 4 + position_bin)
            event_positions[cursor] = position_bin
            cursor += 1
    if cursor != event_count:
        raise AssertionError("event metadata length mismatch")
    return {
        "condition": condition,
        "trajectory_ids": trajectory_ids,
        "position_bin": event_positions,
    }


def linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    codebook_size: int,
    device: str,
    seed: int,
    epochs: int = 5,
    max_train: int = 120_000,
    max_test: int = 120_000,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    train_indices = rng.choice(len(train_labels), size=min(max_train, len(train_labels)), replace=False)
    test_indices = rng.choice(len(test_labels), size=min(max_test, len(test_labels)), replace=False)
    x_train = torch.from_numpy(train_features[train_indices].astype(np.float32, copy=False))
    y_train = torch.from_numpy(train_labels[train_indices].astype(np.int64, copy=False))
    x_test = torch.from_numpy(test_features[test_indices].astype(np.float32, copy=False))
    y_test = torch.from_numpy(test_labels[test_indices].astype(np.int64, copy=False))
    torch.manual_seed(seed)
    target_device = torch.device(device)
    model = torch.nn.Linear(x_train.shape[1], codebook_size).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    batch_size = 2048
    for _ in range(epochs):
        order = torch.randperm(len(y_train), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            features = x_train[index].to(target_device)
            labels = y_train[index].to(target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(features), labels)
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        correct = 0
        for start in range(0, len(y_test), batch_size):
            features = x_test[start : start + batch_size].to(target_device)
            predictions = model(features).argmax(dim=-1).cpu()
            correct += int((predictions == y_test[start : start + batch_size]).sum())
    counts = np.bincount(train_labels.astype(np.int64), minlength=codebook_size)
    majority = float(counts.max() / max(counts.sum(), 1))
    return {
        "accuracy": float(correct / len(y_test)),
        "majority_baseline": majority,
        "accuracy_lift": float(correct / len(y_test) - majority),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "epochs": epochs,
    }


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
