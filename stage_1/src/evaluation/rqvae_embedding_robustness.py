"""Measure cross-sampling robustness of released HTP RQ-VAE embeddings.

This evaluation reuses the trajectory IDs, checkpoints, and seed from an existing
Stage 1 confirmation report. It does not train or modify the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denom, 1e-12)


def normalized_l2_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.linalg.norm(a - b, axis=1)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def load_rows(repo: Path, city: str, trajectory_ids: list[int], seed: int, max_points: int = 256) -> list[dict]:
    frame = pd.read_parquet(
        repo / "traj_data" / city / "traj.parquet",
        columns=["traj_id", "time", "geometry", "opath", "cpath"],
    )
    test_indices = np.load(repo / "traj_data" / city / "test_index.npy")
    frame = frame.iloc[test_indices]
    lookup = {int(row.traj_id): row for row in frame.itertuples(index=False)}
    rng = np.random.default_rng(seed)
    rows = []
    for trajectory_id in trajectory_ids:
        row = lookup[int(trajectory_id)]
        times = np.asarray(row.time, dtype=np.int64)
        points = np.vstack(np.asarray(row.geometry, dtype=object)).astype(np.float64)
        roads = np.asarray(row.opath, dtype=np.int64)
        n = min(len(times), len(points), len(roads))
        if n > max_points:
            start = int(rng.integers(0, n - max_points + 1))
            stop = start + max_points
        else:
            start, stop = 0, n
        rows.append({
            "traj_id": int(trajectory_id),
            "times": times[start:stop],
            "points": points[start:stop],
            "roads": roads[start:stop],
            "primary_road": int(Counter(np.asarray(row.cpath, dtype=np.int64).tolist()).most_common(1)[0][0]),
        })
    return rows


def encode_embeddings(model, sequences: list[np.ndarray], device: str, batch_size: int) -> list[dict]:
    outputs: list[dict] = []
    for start in range(0, len(sequences), batch_size):
        tensors = [torch.as_tensor(value, dtype=torch.float32) for value in sequences[start:start + batch_size]]
        lengths = torch.as_tensor([len(value) for value in tensors], dtype=torch.long, device=device)
        padded = pad_sequence(tensors, batch_first=True).to(device)
        with torch.inference_mode():
            latent, _, _, mask = model.encoder(padded, lengths)
            pre_quant = model.rq_pre_quant_linear(latent[mask])
            quantized, codes, _, _ = model.rq_quant(pre_quant)
        split_lengths = mask.sum(dim=-1).tolist()
        pre_split = torch.split(pre_quant.cpu(), split_lengths)
        quant_split = torch.split(quantized.cpu(), split_lengths)
        code_split = torch.split(codes.cpu(), split_lengths)
        outputs.extend({
            "pre_quant": pre.numpy(),
            "quantized": quant.numpy(),
            "codes": code.numpy(),
        } for pre, quant, code in zip(pre_split, quant_split, code_split))
    return outputs


def aligned_indices(ref_centers: np.ndarray, view_centers: np.ndarray, tolerance: int = 0) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(ref_centers, view_centers)
    positions = np.clip(positions, 0, len(ref_centers) - 1)
    left = np.clip(positions - 1, 0, len(ref_centers) - 1)
    choose_left = np.abs(ref_centers[left] - view_centers) < np.abs(ref_centers[positions] - view_centers)
    positions[choose_left] = left[choose_left]
    valid = np.abs(ref_centers[positions] - view_centers) <= tolerance
    return positions[valid], np.flatnonzero(valid)


def local_report(native: list[dict], view: list[dict], rows: list[dict], representation: str, motion_features) -> dict:
    positives_a, positives_b, negatives_b = [], [], []
    for row, ref, other in zip(rows, native, view):
        ref_indices, view_indices = aligned_indices(ref["centers"], other["centers"])
        if not len(ref_indices):
            continue
        a = ref[representation][ref_indices]
        b = other[representation][view_indices]
        positives_a.append(a)
        positives_b.append(b)

        speed_states = motion_features(row["times"], row["points"])["speed_state"][ref["centers"]]
        negative_indices = []
        kept = []
        for pair_index, ref_index in enumerate(ref_indices):
            candidates = np.flatnonzero(speed_states != speed_states[ref_index])
            if len(candidates):
                negative_indices.append(candidates[(pair_index + ref_index) % len(candidates)])
                kept.append(pair_index)
        if kept:
            negatives_b.append(ref[representation][np.asarray(negative_indices)])

    a = np.concatenate(positives_a)
    b = np.concatenate(positives_b)
    positive_cos = cosine_rows(a, b)
    raw_l2 = np.linalg.norm(a - b, axis=1)
    normalized_l2 = normalized_l2_rows(a, b)

    # Rebuild only positive anchors for which a different-speed negative exists.
    anchor_for_negative, positive_for_negative = [], []
    for row, ref, other in zip(rows, native, view):
        ref_indices, view_indices = aligned_indices(ref["centers"], other["centers"])
        speed_states = motion_features(row["times"], row["points"])["speed_state"][ref["centers"]]
        kept = [i for i, ref_index in enumerate(ref_indices) if np.any(speed_states != speed_states[ref_index])]
        if kept:
            anchor_for_negative.append(ref[representation][ref_indices[np.asarray(kept)]])
            positive_for_negative.append(other[representation][view_indices[np.asarray(kept)]])
    anchor = np.concatenate(anchor_for_negative)
    positive_subset = np.concatenate(positive_for_negative)
    negative = np.concatenate(negatives_b)
    paired_positive_cos = cosine_rows(anchor, positive_subset)
    negative_cos = cosine_rows(anchor, negative)

    return {
        "aligned_positive_cosine": summarize(positive_cos),
        "aligned_positive_raw_l2": summarize(raw_l2),
        "aligned_positive_normalized_l2": summarize(normalized_l2),
        "different_speed_same_sampling_cosine": summarize(negative_cos),
        "selective_cosine_margin_mean": float(np.mean(paired_positive_cos - negative_cos)),
        "selective_pairwise_accuracy": float(np.mean(paired_positive_cos > negative_cos)),
        "selective_pairs": int(len(negative_cos)),
    }


def retrieval_report(native: list[dict], view: list[dict], rows: list[dict], representation: str, motion_features) -> dict:
    gallery = np.stack([value[representation].mean(axis=0) for value in native])
    queries = np.stack([value[representation].mean(axis=0) for value in view])
    gallery /= np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    similarities = queries @ gallery.T
    matched = np.diag(similarities)
    order = np.argsort(-similarities, axis=1)
    ranks = np.asarray([int(np.flatnonzero(order[i] == i)[0]) + 1 for i in range(len(rows))])

    behavior = []
    for row in rows:
        features = motion_features(row["times"], row["points"])
        behavior.append([
            float(np.mean(features["speed"])),
            float(np.mean(features["stop"])),
            float(np.mean(features["turn"])),
        ])
    behavior = np.asarray(behavior)
    behavior = (behavior - behavior.mean(axis=0)) / np.maximum(behavior.std(axis=0), 1e-12)
    hard_negative_cos = []
    hard_negative_matched = []
    for i, row in enumerate(rows):
        candidates = [j for j, other in enumerate(rows) if j != i and other["primary_road"] == row["primary_road"]]
        if not candidates:
            continue
        distances = np.linalg.norm(behavior[candidates] - behavior[i], axis=1)
        selected = candidates[int(np.argmax(distances))]
        hard_negative_cos.append(similarities[i, selected])
        hard_negative_matched.append(matched[i])
    hard_negative_cos = np.asarray(hard_negative_cos)
    hard_negative_matched = np.asarray(hard_negative_matched)

    return {
        "matched_trajectory_cosine": summarize(matched),
        "cross_sampling_retrieval": {
            "queries": int(len(rows)),
            "top1": float(np.mean(ranks <= 1)),
            "top5": float(np.mean(ranks <= 5)),
            "mrr": float(np.mean(1.0 / ranks)),
            "median_rank": float(np.median(ranks)),
            "random_top1": float(1.0 / len(rows)),
            "random_top5": float(min(5.0 / len(rows), 1.0)),
        },
        "same_road_different_behavior_cosine": summarize(hard_negative_cos),
        "matched_minus_hard_negative_margin": float(np.mean(hard_negative_matched - hard_negative_cos)) if len(hard_negative_cos) else None,
        "matched_beats_hard_negative": float(np.mean(hard_negative_matched > hard_negative_cos)) if len(hard_negative_cos) else None,
        "hard_negative_pairs": int(len(hard_negative_cos)),
    }


def evaluate_city(repo: Path, city_record: dict, device: str, batch_size: int) -> dict:
    sys.path.insert(0, str(repo / "stage_1" / "src" / "evaluation"))
    from rqvae_sampling_pilot import load_model, motion_features, normalize, view_indices

    city = city_record["city"]
    seed = int(city_record["seed"])
    rows = load_rows(repo, city, list(map(int, city_record["sampled_trajectory_ids"])), seed)
    model, config = load_model(repo, city, Path(city_record["checkpoint"]), device)
    encoded_views = {}
    for name, factor in (("native", 1), ("downsample_2x", 2), ("downsample_3x", 3)):
        sequences, indices_per_row = [], []
        for row in rows:
            indices = view_indices(len(row["times"]), factor)
            indices_per_row.append(indices)
            sequences.append(normalize(row["points"][indices], config["mbr"]))
        encoded = encode_embeddings(model, sequences, device, batch_size)
        encoded_views[name] = []
        for value, indices in zip(encoded, indices_per_row):
            centers = indices[np.minimum(np.arange(len(value["codes"])) * 8, len(indices) - 1)]
            encoded_views[name].append({**value, "centers": centers})

    views = {}
    for view_name in ("downsample_2x", "downsample_3x"):
        views[view_name] = {}
        for representation in ("pre_quant", "quantized"):
            views[view_name][representation] = {
                "local": local_report(encoded_views["native"], encoded_views[view_name], rows, representation, motion_features),
                "trajectory": retrieval_report(encoded_views["native"], encoded_views[view_name], rows, representation, motion_features),
            }
    return {
        "city": city,
        "seed": seed,
        "sampled_trajectories": len(rows),
        "checkpoint": city_record["checkpoint"],
        "views": views,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--confirmation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    confirmation = json.loads(args.confirmation_report.read_text())
    result = {
        "schema_version": 1,
        "source_confirmation_report": str(args.confirmation_report),
        "cities": [evaluate_city(args.repo, city, args.device, args.batch_size) for city in confirmation["cities"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
