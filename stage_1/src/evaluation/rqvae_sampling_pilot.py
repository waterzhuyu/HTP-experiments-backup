"""Pilot E1-E6 evaluation of released HTP RQ-VAE tokens.

The encoder is evaluated without retraining. Views are generated only by removing or
perturbing observed points; no interpolation is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence


EARTH_RADIUS_M = 6_371_008.8


def comb2(value: np.ndarray | float) -> np.ndarray | float:
    return value * (value - 1) / 2


def contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    _, a = np.unique(labels_a, return_inverse=True, axis=0)
    _, b = np.unique(labels_b, return_inverse=True, axis=0)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    return table


def normalized_mutual_info(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if not len(labels_a):
        return float("nan")
    table = contingency(labels_a, labels_b).astype(np.float64)
    n = table.sum()
    pa = table.sum(axis=1) / n
    pb = table.sum(axis=0) / n
    joint = table / n
    rows, cols = np.nonzero(joint)
    mi = float(np.sum(joint[rows, cols] * np.log(joint[rows, cols] / (pa[rows] * pb[cols]))))
    ha = float(-np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    hb = float(-np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    return mi / max(math.sqrt(ha * hb), 1e-12)


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if len(labels_a) < 2:
        return float("nan")
    table = contingency(labels_a, labels_b)
    n = table.sum()
    sum_cells = float(np.sum(comb2(table)))
    sum_rows = float(np.sum(comb2(table.sum(axis=1))))
    sum_cols = float(np.sum(comb2(table.sum(axis=0))))
    total = float(comb2(n))
    expected = sum_rows * sum_cols / max(total, 1.0)
    maximum = 0.5 * (sum_rows + sum_cols)
    return (sum_cells - expected) / max(maximum - expected, 1e-12)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in np.unique(y_true):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores)) if scores else float("nan")


def majority_probe(train_x, train_y, test_x, test_y) -> dict:
    counts: dict[object, Counter] = defaultdict(Counter)
    for key, label in zip(train_x, train_y):
        counts[key][int(label)] += 1
    default = Counter(map(int, train_y)).most_common(1)[0][0]
    predictions = np.asarray([counts[key].most_common(1)[0][0] if key in counts else default for key in test_x])
    baseline = np.full_like(test_y, default)
    return {
        "macro_f1": macro_f1(test_y, predictions),
        "majority_macro_f1": macro_f1(test_y, baseline),
        "coverage": float(np.mean([key in counts for key in test_x])),
        "samples": len(test_y),
    }


def haversine_m(points: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(points[:, 0].astype(np.float64))
    lat = np.deg2rad(points[:, 1].astype(np.float64))
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def motion_features(times: np.ndarray, points: np.ndarray) -> dict[str, np.ndarray]:
    dt = np.diff(times).astype(np.float64)
    speed_edges = haversine_m(points) / np.maximum(dt, 1)
    speed = np.r_[speed_edges[:1], speed_edges]
    acceleration = np.zeros(len(points), dtype=np.float64)
    if len(speed_edges) > 1:
        acceleration[2:] = np.diff(speed_edges) / np.maximum((dt[:-1] + dt[1:]) / 2, 1)
    lat_scale = 111_195.0
    lon_scale = lat_scale * math.cos(math.radians(float(np.mean(points[:, 1]))))
    displacement = np.diff(points, axis=0) * np.array([lon_scale, lat_scale])
    bearing = np.arctan2(displacement[:, 1], displacement[:, 0])
    turn = np.zeros(len(points), dtype=np.float64)
    if len(bearing) > 1:
        turn_delta = np.arctan2(np.sin(np.diff(bearing)), np.cos(np.diff(bearing)))
        turn[2:] = np.abs(turn_delta)
    speed_state = np.digitize(speed, [1.0, 5.0, 15.0]).astype(np.int64)
    accel_state = np.digitize(acceleration, [-0.3, 0.3]).astype(np.int64)
    return {
        "speed": speed,
        "speed_state": speed_state,
        "accel_state": accel_state,
        "stop": (speed < 1.0).astype(np.int64),
        "turn": (turn > math.radians(30)).astype(np.int64),
    }


def normalize(points: np.ndarray, mbr: dict) -> np.ndarray:
    result = points.copy().astype(np.float32)
    result[:, 0] = (result[:, 0] - mbr["min_lon"]) / (mbr["max_lon"] - mbr["min_lon"])
    result[:, 1] = (result[:, 1] - mbr["min_lat"]) / (mbr["max_lat"] - mbr["min_lat"])
    return result


def encode_sequences(model, sequences: list[np.ndarray], device: str, batch_size: int) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        tensors = [torch.as_tensor(value, dtype=torch.float32) for value in sequences[start:start + batch_size]]
        lengths = torch.as_tensor([len(value) for value in tensors], dtype=torch.long, device=device)
        padded = pad_sequence(tensors, batch_first=True).to(device)
        with torch.inference_mode():
            latent, _, _, mask = model.encoder(padded, lengths)
            valid = model.rq_pre_quant_linear(latent[mask])
            _, codes, _, _ = model.rq_quant(valid)
        split = torch.split(codes.cpu(), mask.sum(dim=-1).tolist())
        outputs.extend(value.numpy() for value in split)
    return outputs


def view_indices(length: int, factor: int, phase: int = 0, missing_rate: float = 0.0, rng=None) -> np.ndarray:
    indices = np.arange(phase, length, factor, dtype=np.int64)
    if missing_rate and len(indices) > 2:
        keep = rng.random(len(indices)) >= missing_rate
        keep[0] = True
        keep[-1] = True
        indices = indices[keep]
    return indices


def add_noise(points: np.ndarray, meters: float, rng: np.random.Generator) -> np.ndarray:
    result = points.copy()
    lat_scale = 111_195.0
    lon_scale = lat_scale * np.cos(np.deg2rad(result[:, 1]))
    result[:, 0] += rng.normal(0, meters, len(result)) / lon_scale
    result[:, 1] += rng.normal(0, meters, len(result)) / lat_scale
    return result


def align_codes(ref: dict, view: dict, max_index_distance: int) -> tuple[np.ndarray, np.ndarray]:
    ref_centers = ref["centers"]
    view_centers = view["centers"]
    positions = np.searchsorted(ref_centers, view_centers)
    positions = np.clip(positions, 0, len(ref_centers) - 1)
    left = np.clip(positions - 1, 0, len(ref_centers) - 1)
    choose_left = np.abs(ref_centers[left] - view_centers) < np.abs(ref_centers[positions] - view_centers)
    positions[choose_left] = left[choose_left]
    valid = np.abs(ref_centers[positions] - view_centers) <= max_index_distance
    return ref["codes"][positions[valid]], view["codes"][valid]


def pair_metrics(ref_codes: list[np.ndarray], view_codes: list[np.ndarray]) -> dict:
    a = np.concatenate(ref_codes)
    b = np.concatenate(view_codes)
    report = {"pairs": len(a), "tuple_agreement": float(np.mean(np.all(a == b, axis=1)))}
    for layer in range(a.shape[1]):
        report[f"layer_{layer + 1}"] = {
            "agreement": float(np.mean(a[:, layer] == b[:, layer])),
            "nmi": normalized_mutual_info(a[:, layer], b[:, layer]),
            "ari": adjusted_rand_index(a[:, layer], b[:, layer]),
        }
    return report


def js_similarity(a: np.ndarray, b: np.ndarray, size: int) -> float:
    pa = np.bincount(a, minlength=size).astype(np.float64) + 1e-9
    pb = np.bincount(b, minlength=size).astype(np.float64) + 1e-9
    pa /= pa.sum()
    pb /= pb.sum()
    mean = (pa + pb) / 2
    js = 0.5 * np.sum(pa * np.log(pa / mean)) + 0.5 * np.sum(pb * np.log(pb / mean))
    return float(1 - js / math.log(2))


def load_model(repo: Path, city: str, checkpoint: Path, device: str):
    sys.path.insert(0, str(repo))
    from model.rqvae import RQVAE

    config = yaml.safe_load((repo / "config" / city / "stage1_config.yaml").read_text())
    road_emb = torch.load(repo / "traj_data" / city / "road_emb_128d.pt", map_location="cpu", weights_only=True)
    model = RQVAE(config, road_emb)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), config


def prepare_rows(repo: Path, city: str, count: int, seed: int, max_points: int, exclude_ids: set[int]) -> list[dict]:
    data_path = repo / "traj_data" / city / "traj.parquet"
    frame = pd.read_parquet(data_path, columns=["traj_id", "time", "geometry", "opath", "cpath"])
    test_indices = np.load(repo / "traj_data" / city / "test_index.npy")
    frame = frame.iloc[test_indices]
    eligible = frame[frame["time"].map(len).between(48, 2048)]
    if exclude_ids:
        eligible = eligible[~eligible["traj_id"].isin(exclude_ids)]
    sample = eligible.sample(n=min(count, len(eligible)), random_state=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for row in sample.itertuples(index=False):
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
            "traj_id": int(row.traj_id),
            "times": times[start:stop],
            "points": points[start:stop],
            "roads": roads[start:stop],
            "primary_road": int(Counter(np.asarray(row.cpath, dtype=np.int64).tolist()).most_common(1)[0][0]),
        })
    return rows


def evaluate_city(repo: Path, city: str, checkpoint: Path, count: int, seed: int, device: str, batch_size: int, exclude_ids: set[int]) -> dict:
    model, config = load_model(repo, city, checkpoint, device)
    rows = prepare_rows(repo, city, count, seed, max_points=256, exclude_ids=exclude_ids)
    rng = np.random.default_rng(seed)
    view_specs = {
        "native": (1, 0, 0.0, 0.0),
        "downsample_2x": (2, 0, 0.0, 0.0),
        "downsample_3x": (3, 0, 0.0, 0.0),
        "phase_shift": (2, 1, 0.0, 0.0),
        "missing_20pct": (1, 0, 0.2, 0.0),
        "noise_10m": (1, 0, 0.0, 10.0),
    }
    encoded: dict[str, list[dict]] = {}
    for name, (factor, phase, missing_rate, noise_m) in view_specs.items():
        sequences = []
        metadata = []
        for row in rows:
            indices = view_indices(len(row["times"]), factor, phase, missing_rate, rng)
            points = row["points"][indices]
            if noise_m:
                points = add_noise(points, noise_m, rng)
            sequences.append(normalize(points, config["mbr"]))
            metadata.append({"indices": indices, "points": points})
        codes = encode_sequences(model, sequences, device, batch_size)
        encoded[name] = []
        for code, meta in zip(codes, metadata):
            centers = meta["indices"][np.minimum(np.arange(len(code)) * 8, len(meta["indices"]) - 1)]
            encoded[name].append({"codes": code, "centers": centers, **meta})

    perturbation_reports = {}
    for name in view_specs:
        if name == "native":
            continue
        ref_pairs, view_pairs = [], []
        tolerance = 4 if name in {"phase_shift", "missing_20pct"} else 0
        for ref, view in zip(encoded["native"], encoded[name]):
            a, b = align_codes(ref, view, tolerance)
            if len(a):
                ref_pairs.append(a)
                view_pairs.append(b)
        perturbation_reports[name] = pair_metrics(ref_pairs, view_pairs)

    rule_reports = {}
    for name in ("downsample_2x", "downsample_3x", "phase_shift"):
        reference_labels, view_labels = [], []
        for row, view in zip(rows, encoded[name]):
            ref_features = motion_features(row["times"], row["points"])
            view_features = motion_features(row["times"][view["indices"]], view["points"])
            reference_labels.append(ref_features["speed_state"][view["indices"]])
            view_labels.append(view_features["speed_state"])
        a = np.concatenate(reference_labels)
        b = np.concatenate(view_labels)
        rule_reports[name] = {
            "agreement": float(np.mean(a == b)),
            "nmi": normalized_mutual_info(a, b),
            "ari": adjusted_rand_index(a, b),
            "pairs": len(a),
        }

    token_records = []
    for trajectory_index, (row, native) in enumerate(zip(rows, encoded["native"])):
        features = motion_features(row["times"], row["points"])
        for code, center in zip(native["codes"], native["centers"]):
            point = row["points"][center]
            grid_x = int(np.clip((point[0] - config["mbr"]["min_lon"]) / (config["mbr"]["max_lon"] - config["mbr"]["min_lon"]) * 8, 0, 7))
            grid_y = int(np.clip((point[1] - config["mbr"]["min_lat"]) / (config["mbr"]["max_lat"] - config["mbr"]["min_lat"]) * 8, 0, 7))
            token_records.append({
                "trajectory_index": trajectory_index,
                "codes": code,
                "speed_state": int(features["speed_state"][center]),
                "accel_state": int(features["accel_state"][center]),
                "stop": int(features["stop"][center]),
                "turn": int(features["turn"][center]),
                "road": int(row["roads"][center]),
                "grid": grid_y * 8 + grid_x,
            })
    split = int(0.7 * len(rows))
    train = [record for record in token_records if record["trajectory_index"] < split]
    test = [record for record in token_records if record["trajectory_index"] >= split]
    probes = {}
    for representation, key_fn in {
        "layer_1": lambda value: int(value[0]),
        "layer_2": lambda value: int(value[1]),
        "tuple": lambda value: tuple(map(int, value)),
    }.items():
        train_x = [key_fn(record["codes"]) for record in train]
        test_x = [key_fn(record["codes"]) for record in test]
        probes[representation] = {}
        for target in ("speed_state", "accel_state", "stop", "turn", "grid"):
            probes[representation][target] = majority_probe(
                train_x,
                np.asarray([record[target] for record in train]),
                test_x,
                np.asarray([record[target] for record in test]),
            )

    top_roads = {road for road, _ in Counter(record["road"] for record in train if record["road"] >= 0).most_common(32)}
    road_train = [record for record in train if record["road"] in top_roads]
    road_test = [record for record in test if record["road"] in top_roads]
    for representation, key_fn in {
        "layer_1": lambda value: int(value[0]),
        "layer_2": lambda value: int(value[1]),
        "tuple": lambda value: tuple(map(int, value)),
    }.items():
        probes[representation]["road_top32"] = majority_probe(
            [key_fn(record["codes"]) for record in road_train],
            np.asarray([record["road"] for record in road_train]),
            [key_fn(record["codes"]) for record in road_test],
            np.asarray([record["road"] for record in road_test]),
        )

    all_layer1 = np.asarray([record["codes"][0] for record in token_records])
    information = {
        target: normalized_mutual_info(all_layer1, np.asarray([record[target] for record in token_records]))
        for target in ("speed_state", "accel_state", "stop", "turn", "road", "grid")
    }

    positive, negative = [], []
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["primary_road"]].append(index)
    for index, row in enumerate(rows):
        positive.append(js_similarity(encoded["native"][index]["codes"][:, 0], encoded["downsample_2x"][index]["codes"][:, 0], 32))
        candidates = groups[row["primary_road"]]
        if len(candidates) > 1:
            other = candidates[(candidates.index(index) + 1) % len(candidates)]
            negative.append(js_similarity(encoded["native"][index]["codes"][:, 0], encoded["native"][other]["codes"][:, 0], 32))

    collapse_path = checkpoint.parent.parent / "codebook_evaluations" / "final_full_test.json"
    collapse = json.loads(collapse_path.read_text()) if collapse_path.exists() else None
    native_interval = int(np.median(np.concatenate([np.diff(row["times"]) for row in rows])))
    return {
        "city": city,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "sampled_trajectories": len(rows),
        "sampled_trajectory_ids": [row["traj_id"] for row in rows],
        "native_interval_seconds": native_interval,
        "token_temporal_support_seconds": native_interval * 8,
        "perturbation_metrics": perturbation_reports,
        "rule_speed_baseline": rule_reports,
        "selective_invariance": {
            "positive_same_motion_2x_mean": float(np.mean(positive)),
            "hard_negative_same_road_mean": float(np.mean(negative)) if negative else None,
            "hard_negative_pairs": len(negative),
        },
        "behavior_and_identity_probes": probes,
        "layer_1_nmi": information,
        "codebook_collapse": {
            "unique_code_tuples": collapse["unique_code_tuples"],
            "layers": [
                {key: layer[key] for key in ("codebook_size", "used_codes", "normalized_perplexity", "max_code_share")}
                for layer in collapse["layers"]
            ],
        } if collapse else None,
        "uncertainty_support": {
            "available": False,
            "reason": "Released RQ-VAE uses deterministic nearest-code assignment and exposes no posterior confidence or rejection score.",
        },
    }


def markdown_report(report: dict) -> str:
    lines = ["# Released HTP Token Sampling Pilot", ""]
    for city in report["cities"]:
        p = city["perturbation_metrics"]
        probes = city["behavior_and_identity_probes"]["layer_1"]
        info = city["layer_1_nmi"]
        lines.extend([
            f"## {city['city'].title()}", "",
            f"- Sample: {city['sampled_trajectories']} held-out trajectories; native cadence {city['native_interval_seconds']} s; one token spans roughly {city['token_temporal_support_seconds']} s before receptive-field effects.",
            f"- 2× downsampling: layer-1 agreement {p['downsample_2x']['layer_1']['agreement']:.3f}, tuple agreement {p['downsample_2x']['tuple_agreement']:.3f}, ARI {p['downsample_2x']['layer_1']['ari']:.3f}.",
            f"- 3× downsampling: layer-1 agreement {p['downsample_3x']['layer_1']['agreement']:.3f}, tuple agreement {p['downsample_3x']['tuple_agreement']:.3f}, ARI {p['downsample_3x']['layer_1']['ari']:.3f}.",
            f"- 20% missing / 10 m noise layer-1 agreement: {p['missing_20pct']['layer_1']['agreement']:.3f} / {p['noise_10m']['layer_1']['agreement']:.3f}.",
            f"- Speed-state probe macro-F1: {probes['speed_state']['macro_f1']:.3f} vs majority {probes['speed_state']['majority_macro_f1']:.3f}; token coverage {probes['speed_state']['coverage']:.3f}.",
            f"- NMI with speed/road/grid: {info['speed_state']:.3f} / {info['road']:.3f} / {info['grid']:.3f}.",
            f"- Same-motion 2× vs same-road hard-negative histogram similarity: {city['selective_invariance']['positive_same_motion_2x_mean']:.3f} vs {city['selective_invariance']['hard_negative_same_road_mean']}.",
            f"- Uncertainty: unavailable in the deterministic released encoder.", "",
        ])
    lines.extend(["## Pilot Verdict", "", report["verdict"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chengdu-checkpoint", type=Path, required=True)
    parser.add_argument("--porto-checkpoint", type=Path, required=True)
    parser.add_argument("--exclude-report", type=Path)
    args = parser.parse_args()

    excluded = {"chengdu": set(), "porto": set()}
    if args.exclude_report:
        previous = json.loads(args.exclude_report.read_text())
        excluded = {
            city["city"]: set(map(int, city.get("sampled_trajectory_ids", [])))
            for city in previous["cities"]
        }
    cities = [
        evaluate_city(args.repo, "chengdu", args.chengdu_checkpoint, args.count, args.seed, args.device, args.batch_size, excluded["chengdu"]),
        evaluate_city(args.repo, "porto", args.porto_checkpoint, args.count, args.seed, args.device, args.batch_size, excluded["porto"]),
    ]
    failed_sampling = any(city["perturbation_metrics"]["downsample_2x"]["layer_1"]["agreement"] < 0.7 for city in cities)
    verdict = (
        "The released HTP tokens are a valid non-collapsed reconstruction baseline but do not pass the Stage 1 semantic-primitive gate. "
        "Sampling consistency is below the provisional 0.70 layer-1 threshold in at least one city, and the model has no uncertainty mechanism. "
        "These results support testing P1–P4, not interpreting the released tokens as motion primitives."
        if failed_sampling else
        "The pilot does not falsify sampling stability at the provisional threshold, but behavior, identity, transfer, and uncertainty gates remain required."
    )
    report = {"schema_version": 1, "cities": cities, "verdict": verdict}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rqvae_sampling_pilot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "rqvae_sampling_pilot.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
