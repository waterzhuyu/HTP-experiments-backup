"""P1 pilot: fixed-time trajectory windows with cross-sampling consistency.

The experiment compares two otherwise identical models:
  P0-time: reconstruction + residual vector quantization.
  P1-time: P0-time + aligned native/2x/3x cosine consistency.

Each example is one physical 120-second interval. Sampling views share the same
interval boundaries, so their embeddings are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


EARTH_RADIUS_M = 6_371_008.8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lonlat_to_local_m(points: np.ndarray) -> np.ndarray:
    lat0 = math.radians(float(points[0, 1]))
    scale_x = math.pi * EARTH_RADIUS_M / 180.0 * math.cos(lat0)
    scale_y = math.pi * EARTH_RADIUS_M / 180.0
    local = (points - points[:1]) * np.asarray([scale_x, scale_y])
    return local.astype(np.float32)


def fixed_time_views(
    times: np.ndarray,
    points: np.ndarray,
    window_seconds: int,
    min_native_points: int = 7,
    min_3x_points: int = 3,
) -> list[dict]:
    """Create aligned native/2x/3x views on shared physical-time boundaries."""
    times = np.asarray(times, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)
    if len(times) != len(points) or len(times) < min_native_points:
        return []
    results = []
    start0 = int(times[0])
    stop_limit = int(times[-1])
    for start in range(start0, stop_limit, window_seconds):
        stop = start + window_seconds
        if stop > stop_limit:
            break
        indices = np.flatnonzero((times >= start) & (times <= stop))
        if len(indices) < min_native_points:
            continue
        native_times = times[indices]
        native_points = points[indices]
        if native_times[-1] - native_times[0] < 0.8 * window_seconds:
            continue
        local = lonlat_to_local_m(native_points)
        views = {}
        valid = True
        for name, factor in (("native", 1), ("downsample_2x", 2), ("downsample_3x", 3)):
            selected = np.arange(0, len(indices), factor, dtype=np.int64)
            if selected[-1] != len(indices) - 1:
                selected = np.r_[selected, len(indices) - 1]
            if name == "downsample_3x" and len(selected) < min_3x_points:
                valid = False
                break
            rel_t = (native_times[selected] - start).astype(np.float32) / window_seconds
            xy = local[selected] / 1000.0
            views[name] = np.column_stack([xy, rel_t]).astype(np.float32)
        if not valid:
            continue

        anchors = np.linspace(0.0, 1.0, 9, dtype=np.float32)
        native_rel_t = (native_times - start).astype(np.float32) / window_seconds
        target_x = np.interp(anchors, native_rel_t, local[:, 0]) / 1000.0
        target_y = np.interp(anchors, native_rel_t, local[:, 1]) / 1000.0
        target = np.column_stack([target_x, target_y]).reshape(-1).astype(np.float32)
        elapsed = np.diff(native_times).astype(np.float64)
        distance = np.linalg.norm(np.diff(local, axis=0), axis=1)
        speeds = distance / np.maximum(elapsed, 1.0)
        mean_speed = float(distance.sum() / max(native_times[-1] - native_times[0], 1))
        stop_fraction = float(np.mean(speeds < 1.0)) if len(speeds) else 1.0
        stop_label = int(stop_fraction >= 0.5)
        results.append({
            "start": start,
            "stop": stop,
            "views": views,
            "target": target,
            "mean_speed": mean_speed,
            "speed_state": int(np.digitize(mean_speed, [1.0, 5.0, 15.0])),
            "stop_label": stop_label,
        })
    return results


def build_examples(
    repo: Path,
    city: str,
    split: str,
    limit: int,
    seed: int,
    window_seconds: int,
) -> list[dict]:
    frame = pd.read_parquet(repo / "traj_data" / city / "traj.parquet", columns=["traj_id", "time", "geometry"])
    split_indices = np.load(repo / "traj_data" / city / f"{split}_index.npy")
    rng = np.random.default_rng(seed)
    order = rng.permutation(split_indices)
    examples = []
    for row_index in order:
        row = frame.iloc[int(row_index)]
        points = np.vstack(np.asarray(row.geometry, dtype=object)).astype(np.float64)
        windows = fixed_time_views(np.asarray(row.time), points, window_seconds)
        for window_index, window in enumerate(windows):
            examples.append({
                **window,
                "example_id": f"{int(row.traj_id)}:{window_index}:{window['start']}",
                "traj_id": int(row.traj_id),
            })
            if len(examples) >= limit:
                return examples
    return examples


class WindowDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def collate(examples: list[dict]) -> dict:
    result = {
        "target": torch.as_tensor(np.stack([x["target"] for x in examples])),
        "speed_state": torch.as_tensor([x["speed_state"] for x in examples], dtype=torch.long),
        "stop": torch.as_tensor([x["stop_label"] for x in examples], dtype=torch.long),
        "example_id": [x["example_id"] for x in examples],
    }
    for view in ("native", "downsample_2x", "downsample_3x"):
        tensors = [torch.as_tensor(x["views"][view]) for x in examples]
        result[view] = pad_sequence(tensors, batch_first=True)
        result[f"{view}_length"] = torch.as_tensor([len(x) for x in tensors], dtype=torch.long)
    return result


class ResidualVQ(nn.Module):
    def __init__(self, dim: int, vocab_size: int = 32, layers: int = 2, beta: float = 0.25):
        super().__init__()
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(vocab_size, dim).uniform_(-1.0 / vocab_size, 1.0 / vocab_size))
            for _ in range(layers)
        ])
        self.beta = beta

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        quantized_parts, codes = [], []
        codebook_loss = x.new_zeros(())
        for codebook in self.codebooks:
            distances = (residual.square().sum(-1, keepdim=True) + codebook.square().sum(-1)[None] - 2 * residual @ codebook.T)
            code = distances.argmin(-1)
            quant = F.embedding(code, codebook)
            codebook_loss = codebook_loss + F.mse_loss(quant, residual.detach()) + self.beta * F.mse_loss(residual, quant.detach())
            quantized_parts.append(quant)
            codes.append(code)
            residual = residual - quant.detach()
        quantized = torch.stack(quantized_parts).sum(0)
        straight_through = x + (quantized - x).detach()
        return straight_through, torch.stack(codes, dim=-1), codebook_loss / len(self.codebooks)


class FixedTimeRQVAE(nn.Module):
    def __init__(self, dim: int = 64, vocab_size: int = 32, codebooks: int = 2):
        super().__init__()
        self.point_projection = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, 4 * dim, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.pool_query = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.pre_quant = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.quantizer = ResidualVQ(dim, vocab_size, codebooks)
        self.decoder = nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, 18))

    def encode(self, points: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = torch.arange(points.shape[1], device=points.device)[None] >= lengths[:, None]
        hidden = self.encoder(self.point_projection(points), src_key_padding_mask=mask)
        logits = hidden @ self.pool_query / math.sqrt(hidden.shape[-1])
        logits = logits.masked_fill(mask, -torch.inf)
        pooled = torch.sum(hidden * torch.softmax(logits, dim=1).unsqueeze(-1), dim=1)
        pre_quant = self.pre_quant(pooled)
        quantized, codes, vq_loss = self.quantizer(pre_quant)
        return pre_quant, quantized, codes, vq_loss

    def forward(self, points: torch.Tensor, lengths: torch.Tensor) -> dict:
        pre, quant, codes, vq_loss = self.encode(points, lengths)
        return {"pre": pre, "quant": quant, "codes": codes, "vq_loss": vq_loss, "reconstruction": self.decoder(quant)}


@dataclass
class TrainConfig:
    city: str
    variant: str
    seed: int
    window_seconds: int
    train_windows: int
    test_windows: int
    epochs: int
    batch_size: int
    learning_rate: float
    consistency_weight: float


def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


def train_model(model, loader, config: TrainConfig, device: str) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    history = []
    model.to(device)
    for epoch in range(config.epochs):
        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "vq": 0.0, "consistency": 0.0, "batches": 0}
        for batch in loader:
            target = batch["target"].to(device)
            outputs = {}
            for view in ("native", "downsample_2x", "downsample_3x"):
                outputs[view] = model(batch[view].to(device), batch[f"{view}_length"].to(device))
            reconstruction = torch.stack([F.mse_loss(value["reconstruction"], target) for value in outputs.values()]).mean()
            vq_loss = torch.stack([value["vq_loss"] for value in outputs.values()]).mean()
            consistency = 0.5 * (
                cosine_loss(outputs["native"]["pre"], outputs["downsample_2x"]["pre"])
                + cosine_loss(outputs["native"]["pre"], outputs["downsample_3x"]["pre"])
            )
            loss = reconstruction + vq_loss + config.consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in (("loss", loss), ("reconstruction", reconstruction), ("vq", vq_loss), ("consistency", consistency)):
                totals[key] += float(value.detach())
            totals["batches"] += 1
        record = {"epoch": epoch + 1, **{key: value / totals["batches"] for key, value in totals.items() if key != "batches"}}
        history.append(record)
        print(json.dumps({"city": config.city, "variant": config.variant, **record}), flush=True)
    return history


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    values = []
    for label in np.unique(y_true):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        values.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(values))


def token_probe(train_codes: np.ndarray, train_y: np.ndarray, test_codes: np.ndarray, test_y: np.ndarray) -> dict:
    mapping = {}
    for code in np.unique(train_codes):
        labels = train_y[train_codes == code]
        mapping[int(code)] = int(np.bincount(labels).argmax())
    default = int(np.bincount(train_y).argmax())
    pred = np.asarray([mapping.get(int(code), default) for code in test_codes])
    baseline = np.full_like(test_y, default)
    return {"macro_f1": macro_f1(test_y, pred), "majority_macro_f1": macro_f1(test_y, baseline)}


def collect(model, loader, device: str) -> dict:
    model.eval()
    result = {view: {key: [] for key in ("pre", "quant", "codes", "reconstruction")} for view in ("native", "downsample_2x", "downsample_3x")}
    result["target"], result["speed_state"], result["stop"] = [], [], []
    with torch.inference_mode():
        for batch in loader:
            result["target"].append(batch["target"].numpy())
            result["speed_state"].append(batch["speed_state"].numpy())
            result["stop"].append(batch["stop"].numpy())
            for view in ("native", "downsample_2x", "downsample_3x"):
                output = model(batch[view].to(device), batch[f"{view}_length"].to(device))
                for key in result[view]:
                    result[view][key].append(output[key].cpu().numpy())
    for key in ("target", "speed_state", "stop"):
        result[key] = np.concatenate(result[key])
    for view in ("native", "downsample_2x", "downsample_3x"):
        for key in result[view]:
            result[view][key] = np.concatenate(result[view][key])
    return result


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def representation_metrics(data: dict, view: str, key: str) -> dict:
    native = data["native"][key]
    sampled = data[view][key]
    positive = cosine_rows(native, sampled)
    normalized_l2 = np.linalg.norm(
        native / np.maximum(np.linalg.norm(native, axis=1, keepdims=True), 1e-12)
        - sampled / np.maximum(np.linalg.norm(sampled, axis=1, keepdims=True), 1e-12), axis=1
    )
    normalized_native = native / np.maximum(np.linalg.norm(native, axis=1, keepdims=True), 1e-12)
    normalized_sampled = sampled / np.maximum(np.linalg.norm(sampled, axis=1, keepdims=True), 1e-12)
    similarities = normalized_sampled @ normalized_native.T
    order = np.argsort(-similarities, axis=1)
    ranks = np.asarray([np.flatnonzero(order[i] == i)[0] + 1 for i in range(len(native))])
    negative_indices = []
    valid = []
    for index, state in enumerate(data["speed_state"]):
        candidates = np.flatnonzero(data["speed_state"] != state)
        if len(candidates):
            negative_indices.append(candidates[index % len(candidates)])
            valid.append(index)
    negative = cosine_rows(native[np.asarray(valid)], native[np.asarray(negative_indices)])
    positive_subset = positive[np.asarray(valid)]
    return {
        "cosine_mean": float(positive.mean()),
        "cosine_median": float(np.median(positive)),
        "normalized_l2_mean": float(normalized_l2.mean()),
        "retrieval_top1": float(np.mean(ranks <= 1)),
        "retrieval_top5": float(np.mean(ranks <= 5)),
        "retrieval_mrr": float(np.mean(1.0 / ranks)),
        "random_top1": float(1.0 / len(native)),
        "different_speed_cosine_mean": float(negative.mean()),
        "selective_margin_mean": float(np.mean(positive_subset - negative)),
        "selective_pairwise_accuracy": float(np.mean(positive_subset > negative)),
    }


def code_metrics(codes: np.ndarray, vocab_size: int = 32) -> dict:
    layers = []
    for layer in range(codes.shape[1]):
        counts = np.bincount(codes[:, layer], minlength=vocab_size).astype(np.float64)
        probs = counts / max(counts.sum(), 1)
        nonzero = probs > 0
        perplexity = math.exp(float(-np.sum(probs[nonzero] * np.log(probs[nonzero]))))
        layers.append({
            "used_codes": int(nonzero.sum()),
            "normalized_perplexity": float(perplexity / vocab_size),
            "max_code_share": float(probs.max()),
        })
    return {"layers": layers}


def evaluate(model, train_loader, test_loader, device: str) -> dict:
    train = collect(model, train_loader, device)
    test = collect(model, test_loader, device)
    report = {
        "native_reconstruction_mse": float(np.mean((test["native"]["reconstruction"] - test["target"]) ** 2)),
        "codebook": code_metrics(test["native"]["codes"]),
        "speed_probe": token_probe(train["native"]["codes"][:, 0], train["speed_state"], test["native"]["codes"][:, 0], test["speed_state"]),
        "stop_probe": token_probe(train["native"]["codes"][:, 0], train["stop"], test["native"]["codes"][:, 0], test["stop"]),
        "views": {},
    }
    for view in ("downsample_2x", "downsample_3x"):
        report["views"][view] = {
            "pre_quant": representation_metrics(test, view, "pre"),
            "quantized": representation_metrics(test, view, "quant"),
            "layer_1_agreement": float(np.mean(test["native"]["codes"][:, 0] == test[view]["codes"][:, 0])),
            "tuple_agreement": float(np.mean(np.all(test["native"]["codes"] == test[view]["codes"], axis=1))),
            "reconstruction_mse": float(np.mean((test[view]["reconstruction"] - test["target"]) ** 2)),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--city", choices=("chengdu", "porto"), required=True)
    parser.add_argument("--variant", choices=("p0_time", "p1_time"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--window-seconds", type=int, default=120)
    parser.add_argument("--train-windows", type=int, default=8192)
    parser.add_argument("--test-windows", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    consistency_weight = 0.0 if args.variant == "p0_time" else args.consistency_weight
    config = TrainConfig(
        city=args.city, variant=args.variant, seed=args.seed, window_seconds=args.window_seconds,
        train_windows=args.train_windows, test_windows=args.test_windows, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, consistency_weight=consistency_weight,
    )
    seed_everything(args.seed)
    train_examples = build_examples(args.repo, args.city, "train", args.train_windows, args.seed, args.window_seconds)
    test_examples = build_examples(args.repo, args.city, "test", args.test_windows, args.seed + 1, args.window_seconds)
    if len(train_examples) < args.train_windows or len(test_examples) < args.test_windows:
        raise RuntimeError(f"insufficient windows: train={len(train_examples)}, test={len(test_examples)}")
    train_loader = DataLoader(WindowDataset(train_examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    train_eval_loader = DataLoader(WindowDataset(train_examples), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(WindowDataset(test_examples), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = FixedTimeRQVAE()
    history = train_model(model, train_loader, config, args.device)
    metrics = evaluate(model, train_eval_loader, test_loader, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.city}_{args.variant}_seed{args.seed}.pt"
    report_path = args.output_dir / f"{args.city}_{args.variant}_seed{args.seed}.json"
    torch.save({"model": model.cpu().state_dict(), "config": asdict(config)}, checkpoint_path)
    report = {
        "schema_version": 1,
        "status": "pilot",
        "config": asdict(config),
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
