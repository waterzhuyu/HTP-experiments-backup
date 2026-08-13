"""Fine-tune the released fixed-8-point HTP tokenizer with multi-view training.

This keeps the original HTP encoder/decoder/RQ architecture. Native, 2x, and
3x point-removal views are each reconstructed with the original Stage-1 loss,
while pre-quant embeddings at identical original-point anchors receive a cosine
consistency loss. No interpolation is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def view_indices(length: int, factor: int) -> np.ndarray:
    """Match the released-token evaluation: remove points without interpolation."""
    return np.arange(0, length, factor, dtype=np.int64)


def make_view(record: dict, factor: int) -> tuple[dict, np.ndarray]:
    indices = view_indices(len(record["gps_seq"]), factor)
    result = dict(record)
    for key in ("gps_seq", "time_seq", "dxy", "percent_dist"):
        result[key] = np.asarray(record[key])[indices].copy()
    result["path"] = np.asarray(record["path"]).copy()
    return result, indices


def token_centers(indices: np.ndarray, token_count: int) -> np.ndarray:
    positions = np.minimum(np.arange(token_count, dtype=np.int64) * 8, len(indices) - 1)
    return indices[positions]


def aligned_cosine_loss(
    native: list[torch.Tensor],
    native_indices: list[np.ndarray],
    view: list[torch.Tensor],
    view_indices_per_item: list[np.ndarray],
) -> torch.Tensor:
    anchors, positives = [], []
    for native_value, native_idx, view_value, view_idx in zip(
        native, native_indices, view, view_indices_per_item
    ):
        native_centers = token_centers(native_idx, len(native_value))
        view_centers = token_centers(view_idx, len(view_value))
        _, native_positions, view_positions = np.intersect1d(
            native_centers, view_centers, return_indices=True
        )
        if len(native_positions):
            anchors.append(native_value[torch.as_tensor(native_positions, device=native_value.device)])
            positives.append(view_value[torch.as_tensor(view_positions, device=view_value.device)])
    if not anchors:
        raise RuntimeError("no exactly aligned fixed-8 token anchors in batch")
    a = torch.cat(anchors)
    b = torch.cat(positives)
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


class MultiViewDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


def load_road_data(dataset_root: Path, city: str, mbr: dict):
    road_info = pd.read_csv(dataset_root / city / "rn" / "edge_info.csv")
    road_geo_list, road_geo_lens = [], []
    for value in road_info["geometry"]:
        geo = np.array(list(shapely.from_wkt(value).coords), dtype=np.float64)
        geo[:, 0] = (geo[:, 0] - mbr["min_lon"]) / (mbr["max_lon"] - mbr["min_lon"])
        geo[:, 1] = (geo[:, 1] - mbr["min_lat"]) / (mbr["max_lat"] - mbr["min_lat"])
        road_geo_lens.append(len(geo))
        road_geo_list.append(torch.as_tensor(geo, dtype=torch.float32))
    return (
        pad_sequence(road_geo_list, batch_first=True, padding_value=0.0),
        torch.as_tensor(road_geo_lens, dtype=torch.long),
        json.loads((dataset_root / city / "z_score.json").read_text()),
    )


def build_collate(repo: Path, dataset_root: Path, city: str, config: dict):
    sys.path.insert(0, str(repo))
    from dataloader_stage1 import batch_loader

    road_geo, road_geo_lens, z_score = load_road_data(dataset_root, city, config["mbr"])

    def collate(records: list[dict]) -> dict:
        result = {}
        for name, factor in (("native", 1), ("downsample_2x", 2), ("downsample_3x", 3)):
            views, indices = zip(*(make_view(record, factor) for record in records))
            items = [(view, len(view["gps_seq"]), len(view["path"])) for view in views]
            result[name] = batch_loader(items, config["mbr"], z_score, road_geo, road_geo_lens)
            result[name]["source_indices"] = list(indices)
        return result

    return collate


def load_training_records(dataset_root: Path, city: str, count: int, seed: int) -> list[dict]:
    path = dataset_root / city / "processed_train_traj.pkl"
    with path.open("rb") as handle:
        all_records = pickle.load(handle)
    eligible = np.asarray([i for i, value in enumerate(all_records) if len(value["gps_seq"]) >= 48])
    if len(eligible) < count:
        raise RuntimeError(f"only {len(eligible)} eligible training trajectories")
    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible, size=count, replace=False)
    records = [all_records[int(i)] for i in selected]
    del all_records
    return records


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def forward_with_prequant(model, batch: dict) -> tuple[dict, list[torch.Tensor]]:
    captured = []
    handle = model.rq_pre_quant_linear.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        result = model(
            batch["x"], batch["traj_len"], batch["road_seq"], batch["road_gps"],
            batch["road_gps_len"], batch["dxy"], batch["road_percent"],
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one pre-quant capture, got {len(captured)}")
    token_lengths = (result["code"][..., 0] >= 0).sum(dim=-1).tolist()
    return result, list(torch.split(captured[0], token_lengths))


@dataclass
class ExperimentConfig:
    city: str
    seed: int
    train_examples: int
    epochs: int
    batch_size: int
    learning_rate: float
    consistency_weight: float
    checkpoint: str


def train(model, loader, cfg: ExperimentConfig, device: str) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        totals = {key: 0.0 for key in ("loss", "reconstruction", "quantization", "consistency")}
        batches = 0
        for batch_index, multi_batch in enumerate(loader):
            outputs, embeddings = {}, {}
            for name in ("native", "downsample_2x", "downsample_3x"):
                current = move_batch(multi_batch[name], device)
                outputs[name], embeddings[name] = forward_with_prequant(model, current)
            reconstruction = torch.stack([value["recon_loss"] for value in outputs.values()]).mean()
            quantization = torch.stack([value["quant_loss"] for value in outputs.values()]).mean()
            consistency = 0.5 * (
                aligned_cosine_loss(
                    embeddings["native"], multi_batch["native"]["source_indices"],
                    embeddings["downsample_2x"], multi_batch["downsample_2x"]["source_indices"],
                )
                + aligned_cosine_loss(
                    embeddings["native"], multi_batch["native"]["source_indices"],
                    embeddings["downsample_3x"], multi_batch["downsample_3x"]["source_indices"],
                )
            )
            loss = reconstruction + quantization + cfg.consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in (
                ("loss", loss), ("reconstruction", reconstruction),
                ("quantization", quantization), ("consistency", consistency),
            ):
                totals[key] += float(value.detach())
            batches += 1
            if batch_index % 25 == 0:
                print(json.dumps({
                    "city": cfg.city, "epoch": epoch + 1, "batch": batch_index,
                    "loss": float(loss.detach()), "reconstruction": float(reconstruction.detach()),
                    "quantization": float(quantization.detach()), "consistency": float(consistency.detach()),
                }), flush=True)
        record = {"epoch": epoch + 1, **{key: value / batches for key, value in totals.items()}}
        history.append(record)
        print(json.dumps({"city": cfg.city, "epoch_summary": record}), flush=True)
    return history


def codebook_metrics(repo: Path, city_record: dict, checkpoint: Path, device: str, batch_size: int) -> dict:
    sys.path.insert(0, str(repo / "stage_1" / "src" / "evaluation"))
    from rqvae_sampling_pilot import encode_sequences, load_model, normalize, prepare_rows

    exploration = json.loads((repo / "stage_1/reports/rqvae_sampling_pilot_256_seed42.json").read_text())
    exploration_ids = next(
        set(map(int, value["sampled_trajectory_ids"]))
        for value in exploration["cities"] if value["city"] == city_record["city"]
    )
    rows = prepare_rows(
        repo, city_record["city"], len(city_record["sampled_trajectory_ids"]),
        int(city_record["seed"]), 256, exploration_ids,
    )
    if [row["traj_id"] for row in rows] != list(map(int, city_record["sampled_trajectory_ids"])):
        raise RuntimeError("confirmation sample could not be reproduced exactly")
    model, config = load_model(repo, city_record["city"], checkpoint, device)
    sequences = [normalize(row["points"], config["mbr"]) for row in rows]
    codes = np.concatenate(encode_sequences(model, sequences, device, batch_size))
    layers = []
    sizes = [32, 64, 128, 256]
    for layer, size in enumerate(sizes):
        counts = np.bincount(codes[:, layer], minlength=size).astype(np.float64)
        probs = counts / counts.sum()
        nonzero = probs > 0
        perplexity = math.exp(float(-np.sum(probs[nonzero] * np.log(probs[nonzero]))))
        layers.append({
            "codebook_size": size,
            "used_codes": int(nonzero.sum()),
            "normalized_perplexity": perplexity / size,
            "max_code_share": float(probs.max()),
        })
    return {"tokens": int(len(codes)), "layers": layers}


def evaluate(repo: Path, city: str, checkpoint: Path, device: str, batch_size: int) -> tuple[dict, dict]:
    sys.path.insert(0, str(repo / "stage_1" / "src" / "evaluation"))
    from rqvae_sampling_pilot import evaluate_city

    confirmation = json.loads((repo / "stage_1/reports/rqvae_sampling_confirmation_1024_seed314159.json").read_text())
    city_record = next(value for value in confirmation["cities"] if value["city"] == city)
    exploration = json.loads((repo / "stage_1/reports/rqvae_sampling_pilot_256_seed42.json").read_text())
    excluded = next(
        set(map(int, value["sampled_trajectory_ids"]))
        for value in exploration["cities"] if value["city"] == city
    )
    post = evaluate_city(
        repo, city, checkpoint, len(city_record["sampled_trajectory_ids"]),
        int(city_record["seed"]), device, batch_size, excluded,
    )
    return city_record, {
        "agreement": post["perturbation_metrics"],
        "selective_invariance": post["selective_invariance"],
        "behavior_and_identity_probes": post["behavior_and_identity_probes"],
        "layer_1_nmi": post["layer_1_nmi"],
        "codebook": codebook_metrics(repo, city_record, checkpoint, device, batch_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-root", type=Path, default=Path("/workspace/guozuyu/HTP-main/traj_dataset"))
    parser.add_argument("--city", choices=("chengdu", "porto"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--train-examples", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    sys.path.insert(0, str(args.repo))
    from model.rqvae import RQVAE

    config = yaml.safe_load((args.repo / "config" / args.city / "stage1_config.yaml").read_text())
    road_emb = torch.load(args.repo / "traj_data" / args.city / "road_emb_128d.pt", map_location="cpu", weights_only=True)
    model = RQVAE(config, road_emb)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    records = load_training_records(args.dataset_root, args.city, args.train_examples, args.seed)
    loader = DataLoader(
        MultiViewDataset(records), batch_size=args.batch_size, shuffle=True,
        collate_fn=build_collate(args.repo, args.dataset_root, args.city, config),
        generator=torch.Generator().manual_seed(args.seed),
    )
    cfg = ExperimentConfig(
        city=args.city, seed=args.seed, train_examples=args.train_examples,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        consistency_weight=args.consistency_weight, checkpoint=str(args.checkpoint),
    )
    history = train(model, loader, cfg, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.city}_fixed8_multiview_seed{args.seed}.pt"
    torch.save(model.cpu().state_dict(), checkpoint_path)
    if args.skip_evaluation:
        report_path = args.output_dir / f"{args.city}_fixed8_multiview_seed{args.seed}.json"
        report_path.write_text(json.dumps({
            "schema_version": 1, "status": "smoke", "config": asdict(cfg),
            "checkpoint": str(checkpoint_path), "history": history,
        }, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(report_path), "status": "smoke"}), flush=True)
        return
    baseline, post = evaluate(args.repo, args.city, checkpoint_path, args.device, max(args.batch_size, 64))
    report = {
        "schema_version": 1,
        "status": "pilot",
        "method": "released fixed-8 HTP RQ-VAE fine-tuned with per-view reconstruction/quantization plus exact-anchor pre-quant cosine consistency",
        "config": asdict(cfg),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "baseline": {
            "checkpoint": baseline["checkpoint"],
            "agreement": baseline["perturbation_metrics"],
            "selective_invariance": baseline["selective_invariance"],
            "behavior_and_identity_probes": baseline["behavior_and_identity_probes"],
            "layer_1_nmi": baseline["layer_1_nmi"],
            "codebook": baseline["codebook_collapse"],
        },
        "multiview": post,
    }
    report_path = args.output_dir / f"{args.city}_fixed8_multiview_seed{args.seed}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "baseline_2x": baseline["perturbation_metrics"]["downsample_2x"], "multiview_2x": post["agreement"]["downsample_2x"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
