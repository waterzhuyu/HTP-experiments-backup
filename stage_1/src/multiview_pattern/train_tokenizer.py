from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import (
    ENCODER_FAMILIES,
    VIEWS,
    IndependentTokenizer,
    MultiViewTrajectoryDataset,
    codebook_statistics,
    collate_multiview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one fully independent view tokenizer.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--road-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=VIEWS, required=True)
    parser.add_argument("--encoder-family", choices=ENCODER_FAMILIES, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--usage-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--probe-train-trajectories", type=int, default=20_000)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def extract_split(
    model: IndependentTokenizer,
    loader: DataLoader,
    device: torch.device,
    n_grid_x: int,
    n_grid_y: int,
    output_path: Path,
    codebook_size: int,
) -> dict[str, object]:
    model.eval()
    codes: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    offsets = [0]
    traj_ids: list[int] = []
    losses: dict[str, list[float]] = {}
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch, n_grid_x, n_grid_y)
        for name, value in output.losses.items():
            losses.setdefault(name, []).append(float(value))
        for row, traj_id in enumerate(batch["traj_id"].cpu().tolist()):
            mask = output.mask[row]
            row_codes = output.codes[row][mask].cpu().numpy().astype(np.int16, copy=False)
            row_latents = output.latents[row][mask].cpu().numpy().astype(np.float16, copy=False)
            codes.append(row_codes)
            latents.append(row_latents)
            traj_ids.append(int(traj_id))
            offsets.append(offsets[-1] + len(row_codes))
    flat_codes = np.concatenate(codes)
    flat_latents = np.concatenate(latents)
    np.savez_compressed(
        output_path,
        codes=flat_codes,
        latents=flat_latents,
        offsets=np.asarray(offsets, dtype=np.int64),
        traj_ids=np.asarray(traj_ids, dtype=np.int64),
    )
    return {
        "trajectories": len(traj_ids),
        "latent_events": int(len(flat_codes)),
        "losses": {name: float(np.mean(values)) for name, values in losses.items()},
        "codebook": codebook_statistics(flat_codes.astype(np.int64), codebook_size),
        "artifact": str(output_path),
        "artifact_sha256": file_sha256(output_path),
    }


def make_loader(
    dataset: MultiViewTrajectoryDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_multiview,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.data_dir / "metadata.json").read_text())
    road_features = torch.load(args.road_embeddings, map_location="cpu", weights_only=True).float()
    road_features = torch.nn.functional.normalize(road_features, dim=-1)
    if len(road_features) != int(metadata["num_roads"]):
        raise ValueError("road embedding count does not match metadata")
    device = torch.device(args.device)
    model = IndependentTokenizer(
        view=args.view,
        encoder_family=args.encoder_family,
        num_grids=int(metadata["num_grids"]),
        road_features=road_features,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        beta=args.beta,
        dropout=args.dropout,
        usage_weight=args.usage_weight,
    ).to(device)
    train_indices = None if args.train_limit <= 0 else np.arange(args.train_limit, dtype=np.int64)
    test_indices = None if args.test_limit <= 0 else np.arange(args.test_limit, dtype=np.int64)
    train_data = MultiViewTrajectoryDataset(args.data_dir, "train", train_indices)
    test_data = MultiViewTrajectoryDataset(args.data_dir, "test", test_indices)
    probe_rng = np.random.default_rng(20260819)
    probe_count = min(args.probe_train_trajectories, len(train_data))
    probe_indices = np.sort(probe_rng.choice(len(train_data), size=probe_count, replace=False))
    probe_data = MultiViewTrajectoryDataset(args.data_dir, "train", probe_indices)
    train_loader = make_loader(train_data, args.batch_size, args.num_workers, True, args.seed)
    test_loader = make_loader(test_data, args.batch_size, args.num_workers, False, args.seed)
    probe_loader = make_loader(probe_data, args.batch_size, args.num_workers, False, args.seed)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: dict[str, list[float]] = {}
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    batch,
                    int(metadata["grid"]["n_grid_x"]),
                    int(metadata["grid"]["n_grid_y"]),
                )
            scaler.scale(output.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            for name, value in output.losses.items():
                epoch_losses.setdefault(name, []).append(float(value.detach()))
        row = {"epoch": epoch, **{name: float(np.mean(values)) for name, values in epoch_losses.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)

    revision = git_revision()
    resolved_config = {
        **{name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "git_revision": revision,
        "data_metadata": metadata,
        "parameter_count_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "parameter_count_total": sum(parameter.numel() for parameter in model.parameters()),
        "precision": "bf16_autocast_with_grad_scaler" if device.type == "cuda" else "float32",
        "gradient_clip_norm": 1.0,
        "probe_sample_seed": 20260819,
    }
    (args.output_dir / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2))
    checkpoint = {
        "model": model.state_dict(),
        "resolved_config": resolved_config,
        "history": history,
    }
    torch.save(checkpoint, args.output_dir / "model.pt")
    test_metrics = extract_split(
        model,
        test_loader,
        device,
        int(metadata["grid"]["n_grid_x"]),
        int(metadata["grid"]["n_grid_y"]),
        args.output_dir / "test_representation.npz",
        args.codebook_size,
    )
    probe_metrics = extract_split(
        model,
        probe_loader,
        device,
        int(metadata["grid"]["n_grid_x"]),
        int(metadata["grid"]["n_grid_y"]),
        args.output_dir / "train_probe_representation.npz",
        args.codebook_size,
    )
    metrics = {
        "view": args.view,
        "encoder_family": args.encoder_family,
        "seed": args.seed,
        "git_revision": revision,
        "history": history,
        "test": test_metrics,
        "probe_train": probe_metrics,
        "runtime_seconds": time.time() - started,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
