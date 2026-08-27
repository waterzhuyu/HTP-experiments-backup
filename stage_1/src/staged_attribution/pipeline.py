from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import torch
import yaml
from torch.utils.data import DataLoader

from .config import Budget, ExperimentConfig
from .data import AlignedHTPDataset, RoadContext, load_road_context, make_collator
from .decode_stage3 import decode_variant_records, reconstruct_gps
from .evaluate_stage1 import codebook_statistics, prefix_conditional_entropy, sampling_robustness
from .export_tokens import encode_loader, export_token_records
from .g_attribution import build_g_variants
from .generate_stage3 import run_hf_generation
from .losses import derive_auxiliary_targets
from .model import AttributionRQVAE
from .prepare_llm_data import build_llm_frame, deterministic_subset
from .run_cell import CellRunError, StageNode
from .stage4_metrics import evaluate_population
from .stage2_subset_manifest import (
    file_sha256 as subset_manifest_sha256,
    load_stage2_subset_manifest,
    trajectory_id_sha256,
)
from .train_stage1 import build_cosine_warmup_scheduler, save_stage1_artifacts, train_epochs
from .train_stage2 import run_hf_stage2
from .views import make_sampling_view


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def select_dataset_indices(
    traj_ids: Sequence[int] | np.ndarray,
    requested: int | str,
    seed: int,
    salt: str,
) -> np.ndarray:
    population = [int(value) for value in traj_ids]
    if requested == "full":
        return np.arange(len(population), dtype=np.int64)
    if not isinstance(requested, int) or requested <= 0 or requested > len(population):
        raise CellRunError(f"invalid requested population {requested!r} for {len(population)} IDs")
    ranked = sorted(
        enumerate(population),
        key=lambda pair: (hashlib.sha256(f"{seed}:{salt}:{pair[1]}".encode()).digest(), pair[1]),
    )[:requested]
    return np.asarray(sorted(index for index, _ in ranked), dtype=np.int64)


def select_stage2_train_frame(
    frame: pd.DataFrame,
    requested: int | str,
    subset_seed: int,
    city: str,
    manifest_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if "traj_id" not in frame or frame["traj_id"].duplicated().any():
        raise CellRunError("Stage2 train frame requires unique traj_id values")
    source_ids = [int(value) for value in frame["traj_id"]]
    manifest_audit: dict[str, object] = {}
    if manifest_path is not None:
        if not isinstance(requested, int):
            raise CellRunError("Stage2 subset manifest requires a finite requested population")
        selected_ids, _ = load_stage2_subset_manifest(
            manifest_path,
            source_ids,
            city,
            requested,
            subset_seed,
        )
        row_by_id = {int(traj_id): index for index, traj_id in enumerate(frame["traj_id"])}
        selected = frame.iloc[[row_by_id[traj_id] for traj_id in selected_ids]].reset_index(
            drop=True
        )
        method = "frozen_manifest_sha256_rank_fixed_city_v1"
        recorded_ids = selected_ids
        resolved_manifest_path = Path(manifest_path).resolve()
        manifest_audit = {
            "manifest_path": str(resolved_manifest_path),
            "manifest_sha256": subset_manifest_sha256(resolved_manifest_path),
        }
    elif requested == "full":
        selected = frame
        method = "full_population_v1"
        recorded_ids: list[int] | None = None
    else:
        indices = select_dataset_indices(
            source_ids,
            requested,
            seed=subset_seed,
            salt=f"stage2_train:{city}",
        )
        selected = frame.iloc[indices].sort_values("traj_id").reset_index(drop=True)
        method = "sha256_rank_fixed_city_v1"
        recorded_ids = [int(value) for value in selected["traj_id"]]
    selected_ids = [int(value) for value in selected["traj_id"]]
    return selected, {
        "method": method,
        "source_population": len(frame),
        "population": len(selected),
        "requested": requested,
        "subset_seed": int(subset_seed),
        "city": city,
        "selected_traj_id_sha256": trajectory_id_sha256(selected_ids),
        "selected_traj_ids": recorded_ids,
        "selection_depends_on_training_seed": False,
        **manifest_audit,
    }


def meters_per_normalized_coordinate(metadata: Mapping[str, object]) -> tuple[float, float]:
    mbr = metadata["mbr"]
    grid = metadata["grid"]
    return (
        (float(mbr["max_lon"]) - float(mbr["min_lon"])) * float(grid["meters_per_lon"]),
        (float(mbr["max_lat"]) - float(mbr["min_lat"])) * float(grid["meters_per_lat"]),
    )


def _event_edges(
    dataset: AlignedHTPDataset,
    meters_per_coordinate: tuple[float, float],
    event_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    speed: list[float] = []
    turn: list[float] = []
    scale = np.asarray(meters_per_coordinate, dtype=np.float64)
    for item in dataset:
        gps = np.asarray(item["gps"], dtype=np.float64)
        time = np.asarray(item["time"], dtype=np.float64)
        for start in range(0, len(gps), event_size):
            event_gps = gps[start : start + event_size]
            event_time = time[start : start + event_size]
            if len(event_gps) >= 2:
                distances = np.linalg.norm(np.diff(event_gps, axis=0) * scale, axis=1)
                delta = np.diff(event_time)
                valid = delta > 1e-6
                speed.append(float(np.mean(distances[valid] / delta[valid])) if valid.any() else 0.0)
            else:
                speed.append(0.0)
            if len(event_gps) >= 3:
                vectors = np.diff(event_gps, axis=0)
                denominator = np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(vectors[1:], axis=1)
                valid = denominator > 1e-12
                if valid.any():
                    cosine = np.sum(vectors[:-1][valid] * vectors[1:][valid], axis=1) / denominator[valid]
                    turn.append(float(np.mean(np.arccos(np.clip(cosine, -1.0, 1.0)))))
                else:
                    turn.append(0.0)
            else:
                turn.append(0.0)
    return (
        torch.tensor(np.quantile(speed, [0.25, 0.5, 0.75]), dtype=torch.float32),
        torch.tensor(np.quantile(turn, [1 / 3, 2 / 3]), dtype=torch.float32),
    )


class PipelineBackend:
    """Artifact-oriented implementation of the complete Stage1→2→3/4 cell DAG."""

    def __init__(self, config_path: str | Path, resolved: Mapping[str, object]) -> None:
        self.config_path = Path(config_path)
        self.config = ExperimentConfig.load(config_path)
        self.resolved = dict(resolved)
        self.city = str(resolved["city"])
        self.seed = int(resolved["seed"])
        self.cell_id = str(resolved["cell_id"])
        self.budget = self.config.budgets[str(resolved["budget"])]
        self.cell = {cell.cell_id: cell for cell in self.config.core_cells()}[self.cell_id]
        requested_device = str(resolved.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise CellRunError("CUDA was requested but is unavailable")
        self.device = torch.device(requested_device)
        self.data_root = self.config.paths["data_root"] / self.city
        self.aligned_root = self.config.paths[f"{self.city}_aligned_root"]
        self._datasets: dict[tuple[str, str], AlignedHTPDataset] = {}
        self._road_context: RoadContext | None = None

    def execute(
        self, node: StageNode, resolved: Mapping[str, object], run_dir: Path
    ) -> Mapping[str, Path]:
        del resolved
        handler = getattr(self, f"_{node.name}", None)
        if handler is None:
            raise CellRunError(f"pipeline backend has no handler for {node.name}")
        return handler(run_dir)

    def _processed_path(self, split: str) -> Path:
        return self.data_root / f"processed_{split}_traj.pkl"

    def _requested(self, split: str) -> int | str:
        return (
            self.budget.stage1_train_trajectories
            if split == "train"
            else self.budget.stage1_test_trajectories
        )

    def _dataset(self, split: str) -> AlignedHTPDataset:
        key = (split, str(self._requested(split)))
        if key not in self._datasets:
            base = AlignedHTPDataset(
                self.aligned_root, self._processed_path(split), split
            )
            indices = select_dataset_indices(
                base.traj_ids,
                self._requested(split),
                self.seed,
                f"{self.city}:{split}:{self.resolved['budget']}",
            )
            base.indices = indices
            self._datasets[key] = base
        return self._datasets[key]

    def _context(self) -> RoadContext:
        if self._road_context is None:
            metadata = json.loads((self.aligned_root / "metadata.json").read_text())
            self._road_context = load_road_context(
                self.data_root / "rn" / "edge_info.csv",
                metadata["mbr"],
                self.data_root / "z_score.json",
            )
        return self._road_context

    def _loader(self, split: str, shuffle: bool = False) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self._dataset(split),
            batch_size=self.budget.stage1_batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=0,
            collate_fn=make_collator(self._context()),
        )

    def _model_config(self) -> dict[str, object]:
        path = Path("config") / self.city / "stage1_config.yaml"
        with path.open(encoding="utf-8") as handle:
            return yaml.load(handle, Loader=yaml.FullLoader)

    def _new_model(self) -> AttributionRQVAE:
        _set_seed(self.seed)
        road_features = torch.load(self.data_root / "road_emb_128d.pt", weights_only=True)
        metadata = json.loads((self.aligned_root / "metadata.json").read_text())
        return AttributionRQVAE(
            self._model_config(),
            road_features,
            num_grids=int(metadata["num_grids"]),
            active_views=self.cell.views,
            q_weights=self.cell.q_weights,
        )

    def _load_model(self, run_dir: Path) -> AttributionRQVAE:
        model = self._new_model()
        checkpoint = torch.load(run_dir / "stage1" / "stage1_model.pt", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        return model

    def _auxiliary_settings(
        self, train_dataset: AlignedHTPDataset
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float]]:
        meters = meters_per_normalized_coordinate(train_dataset.metadata)
        if float(self.cell.q_weights["auxiliary"]) > 0:
            speed_edges, turn_edges = _event_edges(train_dataset, meters)
        else:
            speed_edges = torch.tensor([2.0, 5.0, 12.0])
            turn_edges = torch.tensor([0.25, 0.75])
        return speed_edges, turn_edges, meters

    @torch.no_grad()
    def _evaluate_stage1_model(
        self,
        model: AttributionRQVAE,
        loader: DataLoader,
        speed_edges: torch.Tensor,
        turn_edges: torch.Tensor,
        meters: tuple[float, float],
    ) -> dict[str, object]:
        model = model.to(self.device)
        model.eval()
        loss_rows: list[dict[str, float]] = []
        code_rows: list[torch.Tensor] = []
        sampled_rows: list[torch.Tensor] = []
        consistency = any(
            float(self.cell.q_weights[name]) > 0
            for name in ("invariance", "variance", "covariance")
        )
        auxiliary = float(self.cell.q_weights["auxiliary"]) > 0
        for step, batch in enumerate(loader):
            batch = {name: value.to(self.device) for name, value in batch.items()}
            sampling = make_sampling_view(batch, self.seed + 1_000_000 + step) if consistency else None
            targets = (
                derive_auxiliary_targets(batch, speed_edges, turn_edges, meters)
                if auxiliary
                else None
            )
            output = model(batch, sampling_batch=sampling, auxiliary_targets=targets)
            loss_rows.append(
                {
                    "loss": float(output["loss"]),
                    "base_loss": float(output["base_loss"]),
                    "reconstruction_loss": float(output["reconstruction_loss"]),
                    "quant_loss": float(output["quant_loss"]),
                }
            )
            code_rows.append(output["codes"][output["latent_mask"]].cpu())
            if sampling is not None:
                sampled_rows.append(
                    output["sampling_output"]["codes"][output["latent_mask"]].cpu()
                )
        codes = torch.cat(code_rows).unsqueeze(0)
        mask = torch.ones(codes.shape[:2], dtype=torch.bool)
        layer_statistics = codebook_statistics(codes, mask, self.config.rq_vocab_sizes)
        metrics: dict[str, object] = {
            "population": len(loader.dataset),
            "losses": {
                name: float(np.mean([row[name] for row in loss_rows]))
                for name in loss_rows[0]
            },
            "codebook": layer_statistics,
            "prefix_conditional_entropy": prefix_conditional_entropy(
                codes, mask, self.config.rq_vocab_sizes
            ),
            "gate": {"status": "pending_e0_oracle_ratio", "passed": None, "reasons": []},
        }
        if sampled_rows:
            sampled = torch.cat(sampled_rows).unsqueeze(0)
            metrics["sampling_robustness"] = sampling_robustness(codes, sampled, mask)
        return metrics

    def _stage1(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "stage1"
        train_dataset = self._dataset("train")
        speed_edges, turn_edges, meters = self._auxiliary_settings(train_dataset)
        model = self._new_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        train_loader = self._loader("train", shuffle=True)
        scheduler = build_cosine_warmup_scheduler(
            optimizer, len(train_loader), self.budget.stage1_epochs,
            warmup_epochs=10, warmup_ratio=0.01, minimum_ratio=0.10,
        )
        history = train_epochs(
            model,
            train_loader,
            optimizer,
            self.device,
            self.budget.stage1_epochs,
            self.seed,
            speed_edges,
            turn_edges,
            meters,
            scheduler=scheduler,
        )
        metrics = self._evaluate_stage1_model(
            model, self._loader("test"), speed_edges, turn_edges, meters
        )
        resolved = {
            **self.resolved,
            "config_path": str(self.config_path.resolve()),
            "data_root": str(self.data_root),
            "aligned_root": str(self.aligned_root),
            "stage1_train_population": len(train_dataset),
            "stage1_test_population": len(self._dataset("test")),
            "stage1_batch_size": self.budget.stage1_batch_size,
            "stage1_epochs": self.budget.stage1_epochs,
            "stage2_train_trajectories": self.budget.stage2_train_trajectories,
            "stage2_subset_seed": self.budget.stage2_subset_seed,
            "stage2_batch_size": self.budget.stage2_batch_size,
            "stage2_gradient_accumulation": self.budget.stage2_gradient_accumulation,
            "stage2_effective_batch_size": self.budget.stage2_effective_batch_size,
            "stage2_subset_manifests": dict(self.budget.stage2_subset_manifests),
            "stage2_epochs": self.budget.stage2_epochs,
            "stage2_max_steps": self.budget.stage2_max_steps,
            "speed_edges": speed_edges.tolist(),
            "turn_edges": turn_edges.tolist(),
            "meters_per_normalized_coordinate": list(meters),
            "initialization": "same_seed_random_from_scratch",
            "optimizer": {"name": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4},
            "scheduler": {
                "name": "cosine_with_linear_warmup", "warmup_epochs": 10,
                "warmup_lr_ratio": 0.01, "minimum_lr_ratio": 0.10,
                "steps_per_epoch": len(train_loader),
            },
        }
        saved = save_stage1_artifacts(output, model, resolved, history, metrics)
        return {
            "resolved_config.json": saved["resolved_config"],
            "stage1_model.pt": saved["checkpoint"],
            "stage1_metrics.json": saved["stage1_metrics"],
        }

    def _tokens(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "tokens"
        output.mkdir(parents=True, exist_ok=True)
        model = self._load_model(run_dir)
        audits = {}
        paths = {}
        for split in ("train", "test"):
            records = encode_loader(model, self._loader(split), self.device)
            path = output / f"{split}_tokens.jsonl"
            audit = export_token_records(
                records,
                [int(self._dataset(split)[index]["traj_id"]) for index in range(len(self._dataset(split)))],
                path,
            )
            audits[split] = audit
            paths[f"{split}_tokens.jsonl"] = path
        audit_path = output / "token_audit.json"
        _atomic_json(audit_path, audits)
        paths["token_audit.json"] = audit_path
        return paths

    def _raw_frame(self, traj_ids: Iterable[int]) -> pd.DataFrame:
        ids = {int(value) for value in traj_ids}
        raw = pd.read_parquet(self.data_root / "traj.parquet")
        return raw[raw["traj_id"].isin(ids)].copy()

    def _stage2_data(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "stage2_data"
        output.mkdir(parents=True, exist_ok=True)
        roads = pd.read_csv(self.data_root / "rn" / "edge_info.csv")
        frames = {}
        records_by_split = {}
        train_selection = None
        for split in ("train", "test"):
            records = _read_jsonl(run_dir / "tokens" / f"{split}_tokens.jsonl")
            records_by_split[split] = records
            raw = self._raw_frame(record["traj_id"] for record in records)
            frame = build_llm_frame(records, roads, raw, self.city)
            if split == "train":
                frame, train_selection = select_stage2_train_frame(
                    frame,
                    self.budget.stage2_train_trajectories,
                    self.budget.stage2_subset_seed,
                    self.city,
                    self.budget.stage2_subset_manifests.get(self.city),
                )
            path = output / f"llm_{split}.parquet"
            frame.to_parquet(path, index=False)
            frames[split] = (frame, path)
        test_ids = [int(value) for value in frames["test"][0]["traj_id"]]
        requested = self.budget.stage3_trajectories
        generation_ids = (
            sorted(test_ids)
            if requested == "full"
            else deterministic_subset(test_ids, int(requested), self.seed)
        )
        manifest = {
            "train": {
                "population": len(frames["train"][0]),
                "path": str(frames["train"][1].resolve()),
                "sha256": _sha256(frames["train"][1]),
                "selection": train_selection,
            },
            "test": {
                "population": len(frames["test"][0]),
                "path": str(frames["test"][1].resolve()),
                "sha256": _sha256(frames["test"][1]),
            },
            "generation": {"population": len(generation_ids), "traj_ids": generation_ids},
        }
        manifest_path = output / "subset_manifest.json"
        _atomic_json(manifest_path, manifest)
        return {
            "llm_train.parquet": frames["train"][1],
            "llm_test.parquet": frames["test"][1],
            "subset_manifest.json": manifest_path,
        }

    def _stage2(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "stage2"
        frame = pd.read_parquet(run_dir / "stage2_data" / "llm_train.parquet")
        eval_frame = pd.read_parquet(run_dir / "stage2_data" / "llm_test.parquet")
        metadata = json.loads((self.aligned_root / "metadata.json").read_text())
        result = run_hf_stage2(
            frame,
            self.config.paths["model_root"],
            output,
            int(metadata["num_roads"]),
            self.seed,
            self.budget.stage2_epochs,
            self.budget.stage2_max_steps,
            batch_size=self.budget.stage2_batch_size,
            gradient_accumulation=self.budget.stage2_gradient_accumulation,
            eval_frame=eval_frame,
        )
        adapter_manifest = output / "adapter_manifest.json"
        _atomic_json(
            adapter_manifest,
            {
                "adapter": result["adapter"],
                "base_model": str(self.config.paths["model_root"]),
                "seed": self.seed,
                "train_population": len(frame),
                "batch_size": self.budget.stage2_batch_size,
                "gradient_accumulation": self.budget.stage2_gradient_accumulation,
                "effective_batch_size": self.budget.stage2_effective_batch_size,
            },
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "adapter_manifest.json": adapter_manifest,
            "stage2_plan.json": output / "stage2_plan.json",
            "stage2_metrics.json": output / "stage2_metrics.json",
            "token_budget.json": output / "token_budget.json",
        }

    def _stage3(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "stage3"
        output.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(run_dir / "stage2_data" / "llm_test.parquet")
        manifest = json.loads((run_dir / "stage2_data" / "subset_manifest.json").read_text())
        ids = set(int(value) for value in manifest["generation"]["traj_ids"])
        frame = frame[frame["traj_id"].isin(ids)].copy()
        metadata = json.loads((self.aligned_root / "metadata.json").read_text())
        result = run_hf_generation(
            frame,
            run_dir / "stage2" / "adapter",
            output / "generations.jsonl",
            int(metadata["num_roads"]),
            self.seed,
            device=str(self.device),
        )
        audit_path = output / "generation_audit.json"
        _atomic_json(audit_path, result["audit"])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"generations.jsonl": output / "generations.jsonl", "generation_audit.json": audit_path}

    def _g_attribution(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "g_attribution"
        output.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((run_dir / "stage2_data" / "subset_manifest.json").read_text())
        ids = set(int(value) for value in manifest["generation"]["traj_ids"])
        oracle = [
            record
            for record in _read_jsonl(run_dir / "tokens" / "test_tokens.jsonl")
            if int(record["traj_id"]) in ids
        ]
        generated = _read_jsonl(run_dir / "stage3" / "generations.jsonl")
        variants = build_g_variants(oracle, generated)
        path = output / "g_variants.json"
        _atomic_json(path, variants)
        return {"g_variants.json": path}

    def _road_geometries(self) -> dict[int, np.ndarray]:
        roads = pd.read_csv(self.data_root / "rn" / "edge_info.csv")
        return {
            index: np.asarray(list(shapely.from_wkt(value).coords), dtype=np.float64)
            for index, value in enumerate(roads["geometry"])
        }

    def _decode(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "decode"
        output.mkdir(parents=True, exist_ok=True)
        variants = json.loads((run_dir / "g_attribution" / "g_variants.json").read_text())
        ids = {int(row["traj_id"]) for row in variants["g00"]}
        dataset = self._dataset("test")
        items = {
            int(item["traj_id"]): item
            for item in (dataset[index] for index in range(len(dataset)))
            if int(item["traj_id"]) in ids
        }
        contexts = {}
        collator = make_collator(self._context())
        for traj_id, item in items.items():
            batch = collator([item])
            contexts[traj_id] = {
                name: batch[name]
                for name in ("route", "route_lengths", "route_geometry", "route_geometry_lengths")
            }
        model = self._load_model(run_dir)
        road_geometries = self._road_geometries()
        z_score = self._context().z_score
        artifacts = {}
        for variant in ("g00", "g10", "g01", "g11"):
            decoded = decode_variant_records(model, variants[variant], contexts, self.device)
            reconstructed = [reconstruct_gps(row, road_geometries, z_score) for row in decoded]
            path = output / f"{variant}_reconstructed.jsonl"
            _atomic_jsonl(path, reconstructed)
            artifacts[f"{variant}_reconstructed.jsonl"] = path
        return artifacts

    def _stage4(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "stage4"
        output.mkdir(parents=True, exist_ok=True)
        variants = {}
        ids = None
        for variant in ("g00", "g10", "g01", "g11"):
            records = _read_jsonl(run_dir / "decode" / f"{variant}_reconstructed.jsonl")
            variants[variant] = records
            current = {int(record["traj_id"]) for record in records}
            ids = current if ids is None else ids
            if current != ids:
                raise CellRunError("G decode variants do not share an exact population")
        dataset = self._dataset("test")
        real_gps = {}
        real_roads = {}
        for index in range(len(dataset)):
            item = dataset[index]
            traj_id = int(item["traj_id"])
            if traj_id in ids:
                record = dataset.records_by_id[traj_id]
                real_gps[traj_id] = np.asarray(record["gps_seq"], dtype=np.float64)
                real_roads[traj_id] = [int(value) for value in item["point_road"]]
        geometries = self._road_geometries()
        metrics = {
            variant: evaluate_population(
                real_gps, real_roads, records, geometries, self.city, invalid_penalty=2.0,
                mbr=dataset.metadata["mbr"],
                grid=dataset.metadata["grid"],
            )
            for variant, records in variants.items()
        }
        path = output / "stage4_metrics.json"
        _atomic_json(path, metrics)
        return {"stage4_metrics.json": path}
