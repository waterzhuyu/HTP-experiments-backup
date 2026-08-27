from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import make_collator
from .decode_stage3 import decode_variant_records, reconstruct_gps
from .g_attribution import build_g_variants
from .generate_stage3 import run_hf_generation
from .pipeline import PipelineBackend, _atomic_json, _atomic_jsonl, _read_jsonl
from .population_bootstrap import (
    POPULATION_REGRET_METHOD,
    build_metric_contributions,
    population_regrets,
    save_metric_contributions,
)
from .reevaluation_evidence import (
    SourceEvidence,
    audit_source_population,
    load_stratified_manifest,
    resolve_source_evidence,
    write_source_reference,
)
from .run_cell import CellRunError, StageNode
from .stage4_metrics import evaluate_population


class StratifiedReevaluationPipeline:
    """Reuse immutable Stage1/2 evidence and rerun only corrected Stage3→4."""

    def __init__(self, config_path: str | Path, identity: Mapping[str, object]) -> None:
        self.config_path = Path(config_path)
        self.config = ExperimentConfig.load(config_path)
        self.identity = dict(identity)
        self.city = str(identity["city"])
        self.seed = int(identity["seed"])
        self.cell_id = str(identity["cell_id"])
        self.budget_name = str(identity["budget"])
        self.source_attempt = Path(str(identity["source_attempt"]))
        self.subset_path = Path(str(identity["subset_manifest"]))
        self.device = str(identity["device"])
        self.backend = PipelineBackend(config_path, identity)
        self._evidence: SourceEvidence | None = None
        self._subset: dict[str, object] | None = None
        self._population_audit: dict[str, object] | None = None

    def execute(
        self, node: StageNode, identity: Mapping[str, object], run_dir: Path
    ) -> Mapping[str, Path]:
        del identity
        handler = getattr(self, f"_{node.name}", None)
        if handler is None:
            raise CellRunError(f"reevaluation pipeline has no handler for {node.name}")
        return handler(run_dir)

    def _source(self) -> SourceEvidence:
        if self._evidence is None:
            self._evidence = resolve_source_evidence(
                self.config_path,
                self.source_attempt,
                cell_id=self.cell_id,
                city=self.city,
                seed=self.seed,
                budget=self.budget_name,
            )
        return self._evidence

    def _selection(self) -> dict[str, object]:
        if self._subset is None:
            requested = self.config.budgets[self.budget_name].stage3_trajectories
            if not isinstance(requested, int):
                raise CellRunError("controlled stratified reevaluation requires an integer population")
            self._subset = load_stratified_manifest(
                self.subset_path, city=self.city, seed=self.seed, expected_size=requested
            )
        return self._subset

    def _audit(self) -> dict[str, object]:
        if self._population_audit is None:
            self._population_audit = audit_source_population(
                self._source(), self._selection(), self.backend.aligned_root
            )
        return self._population_audit

    def _selected_ids(self) -> set[int]:
        return {int(value) for value in self._selection()["selected_traj_ids"]}

    def _source_reference(self, run_dir: Path) -> Mapping[str, Path]:
        path = write_source_reference(
            run_dir / "source_reference" / "source_reference.json",
            self._source(), self._selection(), self._audit(),
        )
        return {"source_reference.json": path}

    def _stage3(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "stage3"
        output.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(self._source().llm_test)
        ids = self._selected_ids()
        frame = frame[frame["traj_id"].isin(ids)].copy()
        if len(frame) != len(ids) or set(map(int, frame["traj_id"])) != ids:
            raise CellRunError("source LLM frame does not contain the exact stratified population")
        metadata = json.loads((self.backend.aligned_root / "metadata.json").read_text())
        result = run_hf_generation(
            frame,
            self._source().adapter_dir,
            output / "generations.jsonl",
            int(metadata["num_roads"]),
            self.seed,
            device=self.device,
        )
        audit_path = output / "generation_audit.json"
        _atomic_json(
            audit_path,
            {
                **result["audit"],
                "subset_manifest_sha256": self._selection()["manifest_sha256"],
                "selected_traj_id_sha256": self._selection()["selected_traj_id_sha256"],
                "source_revision": self._source().revision,
            },
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "generations.jsonl": output / "generations.jsonl",
            "generation_audit.json": audit_path,
        }

    def _g_attribution(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "g_attribution"
        output.mkdir(parents=True, exist_ok=True)
        ids = self._selected_ids()
        oracle = [
            record for record in _read_jsonl(self._source().test_tokens)
            if int(record["traj_id"]) in ids
        ]
        generated = _read_jsonl(run_dir / "stage3" / "generations.jsonl")
        variants = build_g_variants(oracle, generated)
        if any({int(row["traj_id"]) for row in rows} != ids for rows in variants.values()):
            raise CellRunError("G variants do not preserve the exact stratified population")
        path = output / "g_variants.json"
        _atomic_json(path, variants)
        return {"g_variants.json": path}

    def _decode(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "decode"
        output.mkdir(parents=True, exist_ok=True)
        variants = json.loads((run_dir / "g_attribution" / "g_variants.json").read_text())
        ids = self._selected_ids()
        dataset = self.backend._dataset("test")
        items = {
            int(item["traj_id"]): item
            for item in (dataset[index] for index in range(len(dataset)))
            if int(item["traj_id"]) in ids
        }
        if set(items) != ids:
            raise CellRunError("decoder dataset does not contain the exact stratified population")
        contexts = {}
        collator = make_collator(self.backend._context())
        for traj_id, item in items.items():
            batch = collator([item])
            contexts[traj_id] = {
                name: batch[name]
                for name in ("route", "route_lengths", "route_geometry", "route_geometry_lengths")
            }
        model = self.backend._load_model(self.source_attempt)
        road_geometries = self.backend._road_geometries()
        z_score = self.backend._context().z_score
        artifacts = {}
        for variant in ("g00", "g10", "g01", "g11"):
            decoded = decode_variant_records(model, variants[variant], contexts, self.backend.device)
            reconstructed = [reconstruct_gps(row, road_geometries, z_score) for row in decoded]
            path = output / f"{variant}_reconstructed.jsonl"
            _atomic_jsonl(path, reconstructed)
            artifacts[f"{variant}_reconstructed.jsonl"] = path
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return artifacts

    def _real_population(self) -> tuple[dict[int, np.ndarray], dict[int, list[int]]]:
        ids = self._selected_ids()
        dataset = self.backend._dataset("test")
        real_gps: dict[int, np.ndarray] = {}
        real_roads: dict[int, list[int]] = {}
        for index in range(len(dataset)):
            item = dataset[index]
            traj_id = int(item["traj_id"])
            if traj_id in ids:
                record = dataset.records_by_id[traj_id]
                real_gps[traj_id] = np.asarray(record["gps_seq"], dtype=np.float64)
                real_roads[traj_id] = [int(value) for value in item["point_road"]]
        if set(real_gps) != ids or set(real_roads) != ids:
            raise CellRunError("Stage4 real population does not match the stratified manifest")
        return real_gps, real_roads

    def _stage4(self, run_dir: Path) -> Mapping[str, Path]:
        output = run_dir / "stage4"
        output.mkdir(parents=True, exist_ok=True)
        ids = self._selected_ids()
        variants = {
            variant: _read_jsonl(run_dir / "decode" / f"{variant}_reconstructed.jsonl")
            for variant in ("g00", "g10", "g01", "g11")
        }
        if any({int(row["traj_id"]) for row in rows} != ids for rows in variants.values()):
            raise CellRunError("decoded G variants do not share the stratified population")
        real_gps, real_roads = self._real_population()
        geometries = self.backend._road_geometries()
        dataset = self.backend._dataset("test")
        metrics = {}
        artifacts: dict[str, Path] = {}
        for variant, records in variants.items():
            payload = evaluate_population(
                real_gps, real_roads, records, geometries, self.city,
                invalid_penalty=2.0, mbr=dataset.metadata["mbr"], grid=dataset.metadata["grid"],
            )
            contributions = build_metric_contributions(
                real_gps, real_roads, records, geometries, self.city,
                mbr=dataset.metadata["mbr"], grid=dataset.metadata["grid"],
            )
            point = population_regrets(contributions)
            payload["population_aware_regret"] = {
                "method": POPULATION_REGRET_METHOD,
                "metrics": {metric: float(value[0]) for metric, value in point.items()},
            }
            metrics[variant] = payload
            manifest = save_metric_contributions(
                contributions, output / f"{variant}_contributions"
            )
            artifacts[f"{variant}_contributions_manifest.json"] = manifest
        path = output / "stage4_metrics.json"
        _atomic_json(path, metrics)
        artifacts["stage4_metrics.json"] = path
        return artifacts
