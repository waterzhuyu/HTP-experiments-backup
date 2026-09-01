from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from staged_attribution.pipeline import _atomic_jsonl
from staged_attribution.population_bootstrap import (
    POPULATION_REGRET_METHOD,
    load_metric_contributions,
)
from staged_attribution.reevaluation_evidence import SourceEvidence
from staged_attribution.stratified_reevaluation_pipeline import StratifiedReevaluationPipeline


class MinimalBackend:
    def __init__(self, aligned_root: Path, dataset=None) -> None:
        self.aligned_root = aligned_root
        self._value = dataset

    def _dataset(self, split):
        assert split == "test"
        return self._value

    def _road_geometries(self):
        return {
            0: np.asarray([[0.0, 0.0], [0.01, 0.0]]),
            1: np.asarray([[0.0, 0.01], [0.01, 0.01]]),
        }


class MinimalDataset:
    def __init__(self) -> None:
        self.items = [
            {"traj_id": 1, "point_road": np.asarray([0, 0, 0])},
            {"traj_id": 2, "point_road": np.asarray([1, 1, 1])},
        ]
        self.records_by_id = {
            1: {"gps_seq": [[0.001, 0.001], [0.004, 0.001], [0.007, 0.001]]},
            2: {"gps_seq": [[0.001, 0.009], [0.004, 0.009], [0.007, 0.009]]},
        }
        self.metadata = {
            "mbr": {"min_lon": 0.0, "max_lon": 0.02, "min_lat": 0.0, "max_lat": 0.02},
            "grid": {
                "grid_size_m": 100.0, "meters_per_lon": 111320.0,
                "meters_per_lat": 110540.0, "n_grid_x": 23, "n_grid_y": 23,
            },
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def pipeline(tmp_path: Path) -> StratifiedReevaluationPipeline:
    llm_test = tmp_path / "llm_test.parquet"
    pd.DataFrame({"traj_id": [1, 2, 3], "desc": ["a", "b", "c"]}).to_parquet(llm_test)
    tokens = tmp_path / "test_tokens.jsonl"
    tokens.write_text(
        "".join(
            json.dumps({"traj_id": value, "lens_type": [0, 0, 0], "codes": [[1, 2, 3, 4]], "road_seq": [0]}) + "\n"
            for value in (1, 2, 3)
        )
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    evidence = SourceEvidence(
        attempt=tmp_path / "source", revision="a" * 40,
        stage1_model=tmp_path / "model.pt", test_tokens=tokens,
        llm_test=llm_test, adapter_manifest=tmp_path / "adapter_manifest.json",
        adapter_dir=adapter, adapter_tree=(),
    )
    value = object.__new__(StratifiedReevaluationPipeline)
    value.city = "porto"
    value.seed = 42
    value.device = "cpu"
    value._evidence = evidence
    value._subset = {
        "selected_traj_ids": [1, 2],
        "selected_traj_id_sha256": "ids",
        "manifest_sha256": "subset",
    }
    value._population_audit = {"lens_formula_verified": True}
    value.backend = MinimalBackend(tmp_path)
    return value


def test_stage3_filters_exact_subset_and_records_provenance(tmp_path: Path, monkeypatch) -> None:
    value = pipeline(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps({"num_roads": 2}))

    def fake_generation(frame, adapter, output, num_roads, seed, device, progress_path):
        assert frame["traj_id"].tolist() == [1, 2]
        assert adapter == value._evidence.adapter_dir
        assert (num_roads, seed, device) == (2, 42, "cpu")
        records = [
            {"traj_id": traj_id, "status": "completed", "lens_type": [0, 0, 0], "codes": [[1, 2, 3, 4]]}
            for traj_id in (1, 2)
        ]
        _atomic_jsonl(Path(output), records)
        Path(progress_path).write_text(json.dumps({"status": "completed"}))
        return {"audit": {"population": 2, "completed": 2, "failed": 0}, "records": records}

    monkeypatch.setattr(
        "staged_attribution.stratified_reevaluation_pipeline.run_hf_generation",
        fake_generation,
    )
    artifacts = value._stage3(tmp_path / "attempt")
    audit = json.loads(artifacts["generation_audit.json"].read_text())
    assert audit["selected_traj_id_sha256"] == "ids"
    assert audit["source_revision"] == "a" * 40
    assert json.loads(artifacts["generation_progress.json"].read_text())["status"] == "completed"


def test_stage4_writes_population_aware_metrics_and_contribution_manifests(tmp_path: Path) -> None:
    value = pipeline(tmp_path)
    dataset = MinimalDataset()
    value.backend = MinimalBackend(tmp_path, dataset)
    run_dir = tmp_path / "attempt"
    for variant in ("g00", "g10", "g01", "g11"):
        rows = [
            {
                "traj_id": traj_id,
                "status": "reconstructed",
                "gps_traj": dataset.records_by_id[traj_id]["gps_seq"],
                "road_traj": dataset.items[traj_id - 1]["point_road"].tolist(),
            }
            for traj_id in (1, 2)
        ]
        _atomic_jsonl(run_dir / "decode" / f"{variant}_reconstructed.jsonl", rows)

    artifacts = value._stage4(run_dir)
    metrics = json.loads(artifacts["stage4_metrics.json"].read_text())
    assert set(metrics) == {"g00", "g10", "g01", "g11"}
    assert all(
        payload["population_aware_regret"]["method"] == POPULATION_REGRET_METHOD
        for payload in metrics.values()
    )
    for variant in ("g00", "g10", "g01", "g11"):
        manifest = artifacts[f"{variant}_contributions_manifest.json"]
        restored = load_metric_contributions(manifest)
        assert restored.traj_ids == (1, 2)
