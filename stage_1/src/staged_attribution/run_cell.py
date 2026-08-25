from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .config import ExperimentConfig


class CellRunError(RuntimeError):
    """Raised when a cell DAG would skip or misreport a required stage."""


@dataclass(frozen=True)
class StageNode:
    name: str
    prerequisites: tuple[str, ...]
    required_artifacts: tuple[str, ...]


def build_cell_dag(cell_id: str) -> tuple[StageNode, ...]:
    if not cell_id:
        raise CellRunError("cell_id must be non-empty")
    return (
        StageNode("stage1", (), ("resolved_config.json", "stage1_model.pt", "stage1_metrics.json")),
        StageNode("tokens", ("stage1",), ("train_tokens.jsonl", "test_tokens.jsonl", "token_audit.json")),
        StageNode("stage2_data", ("tokens",), ("llm_train.parquet", "llm_test.parquet", "subset_manifest.json")),
        StageNode("stage2", ("stage2_data",), ("adapter_manifest.json", "stage2_metrics.json", "token_budget.json")),
        StageNode("stage3", ("stage2",), ("generations.jsonl", "generation_audit.json")),
        StageNode("g_attribution", ("stage3",), ("g_variants.json",)),
        StageNode(
            "decode",
            ("g_attribution",),
            ("g00_reconstructed.jsonl", "g10_reconstructed.jsonl", "g01_reconstructed.jsonl", "g11_reconstructed.jsonl"),
        ),
        StageNode("stage4", ("decode",), ("stage4_metrics.json",)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_path(run_dir: Path, stage: str) -> Path:
    return run_dir / stage / "complete.json"


def _read_marker(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CellRunError(f"invalid stage marker {path}: {error}") from error


def write_stage_marker(
    run_dir: str | Path,
    node: StageNode,
    artifacts: Mapping[str, str | Path],
) -> Path:
    root = Path(run_dir)
    missing_prerequisites = [
        stage for stage in node.prerequisites if not _marker_path(root, stage).is_file()
    ]
    if missing_prerequisites:
        raise CellRunError(f"stage {node.name} missing prerequisite markers {missing_prerequisites}")
    missing = [name for name in node.required_artifacts if name not in artifacts]
    if missing:
        raise CellRunError(f"stage {node.name} missing artifact declarations {missing}")
    recorded: dict[str, object] = {}
    for name in node.required_artifacts:
        path = Path(artifacts[name]).resolve()
        if not path.is_file():
            raise CellRunError(f"stage {node.name} missing artifact file {name}: {path}")
        recorded[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    marker = _marker_path(root, node.name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"stage": node.name, "artifacts": recorded}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)
    return marker


def resume_plan(run_dir: str | Path, dag: tuple[StageNode, ...]) -> dict[str, object]:
    root = Path(run_dir)
    completed: list[str] = []
    gap_found = False
    for node in dag:
        marker_path = _marker_path(root, node.name)
        if not marker_path.is_file():
            gap_found = True
            continue
        if gap_found:
            raise CellRunError(f"stage {node.name} has a marker after an incomplete prerequisite")
        marker = _read_marker(marker_path)
        if marker.get("stage") != node.name:
            raise CellRunError(f"stage marker name mismatch at {marker_path}")
        artifacts = marker.get("artifacts", {})
        for name in node.required_artifacts:
            if name not in artifacts or not Path(artifacts[name]["path"]).is_file():
                raise CellRunError(f"stage marker {node.name} references missing artifact {name}")
        completed.append(node.name)
    remaining = [node.name for node in dag if node.name not in completed]
    return {
        "completed": completed,
        "remaining": remaining,
        "terminal_complete": bool(completed and completed[-1] == dag[-1].name),
    }


def run_cell(
    config_path: str | Path,
    cell_id: str,
    city: str,
    seed: int,
    budget: str,
    run_dir: str | Path,
    dry_run: bool = False,
    stage_backend: Callable[[StageNode, Mapping[str, object], Path], Mapping[str, str | Path]] | None = None,
    device: str = "cuda",
) -> dict[str, object]:
    config = ExperimentConfig.load(config_path)
    cells = {cell.cell_id: cell for cell in config.core_cells()}
    if cell_id not in cells:
        raise CellRunError(f"unknown core cell {cell_id}")
    if city not in config.cities:
        raise CellRunError(f"unknown city {city}")
    if seed not in config.seeds:
        raise CellRunError(f"unregistered seed {seed}")
    if budget not in config.budgets:
        raise CellRunError(f"unknown budget {budget}")
    cell = cells[cell_id]
    dag = build_cell_dag(cell_id)
    resolved = {
        "experiment_id": config.experiment_id,
        "cell_id": cell_id,
        "city": city,
        "seed": seed,
        "budget": budget,
        "views": sorted(cell.views),
        "q_weights": dict(cell.q_weights),
        "stages": [node.name for node in dag],
        "device": device,
        "will_execute": not dry_run,
    }
    if dry_run:
        return resolved
    root = Path(run_dir)
    if stage_backend is None:
        try:
            from .pipeline import PipelineBackend
        except ImportError as error:
            raise CellRunError("default pipeline backend is unavailable") from error
        stage_backend = PipelineBackend(config_path, resolved).execute
    plan = resume_plan(root, dag)
    remaining = set(plan["remaining"])
    for node in dag:
        if node.name not in remaining:
            continue
        artifacts = stage_backend(node, resolved, root)
        write_stage_marker(root, node, artifacts)
    final = resume_plan(root, dag)
    if not final["terminal_complete"]:
        raise CellRunError("cell backend returned without Stage4 terminal completion")
    return {**resolved, **final}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one complete staged-attribution cell")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", choices=("smoke", "controlled", "confirmation"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    result = run_cell(
        arguments.config,
        arguments.cell,
        arguments.city,
        arguments.seed,
        arguments.budget,
        arguments.run_dir,
        dry_run=arguments.dry_run,
        device=arguments.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
