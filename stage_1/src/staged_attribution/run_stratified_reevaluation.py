from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path

from .config import ExperimentConfig
from .run_cell import StageNode, write_stage_marker


class ReevaluationRunError(RuntimeError):
    """Raised when a corrected reevaluation violates immutable evidence rules."""


def build_reevaluation_dag() -> tuple[StageNode, ...]:
    return (
        StageNode("source_reference", (), ("source_reference.json",)),
        StageNode(
            "stage3", ("source_reference",),
            ("generations.jsonl", "generation_audit.json", "generation_progress.json"),
        ),
        StageNode("g_attribution", ("stage3",), ("g_variants.json",)),
        StageNode(
            "decode", ("g_attribution",),
            (
                "g00_reconstructed.jsonl", "g10_reconstructed.jsonl",
                "g01_reconstructed.jsonl", "g11_reconstructed.jsonl",
            ),
        ),
        StageNode(
            "stage4", ("decode",),
            (
                "stage4_metrics.json", "g00_contributions_manifest.json",
                "g10_contributions_manifest.json", "g01_contributions_manifest.json",
                "g11_contributions_manifest.json",
            ),
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReevaluationRunError(f"invalid reevaluation manifest {path}: {error}") from error


def _verified_completed_stages(root: Path, dag: tuple[StageNode, ...]) -> list[str]:
    completed = []
    gap = False
    resolved_root = root.resolve()
    for node in dag:
        marker_path = root / node.name / "complete.json"
        if not marker_path.is_file():
            gap = True
            continue
        if gap:
            raise ReevaluationRunError(f"stage {node.name} has a marker after an incomplete prerequisite")
        marker = _read_manifest(marker_path)
        if marker.get("stage") != node.name:
            raise ReevaluationRunError(f"stage marker name mismatch for {node.name}")
        artifacts = marker.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ReevaluationRunError(f"stage {node.name} marker lacks artifacts")
        for name in node.required_artifacts:
            info = artifacts.get(name)
            if not isinstance(info, Mapping):
                raise ReevaluationRunError(f"stage {node.name} lacks artifact {name}")
            path = Path(str(info.get("path", ""))).resolve()
            if not path.is_relative_to(resolved_root):
                raise ReevaluationRunError(f"stage {node.name} artifact escapes reevaluation attempt")
            if not path.is_file() or path.stat().st_size != int(info.get("size_bytes", -1)):
                raise ReevaluationRunError(f"stage {node.name} artifact missing or size mismatch: {name}")
            if _sha256(path) != str(info.get("sha256")):
                raise ReevaluationRunError(f"stage {node.name} artifact SHA256 mismatch: {name}")
        completed.append(node.name)
    return completed


def _git_identity() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return revision, dirty


def run_stratified_reevaluation(
    config_path: str | Path,
    *,
    source_attempt: str | Path,
    subset_manifest: str | Path,
    cell_id: str,
    city: str,
    seed: int,
    budget: str,
    run_dir: str | Path,
    device: str = "cuda",
    stage_backend: Callable[
        [StageNode, Mapping[str, object], Path], Mapping[str, str | Path]
    ] | None = None,
    command: list[str] | None = None,
    git_identity_fn: Callable[[], tuple[str, bool]] = _git_identity,
) -> dict[str, object]:
    config = ExperimentConfig.load(config_path)
    config.cell(cell_id)
    if city not in config.cities or seed not in config.seeds or budget not in config.budgets:
        raise ReevaluationRunError("unknown city, seed, or budget")
    if budget != "controlled":
        raise ReevaluationRunError("stratified reevaluation is registered only for controlled budget")
    root = Path(run_dir)
    dag = build_reevaluation_dag()
    revision, dirty = git_identity_fn()
    if dirty:
        raise ReevaluationRunError("reevaluation implementation worktree must be clean")
    identity = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "reevaluation": "fixed_stratified_population_aware_v1",
        "cell_id": cell_id,
        "city": city,
        "seed": int(seed),
        "budget": budget,
        "source_attempt": str(Path(source_attempt).resolve()),
        "subset_manifest": str(Path(subset_manifest).resolve()),
        "device": device,
        "git": {"revision": revision, "dirty": dirty},
        "command": command or sys.argv,
    }
    manifest_path = root / "reevaluation_manifest.json"
    if manifest_path.is_file():
        manifest = _read_manifest(manifest_path)
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in identity.items()
            if key not in {"command"} and manifest.get(key) != value
        }
        if mismatches:
            raise ReevaluationRunError(f"reevaluation identity mismatch: {mismatches}")
        if manifest.get("status") == "failed":
            raise ReevaluationRunError("failed reevaluation attempt is immutable")
        if manifest.get("status") == "completed":
            completed = _verified_completed_stages(root, dag)
            if completed != [node.name for node in dag]:
                raise ReevaluationRunError("completed reevaluation lacks terminal Stage4 evidence")
            return {**identity, "status": "completed", "completed": completed, "remaining": []}
    else:
        root.mkdir(parents=True, exist_ok=True)
        unexpected = [path.name for path in root.iterdir() if path.name != "launcher.log"]
        if unexpected:
            raise ReevaluationRunError(
                f"new reevaluation attempt contains unexpected files: {sorted(unexpected)}"
            )
        manifest = {**identity, "status": "running", "failure": None}
        _atomic_json(manifest_path, manifest)
    completed = _verified_completed_stages(root, dag)
    if stage_backend is None:
        from .stratified_reevaluation_pipeline import StratifiedReevaluationPipeline

        stage_backend = StratifiedReevaluationPipeline(config_path, identity).execute
    try:
        for node in dag[len(completed) :]:
            artifacts = stage_backend(node, identity, root)
            resolved_root = root.resolve()
            for name, raw_path in artifacts.items():
                path = Path(raw_path).resolve()
                if not path.is_relative_to(resolved_root):
                    raise ReevaluationRunError(
                        f"stage {node.name} backend artifact escapes attempt: {name}={path}"
                    )
            write_stage_marker(root, node, artifacts)
        completed = _verified_completed_stages(root, dag)
        if completed != [node.name for node in dag]:
            raise ReevaluationRunError("reevaluation backend returned without terminal Stage4 evidence")
        manifest = _read_manifest(manifest_path)
        manifest.update({"status": "completed", "failure": None})
        _atomic_json(manifest_path, manifest)
        return {**identity, "status": "completed", "completed": completed, "remaining": []}
    except Exception as error:
        manifest = _read_manifest(manifest_path)
        manifest.update(
            {
                "status": "failed",
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        _atomic_json(manifest_path, manifest)
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one immutable fixed-stratified reevaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-attempt", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", default="controlled")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    result = run_stratified_reevaluation(
        arguments.config,
        source_attempt=arguments.source_attempt,
        subset_manifest=arguments.subset_manifest,
        cell_id=arguments.cell,
        city=arguments.city,
        seed=arguments.seed,
        budget=arguments.budget,
        run_dir=arguments.run_dir,
        device=arguments.device,
        command=sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
