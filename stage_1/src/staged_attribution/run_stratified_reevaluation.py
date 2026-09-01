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
    anchor = Path(__file__).resolve().parent
    revision = subprocess.run(
        ["git", "-C", str(anchor), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(anchor), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def _validate_digest(value: str, length: int, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ReevaluationRunError(f"{label} must be a full lowercase hex digest")
    return normalized


def _validate_source_audit_binding(
    path: Path,
    *,
    config: ExperimentConfig,
    config_sha256: str,
    source_attempt: Path,
    subset_manifest: Path,
    cell_id: str,
    city: str,
    seed: int,
    budget: str,
    expected_source_revision: str,
) -> None:
    audit = _read_manifest(path)
    if audit.get("schema_version") != 1:
        raise ReevaluationRunError("completed-source audit schema mismatch")
    if audit.get("kind") != "completed_formal_source_evidence_audit":
        raise ReevaluationRunError("completed-source audit kind mismatch")
    if audit.get("experiment_id") != config.experiment_id:
        raise ReevaluationRunError("completed-source audit experiment mismatch")
    if audit.get("budget") != budget:
        raise ReevaluationRunError("completed-source audit budget mismatch")
    if audit.get("all_requested_sources_completed") is not True:
        raise ReevaluationRunError("completed-source audit is not complete")
    if audit.get("missing_completed_source_count") != 0:
        raise ReevaluationRunError("completed-source audit records missing sources")
    audit_config = audit.get("config")
    if not isinstance(audit_config, Mapping) or audit_config.get("sha256") != config_sha256:
        raise ReevaluationRunError("completed-source audit config SHA-256 mismatch")

    run_key = f"{budget}/{city}/seed_{seed}/{cell_id}"
    rows = audit.get("audited_completed_sources")
    if not isinstance(rows, list):
        raise ReevaluationRunError("completed-source audit rows are invalid")
    matching = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("run_key") == run_key
    ]
    if len(matching) != 1:
        raise ReevaluationRunError(
            f"completed-source audit must contain exactly one row for {run_key}"
        )
    row = matching[0]
    if (
        row.get("cell_id") != cell_id
        or row.get("city") != city
        or row.get("seed") != seed
        or row.get("budget") != budget
    ):
        raise ReevaluationRunError("completed-source audit job identity mismatch")
    if Path(str(row.get("source_attempt", ""))).resolve() != source_attempt:
        raise ReevaluationRunError("completed-source audit attempt mismatch")
    if row.get("source_revision") != expected_source_revision:
        raise ReevaluationRunError("completed-source audit source revision mismatch")
    fixed_subset = row.get("fixed_subset")
    if not isinstance(fixed_subset, Mapping):
        raise ReevaluationRunError("completed-source audit lacks subset evidence")
    if Path(str(fixed_subset.get("path", ""))).resolve() != subset_manifest:
        raise ReevaluationRunError("completed-source audit subset path mismatch")
    if (
        not subset_manifest.is_file()
        or subset_manifest.is_symlink()
        or fixed_subset.get("sha256") != _sha256(subset_manifest)
    ):
        raise ReevaluationRunError("completed-source audit subset SHA-256 mismatch")


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
    expected_revision: str,
    expected_config_sha256: str,
    expected_source_revision: str,
    source_audit_path: str | Path,
    expected_source_audit_sha256: str,
    device: str = "cuda",
    stage_backend: Callable[
        [StageNode, Mapping[str, object], Path], Mapping[str, str | Path]
    ] | None = None,
    command: list[str] | None = None,
    git_identity_fn: Callable[[], tuple[str, bool]] = _git_identity,
) -> dict[str, object]:
    expected_revision = _validate_digest(
        expected_revision, 40, "expected reevaluation revision"
    )
    expected_config_sha256 = _validate_digest(
        expected_config_sha256, 64, "expected config SHA-256"
    )
    expected_source_revision = _validate_digest(
        expected_source_revision, 40, "expected source revision"
    )
    expected_source_audit_sha256 = _validate_digest(
        expected_source_audit_sha256, 64, "expected source-audit SHA-256"
    )
    resolved_config = Path(config_path).resolve()
    actual_config_sha256 = _sha256(resolved_config)
    if actual_config_sha256 != expected_config_sha256:
        raise ReevaluationRunError("reevaluation config SHA-256 mismatch")
    resolved_source_audit = Path(source_audit_path).resolve()
    if (
        not resolved_source_audit.is_file()
        or resolved_source_audit.is_symlink()
        or _sha256(resolved_source_audit) != expected_source_audit_sha256
    ):
        raise ReevaluationRunError("completed-source audit SHA-256 mismatch")
    config = ExperimentConfig.load(resolved_config)
    config.cell(cell_id)
    if city not in config.cities or seed not in config.seeds or budget not in config.budgets:
        raise ReevaluationRunError("unknown city, seed, or budget")
    if budget != "controlled":
        raise ReevaluationRunError("stratified reevaluation is registered only for controlled budget")
    resolved_source_attempt = Path(source_attempt).resolve()
    resolved_subset_manifest = Path(subset_manifest).resolve()
    _validate_source_audit_binding(
        resolved_source_audit,
        config=config,
        config_sha256=actual_config_sha256,
        source_attempt=resolved_source_attempt,
        subset_manifest=resolved_subset_manifest,
        cell_id=cell_id,
        city=city,
        seed=int(seed),
        budget=budget,
        expected_source_revision=expected_source_revision,
    )
    root = Path(run_dir)
    dag = build_reevaluation_dag()
    revision, dirty = git_identity_fn()
    if revision != expected_revision:
        raise ReevaluationRunError("reevaluation implementation revision mismatch")
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
        "config_sha256": actual_config_sha256,
        "expected_source_revision": expected_source_revision,
        "source_audit_path": str(resolved_source_audit),
        "source_audit_sha256": expected_source_audit_sha256,
        "source_attempt": str(resolved_source_attempt),
        "subset_manifest": str(resolved_subset_manifest),
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
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--expected-source-audit-sha256", required=True)
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
        expected_revision=arguments.expected_revision,
        expected_config_sha256=arguments.expected_config_sha256,
        expected_source_revision=arguments.expected_source_revision,
        source_audit_path=arguments.source_audit,
        expected_source_audit_sha256=arguments.expected_source_audit_sha256,
        device=arguments.device,
        command=sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
