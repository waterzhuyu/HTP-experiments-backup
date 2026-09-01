from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .config import ExperimentConfig
from .reevaluation_evidence import (
    audit_source_population,
    load_stratified_manifest,
    resolve_source_evidence,
)
from .report_results import G_VARIANTS, _validate_stage4, read_stage_artifact
from .run_experiment import build_jobs, completed_attempt
from .stratified_subset import trajectory_id_sha256


SCHEMA_VERSION = 1
REGISTERED_BUDGET = "controlled"


class CompletedSourceAuditError(RuntimeError):
    """Raised when completed formal sources cannot support corrected reevaluation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git_state() -> dict[str, object]:
    anchor = Path(__file__).resolve().parent
    try:
        root = subprocess.run(
            ["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        revision = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise CompletedSourceAuditError("cannot resolve audit Git provenance") from error
    if len(revision) != 40:
        raise CompletedSourceAuditError("audit Git revision is not a full commit hash")
    return {
        "worktree": str(Path(root).resolve()),
        "revision": revision,
        "dirty": bool(status),
    }


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _write_immutable_json(path: Path, payload: object) -> Path:
    content = _canonical_json(payload)
    if path.is_file():
        if path.read_bytes() != content:
            raise CompletedSourceAuditError(
                f"refusing to replace immutable completed-source audit: {path}"
            )
        return path
    if path.exists():
        raise CompletedSourceAuditError(f"audit output is not a regular file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise CompletedSourceAuditError(
            f"stale completed-source audit temporary file exists: {temporary}"
        ) from error
    try:
        os.link(temporary, path)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != content:
            raise CompletedSourceAuditError(
                f"concurrent immutable completed-source audit mismatch: {path}"
            )
    finally:
        temporary.unlink()
    return path


def _adapter_tree_sha256(rows: tuple[dict[str, object], ...]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stage4_population_audit(
    payload: Mapping[str, object],
    selected_ids: list[int],
    run_key: str,
) -> dict[str, object]:
    _validate_stage4(payload, run_key)
    expected_hash = trajectory_id_sha256(selected_ids)
    variant_hashes = {}
    original_ids: list[int] | None = None
    for variant in G_VARIANTS:
        rows = payload[variant]["per_trajectory"]  # type: ignore[index]
        ids = sorted(int(row["traj_id"]) for row in rows)  # type: ignore[index]
        current_hash = trajectory_id_sha256(ids)
        if len(ids) != len(selected_ids):
            raise CompletedSourceAuditError(
                f"{run_key}/{variant} Stage4 population size differs from the "
                "registered corrected-reevaluation size"
            )
        if original_ids is None:
            original_ids = ids
        variant_hashes[variant] = current_hash
    if original_ids is None:
        raise CompletedSourceAuditError(f"{run_key} has no Stage4 population")
    original_hash = trajectory_id_sha256(original_ids)
    matches_fixed_subset = original_ids == selected_ids and original_hash == expected_hash
    return {
        "population": len(selected_ids),
        "original_stage4_traj_id_sha256": original_hash,
        "fixed_subset_traj_id_sha256": expected_hash,
        "g_variant_traj_id_sha256": variant_hashes,
        "g_variants_use_exact_same_population": True,
        "original_stage4_matches_fixed_subset": matches_fixed_subset,
        "corrected_stage3_stage4_reevaluation_required": not matches_fixed_subset,
    }


def audit_completed_sources(
    config_path: str | Path,
    source_root: str | Path,
    subset_root: str | Path,
    *,
    output_path: str | Path | None = None,
    budget: str = REGISTERED_BUDGET,
    expected_source_revision: str | None = None,
    expected_completed_count: int | None = None,
    expected_generator_revision: str | None = None,
    expected_config_sha256: str | None = None,
    require_clean: bool = False,
) -> dict[str, object]:
    if budget != REGISTERED_BUDGET:
        raise CompletedSourceAuditError(
            "completed-source audit is registered only for controlled"
        )
    resolved_config = Path(config_path).resolve()
    config = ExperimentConfig.load(resolved_config)
    config_artifact = _artifact(resolved_config)
    generator_git = _git_state()
    if (
        expected_generator_revision is not None
        and generator_git["revision"] != expected_generator_revision
    ):
        raise CompletedSourceAuditError("audit generator revision mismatch")
    if (
        expected_config_sha256 is not None
        and config_artifact["sha256"] != expected_config_sha256
    ):
        raise CompletedSourceAuditError("audit config SHA-256 mismatch")
    if require_clean and generator_git["dirty"]:
        raise CompletedSourceAuditError("audit generator worktree must be clean")
    stage3_population = config.budgets[budget].stage3_trajectories
    if not isinstance(stage3_population, int) or isinstance(stage3_population, bool):
        raise CompletedSourceAuditError(
            "controlled Stage3 population must be a fixed integer"
        )

    experiment_root = Path(source_root).resolve()
    subsets = Path(subset_root).resolve()
    subset_index_path = subsets / "index.json"
    if not subset_index_path.is_file():
        raise CompletedSourceAuditError("fixed-subset index is missing")
    rows = []
    missing = []
    jobs = build_jobs(config, budget)
    for job in jobs:
        attempt = completed_attempt(experiment_root / job.run_key)
        if attempt is None:
            missing.append(job.run_key)
            continue
        evidence = resolve_source_evidence(
            resolved_config,
            attempt,
            cell_id=job.cell_id,
            city=job.city,
            seed=job.seed,
            budget=job.budget,
            expected_revision=expected_source_revision,
        )
        subset = load_stratified_manifest(
            subsets / f"{budget}_{job.city}_seed_{job.seed}.json",
            city=job.city,
            seed=job.seed,
            expected_size=stage3_population,
        )
        population = audit_source_population(
            evidence,
            subset,
            config.paths[f"{job.city}_aligned_root"],
        )
        stage4_path = read_stage_artifact(attempt, "stage4", "stage4_metrics.json")
        stage4_payload = json.loads(stage4_path.read_text(encoding="utf-8"))
        selected_ids = [int(value) for value in subset["selected_traj_ids"]]
        stage4_audit = _stage4_population_audit(
            stage4_payload,
            selected_ids,
            job.run_key,
        )
        rows.append(
            {
                "run_key": job.run_key,
                "cell_id": job.cell_id,
                "city": job.city,
                "seed": job.seed,
                "budget": job.budget,
                "source_attempt": str(evidence.attempt),
                "source_revision": evidence.revision,
                "source_artifact_markers_verified": True,
                "adapter_tree_file_count": len(evidence.adapter_tree),
                "adapter_tree_sha256": _adapter_tree_sha256(evidence.adapter_tree),
                "population_audit": population,
                "fixed_subset": {
                    "path": subset["manifest_path"],
                    "sha256": subset["manifest_sha256"],
                    "selected_population": len(selected_ids),
                    "selected_traj_id_sha256": subset[
                        "selected_traj_id_sha256"
                    ],
                },
                "stage4_artifact": _artifact(stage4_path),
                "stage4_population_audit": stage4_audit,
            }
        )

    if expected_completed_count is not None and len(rows) != expected_completed_count:
        raise CompletedSourceAuditError(
            "completed-source count mismatch: "
            f"expected {expected_completed_count}, observed {len(rows)}"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "completed_formal_source_evidence_audit",
        "experiment_id": config.experiment_id,
        "budget": budget,
        "config": config_artifact,
        "generator_git": generator_git,
        "source_root": str(experiment_root),
        "expected_source_revision": expected_source_revision,
        "fixed_subset_index": _artifact(subset_index_path),
        "formal_core_job_count": len(jobs),
        "audited_completed_source_count": len(rows),
        "missing_completed_source_count": len(missing),
        "all_formal_core_sources_completed": not missing,
        "audited_completed_sources": rows,
        "missing_completed_sources": missing,
    }
    if output_path is None:
        return payload
    output = _write_immutable_json(Path(output_path).resolve(), payload)
    return {
        **payload,
        "output_path": str(output),
        "output_sha256": _sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit completed formal sources for corrected reevaluation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", default=REGISTERED_BUDGET)
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-completed-count", type=int)
    parser.add_argument("--expected-generator-revision")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--require-clean", action="store_true")
    arguments = parser.parse_args()
    result = audit_completed_sources(
        arguments.config,
        arguments.source_root,
        arguments.subset_root,
        output_path=arguments.output,
        budget=arguments.budget,
        expected_source_revision=arguments.expected_source_revision,
        expected_completed_count=arguments.expected_completed_count,
        expected_generator_revision=arguments.expected_generator_revision,
        expected_config_sha256=arguments.expected_config_sha256,
        require_clean=arguments.require_clean,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
