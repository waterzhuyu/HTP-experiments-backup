from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .report_results import read_stage_artifact
from .stratified_subset import ALGORITHM, trajectory_id_sha256, verify_token_lens_mapping


class ReevaluationEvidenceError(RuntimeError):
    """Raised when a source run cannot support an immutable corrected reevaluation."""


@dataclass(frozen=True)
class SourceEvidence:
    attempt: Path
    revision: str
    stage1_model: Path
    test_tokens: Path
    llm_test: Path
    adapter_manifest: Path
    adapter_dir: Path
    adapter_tree: tuple[dict[str, object], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _adapter_tree(adapter_dir: Path) -> tuple[dict[str, object], ...]:
    if not adapter_dir.is_dir():
        raise ReevaluationEvidenceError(f"missing Stage2 adapter directory: {adapter_dir}")
    rows = []
    for path in sorted(value for value in adapter_dir.rglob("*") if value.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(adapter_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows or not any(row["relative_path"] == "adapter_model.safetensors" for row in rows):
        raise ReevaluationEvidenceError("Stage2 adapter tree lacks adapter_model.safetensors")
    return tuple(rows)


def resolve_source_evidence(
    config_path: str | Path,
    source_attempt: str | Path,
    *,
    cell_id: str,
    city: str,
    seed: int,
    budget: str,
    expected_revision: str | None = None,
) -> SourceEvidence:
    config = ExperimentConfig.load(config_path)
    config.cell(cell_id)
    attempt = Path(source_attempt).resolve()
    try:
        manifest = json.loads((attempt / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReevaluationEvidenceError(f"invalid source run manifest: {error}") from error
    expected = {
        "experiment_id": config.experiment_id,
        "cell_id": cell_id,
        "city": city,
        "seed": int(seed),
        "budget": budget,
        "status": "completed",
    }
    changed = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if changed:
        raise ReevaluationEvidenceError(f"source manifest identity mismatch: {changed}")
    if bool(manifest.get("git", {}).get("dirty")):
        raise ReevaluationEvidenceError("source run used a dirty worktree")
    revision = str(manifest.get("git", {}).get("revision", ""))
    if len(revision) != 40 or (expected_revision is not None and revision != expected_revision):
        raise ReevaluationEvidenceError("source revision mismatch")
    try:
        stage1_model = read_stage_artifact(attempt, "stage1", "stage1_model.pt")
        test_tokens = read_stage_artifact(attempt, "tokens", "test_tokens.jsonl")
        llm_test = read_stage_artifact(attempt, "stage2_data", "llm_test.parquet")
        adapter_manifest = read_stage_artifact(attempt, "stage2", "adapter_manifest.json")
    except Exception as error:
        raise ReevaluationEvidenceError(str(error)) from error
    adapter_payload = json.loads(adapter_manifest.read_text(encoding="utf-8"))
    adapter_dir = Path(str(adapter_payload.get("adapter", ""))).resolve()
    if not adapter_dir.is_relative_to(attempt) or adapter_dir != (attempt / "stage2" / "adapter").resolve():
        raise ReevaluationEvidenceError("adapter path escapes or differs from the source attempt")
    return SourceEvidence(
        attempt=attempt,
        revision=revision,
        stage1_model=stage1_model,
        test_tokens=test_tokens,
        llm_test=llm_test,
        adapter_manifest=adapter_manifest,
        adapter_dir=adapter_dir,
        adapter_tree=_adapter_tree(adapter_dir),
    )


def load_stratified_manifest(
    manifest_path: str | Path,
    *,
    city: str,
    seed: int,
    expected_size: int,
) -> dict[str, object]:
    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReevaluationEvidenceError(f"invalid stratified subset manifest: {error}") from error
    if payload.get("algorithm") != ALGORITHM:
        raise ReevaluationEvidenceError("unexpected stratified subset algorithm")
    if payload.get("city") != city or int(payload.get("seed", -1)) != int(seed):
        raise ReevaluationEvidenceError("stratified subset city/seed mismatch")
    ids = [int(value) for value in payload.get("selected_traj_ids", [])]
    if len(ids) != expected_size or ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ReevaluationEvidenceError("stratified subset is not the exact sorted unique population")
    if trajectory_id_sha256(ids) != payload.get("selected_traj_id_sha256"):
        raise ReevaluationEvidenceError("stratified subset trajectory hash mismatch")
    strata = payload.get("strata")
    if not isinstance(strata, Mapping) or sum(int(row["quota"]) for row in strata.values()) != expected_size:
        raise ReevaluationEvidenceError("stratified subset quota mismatch")
    return {**payload, "manifest_path": str(path.resolve()), "manifest_sha256": sha256_file(path)}


def audit_source_population(
    evidence: SourceEvidence,
    subset: Mapping[str, object],
    aligned_root: str | Path,
) -> dict[str, object]:
    root = Path(aligned_root)
    traj_ids = np.load(root / "test_traj_ids.npy", allow_pickle=False)
    offsets = np.load(root / "test_offsets.npy", allow_pickle=False)
    if len(offsets) != len(traj_ids) + 1 or np.any(np.diff(offsets) <= 0):
        raise ReevaluationEvidenceError("invalid aligned test offsets")
    lengths_by_id = {
        int(traj_id): int(length)
        for traj_id, length in zip(traj_ids.tolist(), np.diff(offsets).tolist(), strict=True)
    }
    with evidence.test_tokens.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    try:
        token_audit = verify_token_lens_mapping(records, lengths_by_id)
    except ValueError as error:
        raise ReevaluationEvidenceError(str(error)) from error
    frame = pd.read_parquet(evidence.llm_test, columns=["traj_id"])
    frame_ids = [int(value) for value in frame["traj_id"]]
    if len(set(frame_ids)) != len(frame_ids) or set(frame_ids) != set(lengths_by_id):
        raise ReevaluationEvidenceError("full LLM test frame does not match aligned test population")
    selected = [int(value) for value in subset["selected_traj_ids"]]
    if not set(selected).issubset(lengths_by_id):
        raise ReevaluationEvidenceError("stratified subset contains IDs outside the source test population")
    return {
        "source_test_population": len(lengths_by_id),
        "source_test_traj_id_sha256": trajectory_id_sha256(sorted(lengths_by_id)),
        "token_audit": token_audit,
        "llm_test_population": len(frame_ids),
        "selected_population": len(selected),
        "selected_traj_id_sha256": trajectory_id_sha256(selected),
        "lens_formula_verified": True,
    }


def write_source_reference(
    output_path: str | Path,
    evidence: SourceEvidence,
    subset: Mapping[str, object],
    population_audit: Mapping[str, object],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter_payload = json.dumps(evidence.adapter_tree, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "schema_version": 1,
        "source_attempt": str(evidence.attempt),
        "source_revision": evidence.revision,
        "source_artifacts": {
            "stage1_model": _artifact(evidence.stage1_model),
            "test_tokens": _artifact(evidence.test_tokens),
            "llm_test": _artifact(evidence.llm_test),
            "adapter_manifest": _artifact(evidence.adapter_manifest),
            "adapter_tree": list(evidence.adapter_tree),
            "adapter_tree_sha256": hashlib.sha256(adapter_payload).hexdigest(),
        },
        "subset": {
            "path": subset["manifest_path"],
            "sha256": subset["manifest_sha256"],
            "selected_traj_id_sha256": subset["selected_traj_id_sha256"],
        },
        "population_audit": dict(population_audit),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
