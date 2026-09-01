from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .config import ExperimentConfig


CORE_CELLS = (
    "e0_q0", "e1_q0", "e2_q0", "e3_q0", "e4_q0", "e5_q0",
    "e6_q0", "e6_q1", "e6_q2", "e6_q3",
)


class SelectionLaunchError(RuntimeError):
    """Raised before a phase launch can escape its frozen selection contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: str, length: int, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise SelectionLaunchError(f"{label} must be a full lowercase hex digest")
    return normalized


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


def requires_frozen_selection(
    config: ExperimentConfig,
    budget: str,
    cell_ids: Sequence[str] | None,
) -> bool:
    if budget == "confirmation":
        return True
    if budget != "controlled" or cell_ids is None:
        return False
    core = {cell.cell_id for cell in config.core_cells()}
    return any(str(cell_id) not in core for cell_id in cell_ids)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionLaunchError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionLaunchError(f"{label} must be a JSON object: {path}")
    return value


def _audit_final_chain(selection: Mapping[str, object]) -> None:
    from .select_candidates import SelectionError, audit_selection_derivation

    prior_path = selection.get("prior_selection")
    prior_sha = selection.get("prior_selection_sha256")
    prior_cells = selection.get("prior_adaptive_cells")
    if not isinstance(prior_path, str) or not isinstance(prior_sha, str):
        raise SelectionLaunchError("final selection lacks its frozen core-selection identity")
    try:
        prior = audit_selection_derivation(prior_path, prior_sha, "core")
    except SelectionError as error:
        raise SelectionLaunchError(str(error)) from error
    if prior_cells != prior.get("adaptive_cells"):
        raise SelectionLaunchError("final/core adaptive-cell identity mismatch")
    source_path = Path(str(selection.get("source_analysis", ""))).resolve()
    source = _read_json(source_path, "final-selection source analysis")
    coverage = source.get("coverage")
    observed = coverage.get("cells") if isinstance(coverage, Mapping) else None
    expected = set(CORE_CELLS) | {str(value) for value in prior.get("adaptive_cells", [])}
    if not isinstance(observed, list) or set(map(str, observed)) != expected:
        raise SelectionLaunchError(
            "final analysis coverage must exactly equal core plus frozen adaptive cells"
        )


def audit_frozen_launch(
    config_path: str | Path,
    budget: str,
    cell_ids: Sequence[str],
    *,
    selection_path: str | Path,
    expected_selection_sha256: str,
    expected_revision: str,
    expected_config_sha256: str,
    git_identity_fn: Callable[[], tuple[str, bool]] = _git_identity,
) -> dict[str, object]:
    from .select_candidates import SelectionError, audit_selection_derivation

    selected = tuple(str(value) for value in cell_ids)
    if not selected or len(set(selected)) != len(selected):
        raise SelectionLaunchError("phase launch cell list must be non-empty and unique")
    phase = "final" if budget == "confirmation" else "core"
    selection_sha = _digest(
        expected_selection_sha256, 64, "expected selection SHA-256"
    )
    revision = _digest(expected_revision, 40, "expected launch revision")
    config_sha = _digest(expected_config_sha256, 64, "expected config SHA-256")
    actual_revision, dirty = git_identity_fn()
    if actual_revision != revision:
        raise SelectionLaunchError("launch implementation revision mismatch")
    if dirty:
        raise SelectionLaunchError("phase launch requires a clean Git worktree")
    resolved_config = Path(config_path).resolve()
    if not resolved_config.is_file() or resolved_config.is_symlink():
        raise SelectionLaunchError("phase config must be an existing non-symlink file")
    if _sha256(resolved_config) != config_sha:
        raise SelectionLaunchError("phase config SHA-256 mismatch")
    try:
        selection = audit_selection_derivation(selection_path, selection_sha, phase)
    except SelectionError as error:
        raise SelectionLaunchError(str(error)) from error
    if selection.get("source_revision") != revision:
        raise SelectionLaunchError(
            "selection analysis revision differs from the launch implementation revision"
        )
    field = "confirmation_cells" if phase == "final" else "adaptive_cells"
    declared = selection.get(field)
    if not isinstance(declared, list) or tuple(map(str, declared)) != selected:
        raise SelectionLaunchError(
            f"requested cells do not exactly match frozen {phase} selection"
        )
    if phase == "final":
        if selected[0] != "e0_q0":
            raise SelectionLaunchError("confirmation launch must begin with baseline e0_q0")
        _audit_final_chain(selection)
    else:
        core = set(CORE_CELLS)
        if any(cell in core for cell in selected):
            raise SelectionLaunchError("adaptive launch may contain only non-core cells")
    selection_file = Path(selection_path).resolve()
    source_file = Path(str(selection["source_analysis"])).resolve()
    return {
        "schema_version": 1,
        "kind": "frozen_phase_launch_identity",
        "phase": phase,
        "cells": list(selected),
        "selection_path": str(selection_file),
        "selection_sha256": selection_sha,
        "source_analysis": str(source_file),
        "source_analysis_sha256": str(selection["source_analysis_sha256"]),
        "analysis_revision": str(selection["source_revision"]),
        "config_path": str(resolved_config),
        "config_sha256": config_sha,
        "launch_revision": revision,
        "verified": True,
    }
