from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import staged_attribution.select_candidates as select_candidates_module
from staged_attribution.select_candidates import SelectionError, freeze_selection
from staged_attribution.selection_launch import (
    SelectionLaunchError,
    audit_frozen_launch,
)


CONFIG_PATH = Path("stage_1/configs/staged_attribution/experiment.json")
CONFIG_SHA = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
REVISION = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: str,
    cells: list[str],
) -> tuple[Path, str]:
    source = tmp_path / f"{phase}_analysis.json"
    coverage = [
        "e0_q0", "e1_q0", "e2_q0", "e3_q0", "e4_q0",
        "e5_q0", "e6_q0", "e6_q1", "e6_q2", "e6_q3",
    ]
    source.write_text(
        json.dumps(
            {
                "candidate_evidence": "stratified_population_aware",
                "revision": REVISION,
                "coverage": {"cells": coverage},
            }
        )
    )
    deterministic = {
        "schema_version": 1,
        "phase": phase,
        "source_revision": REVISION,
        "best_single": "e1_q0",
        "best_pair": "e3_q0",
        "best_q": "e6_q2",
        "adaptive_cells": cells if phase == "core" else [],
        "pareto_eligible_cells": ["e1_q0"],
        "confirmation_cells": cells if phase == "final" else ["e0_q0"],
        "eligibility": {},
        "paired_effects": [],
    }
    def fake_select_candidates(summary, requested_phase):
        result = dict(deterministic)
        result["phase"] = requested_phase
        if requested_phase == "core" and phase == "final":
            result["adaptive_cells"] = []
            result["confirmation_cells"] = ["e0_q0"]
        return result

    monkeypatch.setattr(
        select_candidates_module,
        "select_candidates",
        fake_select_candidates,
    )
    # audit_selection_derivation resolves select_candidates in its defining module.
    selection = tmp_path / f"{phase}_selection.json"
    payload = {
        **deterministic,
        "source_analysis": str(source.resolve()),
        "source_analysis_sha256": _sha256(source),
    }
    if phase == "final":
        prior_source = tmp_path / "core_analysis.json"
        prior_source.write_text(
            json.dumps(
                {
                    "candidate_evidence": "stratified_population_aware",
                    "revision": REVISION,
                    "coverage": {"cells": coverage},
                }
            )
        )
        prior_payload = {
            **deterministic,
            "phase": "core",
            "adaptive_cells": [],
            "confirmation_cells": ["e0_q0"],
            "source_analysis": str(prior_source.resolve()),
            "source_analysis_sha256": _sha256(prior_source),
        }
        prior = tmp_path / "core_selection.json"
        prior.write_text(json.dumps(prior_payload))
        payload.update(
            {
                "prior_selection": str(prior.resolve()),
                "prior_selection_sha256": _sha256(prior),
                "prior_adaptive_cells": [],
            }
        )
    selection.write_text(json.dumps(payload))
    return selection, _sha256(selection)


def test_adaptive_launch_requires_exact_recomputed_cell_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, selection_sha = _selection_tree(
        tmp_path, monkeypatch, phase="core", cells=["e3_q2"]
    )
    identity = audit_frozen_launch(
        CONFIG_PATH,
        "controlled",
        ["e3_q2"],
        selection_path=selection,
        expected_selection_sha256=selection_sha,
        expected_revision=REVISION,
        expected_config_sha256=CONFIG_SHA,
        git_identity_fn=lambda: (REVISION, False),
    )

    assert identity["phase"] == "core"
    assert identity["cells"] == ["e3_q2"]
    assert identity["verified"] is True
    with pytest.raises(SelectionLaunchError, match="exactly match"):
        audit_frozen_launch(
            CONFIG_PATH,
            "controlled",
            ["e4_q2"],
            selection_path=selection,
            expected_selection_sha256=selection_sha,
            expected_revision=REVISION,
            expected_config_sha256=CONFIG_SHA,
            git_identity_fn=lambda: (REVISION, False),
        )


def test_launch_rejects_dirty_revision_config_or_selection_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, selection_sha = _selection_tree(
        tmp_path, monkeypatch, phase="core", cells=["e3_q2"]
    )
    common = dict(
        config_path=CONFIG_PATH,
        budget="controlled",
        cell_ids=["e3_q2"],
        selection_path=selection,
        expected_selection_sha256=selection_sha,
        expected_revision=REVISION,
        expected_config_sha256=CONFIG_SHA,
    )
    with pytest.raises(SelectionLaunchError, match="clean"):
        audit_frozen_launch(**common, git_identity_fn=lambda: (REVISION, True))
    with pytest.raises(SelectionLaunchError, match="revision mismatch"):
        audit_frozen_launch(**common, git_identity_fn=lambda: ("c" * 40, False))
    with pytest.raises(SelectionLaunchError, match="config SHA"):
        audit_frozen_launch(
            **{**common, "expected_config_sha256": "0" * 64},
            git_identity_fn=lambda: (REVISION, False),
        )
    with pytest.raises(SelectionLaunchError, match="hash-mismatched"):
        audit_frozen_launch(
            **{**common, "expected_selection_sha256": "0" * 64},
            git_identity_fn=lambda: (REVISION, False),
        )


def test_confirmation_launch_requires_final_chain_and_baseline_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = ["e0_q0", "e1_q0"]
    selection, selection_sha = _selection_tree(
        tmp_path, monkeypatch, phase="final", cells=cells
    )
    identity = audit_frozen_launch(
        CONFIG_PATH,
        "confirmation",
        cells,
        selection_path=selection,
        expected_selection_sha256=selection_sha,
        expected_revision=REVISION,
        expected_config_sha256=CONFIG_SHA,
        git_identity_fn=lambda: (REVISION, False),
    )
    assert identity["phase"] == "final"

    with pytest.raises(SelectionLaunchError, match="exactly match|baseline"):
        audit_frozen_launch(
            CONFIG_PATH,
            "confirmation",
            ["e1_q0", "e0_q0"],
            selection_path=selection,
            expected_selection_sha256=selection_sha,
            expected_revision=REVISION,
            expected_config_sha256=CONFIG_SHA,
            git_identity_fn=lambda: (REVISION, False),
        )


def test_final_freeze_binds_prior_core_selection_and_exact_adaptive_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior, prior_sha = _selection_tree(
        tmp_path, monkeypatch, phase="core", cells=["e3_q2"]
    )
    prior_payload = json.loads(prior.read_text())
    derived_fields = {
        key: value
        for key, value in prior_payload.items()
        if key not in {"source_analysis", "source_analysis_sha256"}
    }
    final_derived = {
        **derived_fields,
        "phase": "final",
        "adaptive_cells": [],
        "confirmation_cells": ["e0_q0", "e3_q2"],
    }

    def fake_select(summary, requested_phase):
        return dict(derived_fields if requested_phase == "core" else final_derived)

    monkeypatch.setattr(select_candidates_module, "select_candidates", fake_select)
    core_cells = [
        "e0_q0", "e1_q0", "e2_q0", "e3_q0", "e4_q0",
        "e5_q0", "e6_q0", "e6_q1", "e6_q2", "e6_q3",
    ]
    final_analysis = tmp_path / "final_analysis.json"
    final_analysis.write_text(
        json.dumps(
            {
                "candidate_evidence": "stratified_population_aware",
                "revision": REVISION,
                "coverage": {"cells": [*core_cells, "e3_q2"]},
            }
        )
    )
    frozen = freeze_selection(
        final_analysis,
        "final",
        prior_selection_path=prior,
        expected_prior_selection_sha256=prior_sha,
    )

    assert frozen["prior_selection"] == str(prior.resolve())
    assert frozen["prior_selection_sha256"] == prior_sha
    assert frozen["prior_adaptive_cells"] == ["e3_q2"]

    final_analysis.write_text(
        json.dumps(
            {
                "candidate_evidence": "stratified_population_aware",
                "revision": REVISION,
                "coverage": {"cells": core_cells},
            }
        )
    )
    with pytest.raises(SelectionError, match="core plus frozen adaptive"):
        freeze_selection(
            final_analysis,
            "final",
            prior_selection_path=prior,
            expected_prior_selection_sha256=prior_sha,
        )
