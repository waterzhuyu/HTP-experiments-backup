from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from staged_attribution.run_stratified_reevaluation_batch import (
    ReevaluationBatchError,
    _exclusive_batch_lock,
    run_parallel_stratified_reevaluations,
)


CONFIG_PATH = Path("stage_1/configs/staged_attribution/experiment.json")
EVALUATION_REVISION = "a" * 40
SOURCE_REVISION = "b" * 40
CONFIG_SHA256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_arguments(audit: Path, audit_sha256: str) -> dict[str, object]:
    return {
        "source_audit_path": audit,
        "expected_revision": EVALUATION_REVISION,
        "expected_config_sha256": CONFIG_SHA256,
        "expected_source_audit_sha256": audit_sha256,
        "git_identity_fn": lambda: (EVALUATION_REVISION, False),
    }


def _source_and_subsets(tmp_path: Path):
    source = tmp_path / "source"
    subsets = tmp_path / "subsets"
    subsets.mkdir()
    rows = []
    run_keys = []
    for city in ("porto", "chengdu"):
        for seed in (42, 314159, 20260819):
            run_key = f"controlled/{city}/seed_{seed}/e0_q0"
            attempt = source / "controlled" / city / f"seed_{seed}" / "e0_q0" / "attempt_001"
            (attempt / "stage4").mkdir(parents=True)
            (attempt / "stage4" / "complete.json").write_text("{}")
            (attempt / "run_manifest.json").write_text(json.dumps({"status": "completed"}))
            subset = subsets / f"controlled_{city}_seed_{seed}.json"
            subset.write_text("{}")
            run_keys.append(run_key)
            rows.append(
                {
                    "run_key": run_key,
                    "cell_id": "e0_q0",
                    "city": city,
                    "seed": seed,
                    "budget": "controlled",
                    "source_attempt": str(attempt.resolve()),
                    "source_revision": SOURCE_REVISION,
                    "fixed_subset": {
                        "path": str(subset.resolve()),
                        "sha256": _sha256(subset),
                    },
                }
            )
    index = subsets / "index.json"
    index.write_text(json.dumps({"kind": "fixed_stratified_reevaluation_subsets"}))
    audit_payload = {
        "schema_version": 1,
        "kind": "completed_formal_source_evidence_audit",
        "experiment_id": json.loads(CONFIG_PATH.read_text())["experiment_id"],
        "budget": "controlled",
        "config": {
            "path": str(CONFIG_PATH.resolve()),
            "size_bytes": CONFIG_PATH.stat().st_size,
            "sha256": CONFIG_SHA256,
        },
        "generator_git": {
            "worktree": "/synthetic/worktree",
            "revision": EVALUATION_REVISION,
            "dirty": False,
        },
        "source_root": str(source.resolve()),
        "fixed_subset_index": {
            "path": str(index.resolve()),
            "size_bytes": index.stat().st_size,
            "sha256": _sha256(index),
        },
        "requested_cell_ids": ["e0_q0"],
        "requested_job_count": 6,
        "requested_run_key_sha256": hashlib.sha256(
            "".join(f"{run_key}\n" for run_key in sorted(run_keys)).encode()
        ).hexdigest(),
        "all_requested_sources_completed": True,
        "audited_completed_source_count": 6,
        "missing_completed_source_count": 0,
        "audited_completed_sources": rows,
        "missing_completed_sources": [],
    }
    audit = tmp_path / "source_audit.json"
    audit.write_text(json.dumps(audit_payload, sort_keys=True))
    return source, subsets, audit, _sha256(audit)


class ImmediateProcess:
    def poll(self):
        return 0


class PopenRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        run_dir = Path(command[command.index("--run-dir") + 1])
        (run_dir / "stage4").mkdir(parents=True, exist_ok=True)
        (run_dir / "stage4" / "complete.json").write_text("{}")
        config = Path(command[command.index("--config") + 1])
        (run_dir / "reevaluation_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "experiment_id": json.loads(config.read_text())["experiment_id"],
                    "reevaluation": "fixed_stratified_population_aware_v1",
                    "cell_id": command[command.index("--cell") + 1],
                    "city": command[command.index("--city") + 1],
                    "seed": int(command[command.index("--seed") + 1]),
                    "budget": command[command.index("--budget") + 1],
                    "config_sha256": command[
                        command.index("--expected-config-sha256") + 1
                    ],
                    "expected_source_revision": command[
                        command.index("--expected-source-revision") + 1
                    ],
                    "source_attempt": str(
                        Path(
                            command[command.index("--source-attempt") + 1]
                        ).resolve()
                    ),
                    "subset_manifest": str(
                        Path(
                            command[command.index("--subset-manifest") + 1]
                        ).resolve()
                    ),
                    "source_audit_path": str(
                        Path(command[command.index("--source-audit") + 1]).resolve()
                    ),
                    "source_audit_sha256": command[
                        command.index("--expected-source-audit-sha256") + 1
                    ],
                    "git": {
                        "revision": command[
                            command.index("--expected-revision") + 1
                        ],
                        "dirty": False,
                    },
                }
            )
        )
        return ImmediateProcess()


def gpus():
    return [
        {"index": index, "memory_total": 24576, "memory_used": 0, "utilization": 0}
        for index in range(4)
    ]


def test_batch_requires_all_source_and_subset_evidence_before_launch(tmp_path: Path) -> None:
    source, subsets, audit, audit_sha256 = _source_and_subsets(tmp_path)
    (subsets / "controlled_porto_seed_42.json").unlink()
    recorder = PopenRecorder()
    with pytest.raises(ReevaluationBatchError, match="preflight"):
        run_parallel_stratified_reevaluations(
            CONFIG_PATH, source_root=source, reevaluation_root=tmp_path / "target",
            subset_dir=subsets, cell_ids=["e0_q0"], query_gpu_fn=gpus,
            popen_factory=recorder, sleep_fn=lambda _: None,
            **_identity_arguments(audit, audit_sha256),
        )
    assert recorder.calls == []


def test_batch_refuses_duplicate_queue_while_lock_is_held(tmp_path: Path) -> None:
    source, subsets, audit, audit_sha256 = _source_and_subsets(tmp_path)
    recorder = PopenRecorder()
    target = tmp_path / "target"
    with _exclusive_batch_lock(target):
        with pytest.raises(ReevaluationBatchError, match="another corrected reevaluation"):
            run_parallel_stratified_reevaluations(
                CONFIG_PATH,
                source_root=source,
                reevaluation_root=target,
                subset_dir=subsets,
                cell_ids=["e0_q0"],
                query_gpu_fn=gpus,
                popen_factory=recorder,
                sleep_fn=lambda _: None,
                **_identity_arguments(audit, audit_sha256),
            )
    assert recorder.calls == []


def test_batch_lock_survives_parent_close_while_inherited_child_runs(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    with _exclusive_batch_lock(target) as (_, descriptor):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.5)"],
            pass_fds=(descriptor,),
        )
    try:
        with pytest.raises(ReevaluationBatchError, match="another corrected reevaluation"):
            with _exclusive_batch_lock(target):
                pass
    finally:
        child.wait(timeout=5)

    with _exclusive_batch_lock(target):
        pass


def test_batch_launches_one_queue_with_exact_source_subset_and_logical_gpu(tmp_path: Path) -> None:
    source, subsets, audit, audit_sha256 = _source_and_subsets(tmp_path)
    recorder = PopenRecorder()
    target = tmp_path / "target"
    result = run_parallel_stratified_reevaluations(
        CONFIG_PATH, source_root=source, reevaluation_root=target,
        subset_dir=subsets, cell_ids=["e0_q0"], maximum_parallel=2,
        poll_seconds=0, query_gpu_fn=gpus, popen_factory=recorder,
        sleep_fn=lambda _: None,
        **_identity_arguments(audit, audit_sha256),
    )

    assert len(recorder.calls) == 6
    assert len(result["completed"]) == 6
    assert Path(result["batch_lock"]).is_file()
    assert not result["failed"] and not result["pending"]
    for command, kwargs in recorder.calls:
        assert "staged_attribution.run_stratified_reevaluation" in command
        assert command[command.index("--device") + 1] == "cuda:0"
        source_attempt = Path(command[command.index("--source-attempt") + 1])
        subset = Path(command[command.index("--subset-manifest") + 1])
        assert source_attempt.name == "attempt_001"
        assert subset.is_file()
        assert command[command.index("--expected-revision") + 1] == EVALUATION_REVISION
        assert command[command.index("--expected-config-sha256") + 1] == CONFIG_SHA256
        assert command[command.index("--expected-source-revision") + 1] == SOURCE_REVISION
        assert Path(command[command.index("--source-audit") + 1]) == audit.resolve()
        assert command[
            command.index("--expected-source-audit-sha256") + 1
        ] == audit_sha256
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] in {"0", "1", "2", "3"}

    second = PopenRecorder()
    repeated = run_parallel_stratified_reevaluations(
        CONFIG_PATH, source_root=source, reevaluation_root=target,
        subset_dir=subsets, cell_ids=["e0_q0"], query_gpu_fn=gpus,
        popen_factory=second, sleep_fn=lambda _: None,
        **_identity_arguments(audit, audit_sha256),
    )
    assert second.calls == []
    assert len(repeated["completed"]) == 6

    stale = target / "controlled/porto/seed_42/e0_q0/attempt_001"
    manifest_path = stale / "reevaluation_manifest.json"
    stale_manifest = json.loads(manifest_path.read_text())
    stale_manifest["git"]["revision"] = "f" * 40
    manifest_path.write_text(json.dumps(stale_manifest))
    replacement = PopenRecorder()
    repaired = run_parallel_stratified_reevaluations(
        CONFIG_PATH,
        source_root=source,
        reevaluation_root=target,
        subset_dir=subsets,
        cell_ids=["e0_q0"],
        query_gpu_fn=gpus,
        popen_factory=replacement,
        sleep_fn=lambda _: None,
        **_identity_arguments(audit, audit_sha256),
    )
    assert len(replacement.calls) == 1
    replacement_dir = Path(
        replacement.calls[0][0][replacement.calls[0][0].index("--run-dir") + 1]
    )
    assert replacement_dir.name == "attempt_002"
    assert len(repaired["completed"]) == 6


def test_batch_identity_gates_fail_before_target_or_process_creation(
    tmp_path: Path,
) -> None:
    source, subsets, audit, audit_sha256 = _source_and_subsets(tmp_path)
    recorder = PopenRecorder()

    cases = (
        {
            **_identity_arguments(audit, audit_sha256),
            "expected_revision": "c" * 40,
        },
        {
            **_identity_arguments(audit, audit_sha256),
            "expected_config_sha256": "0" * 64,
        },
        {
            **_identity_arguments(audit, audit_sha256),
            "expected_source_audit_sha256": "0" * 64,
        },
    )
    for index, identity in enumerate(cases):
        target = tmp_path / f"rejected_{index}"
        with pytest.raises(ReevaluationBatchError, match="mismatch"):
            run_parallel_stratified_reevaluations(
                CONFIG_PATH,
                source_root=source,
                reevaluation_root=target,
                subset_dir=subsets,
                cell_ids=["e0_q0"],
                query_gpu_fn=gpus,
                popen_factory=recorder,
                sleep_fn=lambda _: None,
                **identity,
            )
        assert not target.exists()
    assert recorder.calls == []


def test_batch_rejects_rehashed_audit_with_wrong_cell_population(
    tmp_path: Path,
) -> None:
    source, subsets, audit, _ = _source_and_subsets(tmp_path)
    payload = json.loads(audit.read_text())
    payload["requested_cell_ids"] = ["e1_q0"]
    audit.write_text(json.dumps(payload, sort_keys=True))
    audit_sha256 = _sha256(audit)
    target = tmp_path / "target"
    recorder = PopenRecorder()

    with pytest.raises(ReevaluationBatchError, match="cell population mismatch"):
        run_parallel_stratified_reevaluations(
            CONFIG_PATH,
            source_root=source,
            reevaluation_root=target,
            subset_dir=subsets,
            cell_ids=["e0_q0"],
            query_gpu_fn=gpus,
            popen_factory=recorder,
            sleep_fn=lambda _: None,
            **_identity_arguments(audit, audit_sha256),
        )
    assert not target.exists()
    assert recorder.calls == []
