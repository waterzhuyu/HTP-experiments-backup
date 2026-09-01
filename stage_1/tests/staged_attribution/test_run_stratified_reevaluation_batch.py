from __future__ import annotations

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


def _source_and_subsets(tmp_path: Path):
    source = tmp_path / "source"
    subsets = tmp_path / "subsets"
    subsets.mkdir()
    for city in ("porto", "chengdu"):
        for seed in (42, 314159, 20260819):
            attempt = source / "controlled" / city / f"seed_{seed}" / "e0_q0" / "attempt_001"
            (attempt / "stage4").mkdir(parents=True)
            (attempt / "stage4" / "complete.json").write_text("{}")
            (attempt / "run_manifest.json").write_text(json.dumps({"status": "completed"}))
            (subsets / f"controlled_{city}_seed_{seed}.json").write_text("{}")
    return source, subsets


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
        (run_dir / "reevaluation_manifest.json").write_text(
            json.dumps({"status": "completed"})
        )
        return ImmediateProcess()


def gpus():
    return [
        {"index": index, "memory_total": 24576, "memory_used": 0, "utilization": 0}
        for index in range(4)
    ]


def test_batch_requires_all_source_and_subset_evidence_before_launch(tmp_path: Path) -> None:
    source, subsets = _source_and_subsets(tmp_path)
    (subsets / "controlled_porto_seed_42.json").unlink()
    recorder = PopenRecorder()
    with pytest.raises(ReevaluationBatchError, match="preflight"):
        run_parallel_stratified_reevaluations(
            CONFIG_PATH, source_root=source, reevaluation_root=tmp_path / "target",
            subset_dir=subsets, cell_ids=["e0_q0"], query_gpu_fn=gpus,
            popen_factory=recorder, sleep_fn=lambda _: None,
        )
    assert recorder.calls == []


def test_batch_refuses_duplicate_queue_while_lock_is_held(tmp_path: Path) -> None:
    source, subsets = _source_and_subsets(tmp_path)
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
    source, subsets = _source_and_subsets(tmp_path)
    recorder = PopenRecorder()
    target = tmp_path / "target"
    result = run_parallel_stratified_reevaluations(
        CONFIG_PATH, source_root=source, reevaluation_root=target,
        subset_dir=subsets, cell_ids=["e0_q0"], maximum_parallel=2,
        poll_seconds=0, query_gpu_fn=gpus, popen_factory=recorder,
        sleep_fn=lambda _: None,
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
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] in {"0", "1", "2", "3"}

    second = PopenRecorder()
    repeated = run_parallel_stratified_reevaluations(
        CONFIG_PATH, source_root=source, reevaluation_root=target,
        subset_dir=subsets, cell_ids=["e0_q0"], query_gpu_fn=gpus,
        popen_factory=second, sleep_fn=lambda _: None,
    )
    assert second.calls == []
    assert len(repeated["completed"]) == 6
