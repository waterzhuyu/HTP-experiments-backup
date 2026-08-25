from __future__ import annotations

import json
from pathlib import Path

import pytest

from staged_attribution.config import ExperimentConfig
from staged_attribution.run_experiment import (
    build_jobs,
    completed_attempt,
    next_attempt_dir,
    run_parallel_experiment,
    safe_gpu_indices,
)


CONFIG_PATH = Path("stage_1/configs/staged_attribution/experiment.json")


def test_smoke_queue_has_ten_full_pipeline_cells() -> None:
    config = ExperimentConfig.load(Path("stage_1/configs/staged_attribution/experiment.json"))

    jobs = build_jobs(config, budget="smoke")

    assert len(jobs) == 10
    assert {job.city for job in jobs} == {"porto"}
    assert {job.seed for job in jobs} == {42}
    assert {job.cell_id for job in jobs} == {cell.cell_id for cell in config.core_cells()}
    assert all(job.terminal_stage == "stage4" for job in jobs)


def test_controlled_queue_is_city_seed_cell_cartesian_product() -> None:
    config = ExperimentConfig.load(Path("stage_1/configs/staged_attribution/experiment.json"))

    jobs = build_jobs(config, budget="controlled")

    assert len(jobs) == 2 * 3 * 10
    assert len({job.run_key for job in jobs}) == len(jobs)


def test_confirmation_requires_explicit_baseline_and_selected_candidates() -> None:
    config = ExperimentConfig.load(CONFIG_PATH)

    with pytest.raises(ValueError, match="explicit selected cell list"):
        build_jobs(config, budget="confirmation")
    with pytest.raises(ValueError, match="include baseline"):
        build_jobs(config, budget="confirmation", cell_ids=["e6_q3"])
    with pytest.raises(ValueError, match="unknown selected"):
        build_jobs(config, budget="confirmation", cell_ids=["e0_q0", "missing"])

    jobs = build_jobs(config, budget="confirmation", cell_ids=["e0_q0", "e3_q2"])

    assert len(jobs) == 2 * 3 * 2
    assert {job.cell_id for job in jobs} == {"e0_q0", "e3_q2"}


def test_gpu_safety_uses_free_memory_and_utilization() -> None:
    rows = [
        {"index": 0, "memory_total": 24576, "memory_used": 22000, "utilization": 10},
        {"index": 1, "memory_total": 24576, "memory_used": 1000, "utilization": 95},
        {"index": 2, "memory_total": 24576, "memory_used": 1000, "utilization": 5},
    ]

    assert safe_gpu_indices(rows, minimum_free_mib=16000, maximum_utilization=20) == [2]


def test_attempt_paths_are_append_only_and_completed_attempt_is_detected(tmp_path: Path) -> None:
    config = ExperimentConfig.load(CONFIG_PATH)
    job = build_jobs(config, "smoke")[0]
    job_root = tmp_path / job.run_key
    failed = job_root / "attempt_001"
    failed.mkdir(parents=True)
    (failed / "run_manifest.json").write_text(json.dumps({"status": "failed"}))
    (job_root / "README.txt").write_text("operator note")
    (job_root / "attempt_manual").mkdir()

    assert next_attempt_dir(job_root) == job_root / "attempt_002"
    assert completed_attempt(job_root) is None

    completed = job_root / "attempt_002"
    completed.mkdir()
    (completed / "run_manifest.json").write_text(json.dumps({"status": "completed"}))
    (completed / "stage4").mkdir()
    (completed / "stage4" / "complete.json").write_text("{}")

    assert completed_attempt(job_root) == completed
    assert next_attempt_dir(job_root) == job_root / "attempt_003"


def test_parallel_queue_uses_safe_unique_gpus_and_immutable_attempts(tmp_path: Path) -> None:
    launches: list[tuple[list[str], Path]] = []

    class ImmediateProcess:
        returncode = 0

        def poll(self):
            return 0

    def fake_popen(command, cwd, stdout, stderr, text):
        launches.append((list(command), Path(stdout.name).parent))
        return ImmediateProcess()

    rows = [
        {"index": 0, "memory_total": 24576, "memory_used": 1000, "utilization": 5},
        {"index": 1, "memory_total": 24576, "memory_used": 1000, "utilization": 5},
        {"index": 2, "memory_total": 24576, "memory_used": 22000, "utilization": 0},
    ]
    result = run_parallel_experiment(
        CONFIG_PATH,
        "smoke",
        artifact_root=tmp_path,
        maximum_parallel=2,
        wait_for_gpu=False,
        query_gpu_fn=lambda: rows,
        popen_factory=fake_popen,
        sleep_fn=lambda _: None,
    )

    assert len(result["completed"]) == 10
    assert result["failed"] == []
    assert len(launches) == 10
    devices = []
    attempt_paths = []
    for command, attempt_path in launches:
        device_index = command.index("--device") + 1
        run_dir_index = command.index("--run-dir") + 1
        devices.append(command[device_index])
        attempt_paths.append(Path(command[run_dir_index]))
        assert attempt_path == Path(command[run_dir_index])
    assert set(devices) == {"cuda:0", "cuda:1"}
    assert len(set(attempt_paths)) == 10
    assert all(path.name == "attempt_001" for path in attempt_paths)
