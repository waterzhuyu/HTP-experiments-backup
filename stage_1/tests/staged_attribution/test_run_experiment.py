from __future__ import annotations

from pathlib import Path

from staged_attribution.config import ExperimentConfig
from staged_attribution.run_experiment import build_jobs, safe_gpu_indices


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


def test_gpu_safety_uses_free_memory_and_utilization() -> None:
    rows = [
        {"index": 0, "memory_total": 24576, "memory_used": 22000, "utilization": 10},
        {"index": 1, "memory_total": 24576, "memory_used": 1000, "utilization": 95},
        {"index": 2, "memory_total": 24576, "memory_used": 1000, "utilization": 5},
    ]

    assert safe_gpu_indices(rows, minimum_free_mib=16000, maximum_utilization=20) == [2]
