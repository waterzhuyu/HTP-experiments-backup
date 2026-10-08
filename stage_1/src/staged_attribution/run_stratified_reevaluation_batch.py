from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import ExperimentConfig
from .run_experiment import (
    ExperimentJob,
    build_jobs,
    completed_attempt,
    next_attempt_dir,
    query_gpus,
    safe_gpu_indices,
)


class ReevaluationBatchError(RuntimeError):
    """Raised when a corrected reevaluation queue would be incomplete or duplicated."""


def completed_reevaluation_attempt(job_root: str | Path) -> Path | None:
    root = Path(job_root)
    if not root.is_dir():
        return None
    attempts = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("attempt_")),
        reverse=True,
    )
    for attempt in attempts:
        manifest_path = attempt / "reevaluation_manifest.json"
        marker = attempt / "stage4" / "complete.json"
        if not manifest_path.is_file() or not marker.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "completed":
            return attempt
    return None


def _subset_path(subset_dir: Path, job: ExperimentJob) -> Path:
    return subset_dir / f"controlled_{job.city}_seed_{job.seed}.json"


def _command(
    config_path: str | Path,
    job: ExperimentJob,
    source_attempt: Path,
    subset_manifest: Path,
    run_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "staged_attribution.run_stratified_reevaluation",
        "--config", str(config_path),
        "--source-attempt", str(source_attempt),
        "--subset-manifest", str(subset_manifest),
        "--cell", job.cell_id,
        "--city", job.city,
        "--seed", str(job.seed),
        "--budget", "controlled",
        "--run-dir", str(run_dir),
        "--device", "cuda:0",
    ]


def run_parallel_stratified_reevaluations(
    config_path: str | Path,
    *,
    source_root: str | Path,
    reevaluation_root: str | Path,
    subset_dir: str | Path,
    maximum_parallel: int = 4,
    wait_for_gpu: bool = True,
    poll_seconds: float = 60.0,
    minimum_free_mib: int = 16000,
    maximum_utilization: int = 20,
    cell_ids: Sequence[str] | None = None,
    query_gpu_fn: Callable[[], list[dict[str, int]]] = query_gpus,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    if maximum_parallel <= 0 or poll_seconds < 0:
        raise ValueError("invalid parallel or polling settings")
    config = ExperimentConfig.load(config_path)
    jobs = list(build_jobs(config, "controlled", cell_ids))
    source = Path(source_root)
    target = Path(reevaluation_root)
    subsets = Path(subset_dir)
    source_attempts: dict[str, Path] = {}
    missing_source = []
    missing_subset = []
    for job in jobs:
        attempt = completed_attempt(source / job.run_key)
        subset = _subset_path(subsets, job)
        if attempt is None:
            missing_source.append(job.run_key)
        else:
            source_attempts[job.run_key] = attempt
        if not subset.is_file():
            missing_subset.append(str(subset))
    if missing_source or missing_subset:
        raise ReevaluationBatchError(
            f"preflight requires complete source/subset Cartesian evidence: "
            f"missing_source={missing_source[:10]} missing_subset={missing_subset[:10]}"
        )

    completed: list[dict[str, object]] = []
    pending: list[ExperimentJob] = []
    for job in jobs:
        existing = completed_reevaluation_attempt(target / job.run_key)
        if existing is None:
            pending.append(job)
        else:
            completed.append({"run_key": job.run_key, "attempt": str(existing), "status": "existing"})
    running: dict[int, dict[str, object]] = {}
    failed: list[dict[str, object]] = []
    while pending or running:
        for gpu_index, state in list(running.items()):
            returncode = state["process"].poll()
            if returncode is None:
                continue
            state["log_handle"].close()
            record = {
                "run_key": state["job"].run_key,
                "attempt": str(state["attempt"]),
                "gpu": gpu_index,
                "returncode": int(returncode),
            }
            (completed if returncode == 0 else failed).append(record)
            del running[gpu_index]
        if failed:
            if running:
                sleep_fn(poll_seconds)
                continue
            break
        safe = safe_gpu_indices(query_gpu_fn(), minimum_free_mib, maximum_utilization)
        available = [index for index in safe if index not in running]
        slots = max(0, maximum_parallel - len(running))
        for gpu_index in available[:slots]:
            if not pending:
                break
            job = pending.pop(0)
            attempt = next_attempt_dir(target / job.run_key)
            attempt.mkdir(parents=True, exist_ok=False)
            log_handle = (attempt / "launcher.log").open("a", encoding="utf-8")
            command = _command(
                config_path, job, source_attempts[job.run_key],
                _subset_path(subsets, job), attempt,
            )
            process = popen_factory(
                command,
                cwd=Path.cwd(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)},
            )
            running[gpu_index] = {
                "job": job, "attempt": attempt, "process": process,
                "log_handle": log_handle,
            }
        if pending or running:
            if not running and not wait_for_gpu:
                raise ReevaluationBatchError("no GPU satisfies the non-preemption threshold")
            sleep_fn(poll_seconds)
    return {
        "experiment_id": config.experiment_id,
        "reevaluation": "fixed_stratified_population_aware_v1",
        "completed": completed,
        "failed": failed,
        "pending": [job.run_key for job in pending],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the corrected stratified reevaluation queue")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reevaluation-root", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--maximum-parallel", type=int, default=4)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--minimum-free-mib", type=int, default=16000)
    parser.add_argument("--maximum-utilization", type=int, default=20)
    parser.add_argument("--cells", nargs="+")
    arguments = parser.parse_args()
    result = run_parallel_stratified_reevaluations(
        arguments.config,
        source_root=arguments.source_root,
        reevaluation_root=arguments.reevaluation_root,
        subset_dir=arguments.subset_dir,
        maximum_parallel=arguments.maximum_parallel,
        wait_for_gpu=arguments.wait_for_gpu,
        poll_seconds=arguments.poll_seconds,
        minimum_free_mib=arguments.minimum_free_mib,
        maximum_utilization=arguments.maximum_utilization,
        cell_ids=arguments.cells,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
