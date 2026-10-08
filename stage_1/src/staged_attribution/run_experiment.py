from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import ExperimentConfig
from .run_cell import run_cell


@dataclass(frozen=True)
class ExperimentJob:
    cell_id: str
    city: str
    seed: int
    budget: str
    terminal_stage: str = "stage4"

    @property
    def run_key(self) -> str:
        return f"{self.budget}/{self.city}/seed_{self.seed}/{self.cell_id}"


def build_jobs(
    config: ExperimentConfig,
    budget: str,
    cell_ids: Sequence[str] | None = None,
) -> tuple[ExperimentJob, ...]:
    if budget not in config.budgets:
        raise ValueError(f"unknown budget {budget}")
    core_cells = config.core_cells()
    if budget == "confirmation" and cell_ids is None:
        raise ValueError("confirmation requires an explicit selected cell list")
    if cell_ids is None:
        cells = core_cells
    else:
        requested = tuple(cell_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("selected cell list must be non-empty and unique")
        unknown = []
        resolved = []
        for cell_id in requested:
            try:
                resolved.append(config.cell(cell_id))
            except ValueError:
                unknown.append(cell_id)
        if unknown:
            raise ValueError(f"unknown selected cells: {sorted(unknown)}")
        if budget == "confirmation" and "e0_q0" not in requested:
            raise ValueError("confirmation selection must include baseline e0_q0")
        cells = tuple(resolved)
    if budget == "smoke":
        cities = config.cities[:1]
        seeds = config.seeds[:1]
    else:
        cities = config.cities
        seeds = config.seeds
    return tuple(
        ExperimentJob(cell.cell_id, city, seed, budget)
        for city in cities
        for seed in seeds
        for cell in cells
    )


def safe_gpu_indices(
    rows: Sequence[Mapping[str, int]],
    minimum_free_mib: int,
    maximum_utilization: int,
) -> list[int]:
    result = []
    for row in rows:
        free = int(row["memory_total"]) - int(row["memory_used"])
        if free >= minimum_free_mib and int(row["utilization"]) <= maximum_utilization:
            result.append(int(row["index"]))
    return sorted(result)


def query_gpus() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        index, total, used, utilization = [int(value.strip()) for value in line.split(",")]
        rows.append(
            {"index": index, "memory_total": total, "memory_used": used, "utilization": utilization}
        )
    return rows


def _attempt_number(path: Path) -> int | None:
    prefix = "attempt_"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    suffix = path.name[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def _attempts(job_root: str | Path) -> list[Path]:
    root = Path(job_root)
    if not root.is_dir():
        return []
    numbered = [(_attempt_number(path), path) for path in root.iterdir()]
    valid = [(number, path) for number, path in numbered if number is not None]
    return [path for _, path in sorted(valid)]


def next_attempt_dir(job_root: str | Path) -> Path:
    root = Path(job_root)
    attempts = _attempts(root)
    number = max((_attempt_number(path) or 0 for path in attempts), default=0) + 1
    return root / f"attempt_{number:03d}"


def completed_attempt(job_root: str | Path) -> Path | None:
    for attempt in reversed(_attempts(job_root)):
        manifest_path = attempt / "run_manifest.json"
        terminal_marker = attempt / "stage4" / "complete.json"
        if not manifest_path.is_file() or not terminal_marker.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "completed":
            return attempt
    return None


def _cell_command(
    config_path: str | Path,
    job: ExperimentJob,
    attempt: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "staged_attribution.run_cell",
        "--config",
        str(config_path),
        "--cell",
        job.cell_id,
        "--city",
        job.city,
        "--seed",
        str(job.seed),
        "--budget",
        job.budget,
        "--run-dir",
        str(attempt),
        "--device",
        "cuda:0",
    ]


def run_parallel_experiment(
    config_path: str | Path,
    budget: str,
    artifact_root: str | Path | None = None,
    maximum_parallel: int = 4,
    wait_for_gpu: bool = True,
    poll_seconds: float = 60.0,
    minimum_free_mib: int = 16000,
    maximum_utilization: int = 20,
    query_gpu_fn: Callable[[], list[dict[str, int]]] = query_gpus,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    cell_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    if maximum_parallel <= 0:
        raise ValueError("maximum_parallel must be positive")
    if poll_seconds < 0:
        raise ValueError("poll_seconds cannot be negative")
    config = ExperimentConfig.load(config_path)
    jobs = list(build_jobs(config, budget, cell_ids))
    root = (
        Path(artifact_root)
        if artifact_root is not None
        else config.paths["artifact_root"] / "staged_attribution" / config.experiment_id
    )
    completed: list[dict[str, object]] = []
    pending: list[ExperimentJob] = []
    for job in jobs:
        existing = completed_attempt(root / job.run_key)
        if existing is None:
            pending.append(job)
        else:
            completed.append({"run_key": job.run_key, "attempt": str(existing), "status": "existing"})

    running: dict[int, dict[str, object]] = {}
    failed: list[dict[str, object]] = []
    while pending or running:
        for gpu_index, state in list(running.items()):
            process = state["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            log_handle = state["log_handle"]
            log_handle.close()
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
            attempt = next_attempt_dir(root / job.run_key)
            attempt.mkdir(parents=True, exist_ok=False)
            log_handle = (attempt / "launcher.log").open("a", encoding="utf-8")
            command = _cell_command(config_path, job, attempt)
            process = popen_factory(
                command,
                cwd=Path.cwd(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)},
            )
            running[gpu_index] = {
                "job": job,
                "attempt": attempt,
                "process": process,
                "log_handle": log_handle,
            }

        if pending or running:
            if not running and not wait_for_gpu:
                raise RuntimeError("no GPU satisfies the configured non-preemption safety threshold")
            sleep_fn(poll_seconds)

    return {
        "experiment_id": config.experiment_id,
        "budget": budget,
        "completed": completed,
        "failed": failed,
        "pending": [job.run_key for job in pending],
    }


def run_experiment(
    config_path: str | Path,
    budget: str,
    dry_run: bool,
    minimum_free_mib: int = 16000,
    maximum_utilization: int = 20,
    cell_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    config = ExperimentConfig.load(config_path)
    jobs = build_jobs(config, budget, cell_ids)
    if dry_run:
        return [
            {
                "run_key": job.run_key,
                "cell_id": job.cell_id,
                "city": job.city,
                "seed": job.seed,
                "budget": budget,
                "terminal_stage": job.terminal_stage,
            }
            for job in jobs
        ]
    safe = safe_gpu_indices(query_gpus(), minimum_free_mib, maximum_utilization)
    if not safe:
        raise RuntimeError("no GPU satisfies the configured non-preemption safety threshold")
    results = []
    root = config.paths["artifact_root"] / "staged_attribution" / config.experiment_id
    for index, job in enumerate(jobs):
        device = f"cuda:{safe[index % len(safe)]}"
        run_dir = root / job.run_key
        results.append(
            run_cell(
                config_path,
                job.cell_id,
                job.city,
                job.seed,
                budget,
                run_dir,
                device=device,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or inspect the staged-attribution queue")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget", choices=("smoke", "controlled", "confirmation"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--minimum-free-mib", type=int, default=16000)
    parser.add_argument("--maximum-utilization", type=int, default=20)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--maximum-parallel", type=int, default=4)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--cells", nargs="+")
    arguments = parser.parse_args()
    if arguments.parallel and arguments.dry_run:
        parser.error("--parallel and --dry-run are mutually exclusive")
    if arguments.parallel:
        result = run_parallel_experiment(
            arguments.config,
            arguments.budget,
            artifact_root=arguments.artifact_root,
            maximum_parallel=arguments.maximum_parallel,
            wait_for_gpu=arguments.wait_for_gpu,
            poll_seconds=arguments.poll_seconds,
            minimum_free_mib=arguments.minimum_free_mib,
            maximum_utilization=arguments.maximum_utilization,
            cell_ids=arguments.cells,
        )
    else:
        result = run_experiment(
            arguments.config,
            arguments.budget,
            arguments.dry_run,
            arguments.minimum_free_mib,
            arguments.maximum_utilization,
            arguments.cells,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if arguments.parallel and result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
