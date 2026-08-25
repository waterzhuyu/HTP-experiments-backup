from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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


def build_jobs(config: ExperimentConfig, budget: str) -> tuple[ExperimentJob, ...]:
    if budget not in config.budgets:
        raise ValueError(f"unknown budget {budget}")
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
        for cell in config.core_cells()
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


def run_experiment(
    config_path: str | Path,
    budget: str,
    dry_run: bool,
    minimum_free_mib: int = 16000,
    maximum_utilization: int = 20,
) -> list[dict[str, object]]:
    config = ExperimentConfig.load(config_path)
    jobs = build_jobs(config, budget)
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
    arguments = parser.parse_args()
    result = run_experiment(
        arguments.config,
        arguments.budget,
        arguments.dry_run,
        arguments.minimum_free_mib,
        arguments.maximum_utilization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
