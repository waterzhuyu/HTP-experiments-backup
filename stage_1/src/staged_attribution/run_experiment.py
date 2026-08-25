from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import ExperimentConfig


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
