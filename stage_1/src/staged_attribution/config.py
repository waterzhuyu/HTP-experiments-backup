from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ALLOWED_VIEWS = frozenset({"gps", "road", "grid"})
EXPECTED_RQ_VOCAB = (32, 64, 128, 256)
REQUIRED_E_IDS = tuple(f"e{index}" for index in range(7))
REQUIRED_Q_IDS = tuple(f"q{index}" for index in range(4))
LOSS_NAMES = ("usage", "invariance", "variance", "covariance", "auxiliary")
REQUIRED_BUDGETS = ("smoke", "controlled", "confirmation")


class ConfigError(ValueError):
    """Raised when an experiment config violates a frozen design invariant."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class Budget:
    name: str
    stage1_epochs: int
    stage1_train_trajectories: int | str
    stage1_test_trajectories: int | str
    stage1_batch_size: int
    stage2_epochs: int
    stage2_max_steps: int
    stage2_train_trajectories: int | str
    stage2_subset_seed: int
    stage2_batch_size: int
    stage2_gradient_accumulation: int
    stage2_subset_manifests: Mapping[str, str]
    stage3_trajectories: int | str

    @property
    def stage2_effective_batch_size(self) -> int:
        return self.stage2_batch_size * self.stage2_gradient_accumulation

    @classmethod
    def from_dict(cls, name: str, raw: object) -> Budget:
        values = _mapping(raw, f"budget {name}")
        required = (
            "stage1_epochs", "stage1_train_trajectories", "stage1_test_trajectories",
            "stage1_batch_size", "stage2_epochs", "stage2_max_steps", "stage3_trajectories",
        )
        for field in required:
            if field not in values:
                raise ConfigError(f"budget {name} missing {field}")
        stage1_epochs = values["stage1_epochs"]
        stage1_train_trajectories = values["stage1_train_trajectories"]
        stage1_test_trajectories = values["stage1_test_trajectories"]
        stage1_batch_size = values["stage1_batch_size"]
        stage2_epochs = values["stage2_epochs"]
        stage2_max_steps = values["stage2_max_steps"]
        stage2_train_trajectories = values.get("stage2_train_trajectories", "full")
        stage2_subset_seed = values.get("stage2_subset_seed", 20260826)
        stage2_batch_size = values.get("stage2_batch_size", 16)
        stage2_gradient_accumulation = values.get("stage2_gradient_accumulation", 1)
        stage2_subset_manifests_raw = values.get("stage2_subset_manifests", {})
        stage3_trajectories = values["stage3_trajectories"]
        if not isinstance(stage1_epochs, int) or stage1_epochs <= 0:
            raise ConfigError(f"budget {name} stage1_epochs must be positive")
        for field, population in (
            ("stage1_train_trajectories", stage1_train_trajectories),
            ("stage1_test_trajectories", stage1_test_trajectories),
        ):
            if population != "full" and (not isinstance(population, int) or population <= 0):
                raise ConfigError(f"budget {name} {field} must be positive or 'full'")
        if not isinstance(stage1_batch_size, int) or stage1_batch_size <= 0:
            raise ConfigError(f"budget {name} stage1_batch_size must be positive")
        if not isinstance(stage2_epochs, int) or stage2_epochs <= 0:
            raise ConfigError(f"budget {name} stage2_epochs must be positive")
        if not isinstance(stage2_max_steps, int) or stage2_max_steps == 0 or stage2_max_steps < -1:
            raise ConfigError(f"budget {name} stage2_max_steps must be -1 or positive")
        if stage2_train_trajectories != "full" and (
            not isinstance(stage2_train_trajectories, int)
            or stage2_train_trajectories <= 0
        ):
            raise ConfigError(
                f"budget {name} stage2_train_trajectories must be positive or 'full'"
            )
        if not isinstance(stage2_subset_seed, int) or isinstance(stage2_subset_seed, bool):
            raise ConfigError(f"budget {name} stage2_subset_seed must be an integer")
        if (
            not isinstance(stage2_batch_size, int)
            or isinstance(stage2_batch_size, bool)
            or stage2_batch_size <= 0
        ):
            raise ConfigError(f"budget {name} stage2_batch_size must be positive")
        if (
            not isinstance(stage2_gradient_accumulation, int)
            or isinstance(stage2_gradient_accumulation, bool)
            or stage2_gradient_accumulation <= 0
        ):
            raise ConfigError(
                f"budget {name} stage2_gradient_accumulation must be positive"
            )
        stage2_subset_manifests = _mapping(
            stage2_subset_manifests_raw,
            f"budget {name} stage2_subset_manifests",
        )
        if any(
            not isinstance(city, str)
            or not city
            or not isinstance(path, str)
            or not path
            for city, path in stage2_subset_manifests.items()
        ):
            raise ConfigError(
                f"budget {name} stage2_subset_manifests must map city names to paths"
            )
        if stage2_train_trajectories == "full" and stage2_subset_manifests:
            raise ConfigError(
                f"budget {name} stage2_subset_manifests require a finite "
                "stage2_train_trajectories budget"
            )
        if stage3_trajectories != "full" and (
            not isinstance(stage3_trajectories, int) or stage3_trajectories <= 0
        ):
            raise ConfigError(f"budget {name} stage3_trajectories must be positive or 'full'")
        return cls(
            name=name,
            stage1_epochs=stage1_epochs,
            stage1_train_trajectories=stage1_train_trajectories,
            stage1_test_trajectories=stage1_test_trajectories,
            stage1_batch_size=stage1_batch_size,
            stage2_epochs=stage2_epochs,
            stage2_max_steps=stage2_max_steps,
            stage2_train_trajectories=stage2_train_trajectories,
            stage2_subset_seed=stage2_subset_seed,
            stage2_batch_size=stage2_batch_size,
            stage2_gradient_accumulation=stage2_gradient_accumulation,
            stage2_subset_manifests=dict(stage2_subset_manifests),
            stage3_trajectories=stage3_trajectories,
        )


@dataclass(frozen=True)
class CellSpec:
    e_id: str
    q_id: str
    views: frozenset[str]
    q_weights: Mapping[str, float]

    @property
    def cell_id(self) -> str:
        return f"{self.e_id}_{self.q_id}"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    cities: tuple[str, ...]
    seeds: tuple[int, ...]
    rq_vocab_sizes: tuple[int, ...]
    e_variants: Mapping[str, frozenset[str]]
    q_variants: Mapping[str, Mapping[str, float]]
    budgets: Mapping[str, Budget]
    paths: Mapping[str, Path]
    provenance: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExperimentConfig:
        experiment_id = raw.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ConfigError("experiment_id must be a non-empty string")

        cities_raw = raw.get("cities")
        if not isinstance(cities_raw, list) or not cities_raw or any(
            not isinstance(city, str) or not city for city in cities_raw
        ):
            raise ConfigError("cities must be a non-empty string list")
        cities = tuple(cities_raw)
        if len(set(cities)) != len(cities):
            raise ConfigError("cities must be unique")

        seeds_raw = raw.get("seeds")
        if not isinstance(seeds_raw, list) or not seeds_raw or any(
            not isinstance(seed, int) for seed in seeds_raw
        ):
            raise ConfigError("seeds must be a non-empty integer list")
        seeds = tuple(seeds_raw)
        if len(set(seeds)) != len(seeds):
            raise ConfigError("seeds must be unique")

        vocab_raw = raw.get("rq_vocab_sizes")
        if not isinstance(vocab_raw, list) or tuple(vocab_raw) != EXPECTED_RQ_VOCAB:
            raise ConfigError(f"RQ vocabulary must remain {list(EXPECTED_RQ_VOCAB)}")

        e_raw = _mapping(raw.get("e_variants"), "e_variants")
        for e_id, views_raw in e_raw.items():
            if not isinstance(views_raw, list) or not views_raw:
                continue
            unsupported = set(views_raw) - ALLOWED_VIEWS
            if unsupported:
                raise ConfigError(
                    f"unsupported view(s) in {e_id}: {sorted(unsupported)}"
                )
        if tuple(e_raw) != REQUIRED_E_IDS:
            raise ConfigError(f"e_variants must be ordered as {list(REQUIRED_E_IDS)}")
        e_variants: dict[str, frozenset[str]] = {}
        observed_masks: set[frozenset[str]] = set()
        for e_id, views_raw in e_raw.items():
            if not isinstance(views_raw, list) or not views_raw:
                raise ConfigError(f"{e_id} must contain at least one view")
            views = frozenset(views_raw)
            unsupported = views - ALLOWED_VIEWS
            if unsupported:
                raise ConfigError(f"unsupported view(s) in {e_id}: {sorted(unsupported)}")
            if len(views) != len(views_raw):
                raise ConfigError(f"duplicate view name in {e_id}")
            if views in observed_masks:
                raise ConfigError(f"duplicate view mask in {e_id}: {sorted(views)}")
            observed_masks.add(views)
            e_variants[e_id] = views
        expected_masks = {frozenset(mask) for mask in (
            ("gps",), ("road",), ("grid",), ("gps", "road"),
            ("gps", "grid"), ("road", "grid"), ("gps", "road", "grid"),
        )}
        if observed_masks != expected_masks:
            raise ConfigError("e_variants must contain every non-empty GPS/Road/Grid mask exactly once")

        q_raw = _mapping(raw.get("q_variants"), "q_variants")
        if tuple(q_raw) != REQUIRED_Q_IDS:
            raise ConfigError(f"q_variants must be ordered as {list(REQUIRED_Q_IDS)}")
        q_variants: dict[str, dict[str, float]] = {}
        for q_id, weights_raw in q_raw.items():
            weights = _mapping(weights_raw, q_id)
            if tuple(weights) != LOSS_NAMES:
                raise ConfigError(f"{q_id} weights must be ordered as {list(LOSS_NAMES)}")
            parsed: dict[str, float] = {}
            for name, value in weights.items():
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ConfigError(f"{q_id}.{name} must be a finite non-negative weight")
                parsed[name] = float(value)
            q_variants[q_id] = parsed

        budgets_raw = _mapping(raw.get("budgets"), "budgets")
        if tuple(budgets_raw) != REQUIRED_BUDGETS:
            raise ConfigError(f"budgets must be ordered as {list(REQUIRED_BUDGETS)}")
        budgets = {name: Budget.from_dict(name, value) for name, value in budgets_raw.items()}
        for name, budget in budgets.items():
            if budget.stage2_subset_manifests and set(
                budget.stage2_subset_manifests
            ) != set(cities):
                raise ConfigError(
                    f"budget {name} stage2_subset_manifests must cover exactly "
                    f"{sorted(cities)}"
                )

        controlled = budgets["controlled"]
        confirmation = budgets["confirmation"]
        stage2_invariants = (
            "stage2_epochs",
            "stage2_max_steps",
            "stage2_train_trajectories",
            "stage2_subset_seed",
            "stage2_batch_size",
            "stage2_gradient_accumulation",
            "stage2_subset_manifests",
        )
        drift = {
            field: (getattr(confirmation, field), getattr(controlled, field))
            for field in stage2_invariants
            if getattr(confirmation, field) != getattr(controlled, field)
        }
        if drift:
            raise ConfigError(
                f"confirmation Stage2 method/resource controls must match controlled: {drift}"
            )

        paths_raw = _mapping(raw.get("paths"), "paths")
        for required in ("model_root", "data_root", "artifact_root"):
            if required not in paths_raw or not isinstance(paths_raw[required], str):
                raise ConfigError(f"paths missing string {required}")
        paths = {name: Path(value) for name, value in paths_raw.items() if isinstance(value, str)}
        provenance = dict(_mapping(raw.get("provenance", {}), "provenance"))

        return cls(
            experiment_id=experiment_id,
            cities=cities,
            seeds=seeds,
            rq_vocab_sizes=EXPECTED_RQ_VOCAB,
            e_variants=e_variants,
            q_variants=q_variants,
            budgets=budgets,
            paths=paths,
            provenance=provenance,
        )

    def core_cells(self) -> tuple[CellSpec, ...]:
        cells = [
            CellSpec(e_id, "q0", views, self.q_variants["q0"])
            for e_id, views in self.e_variants.items()
        ]
        cells.extend(
            CellSpec("e6", q_id, self.e_variants["e6"], self.q_variants[q_id])
            for q_id in ("q1", "q2", "q3")
        )
        return tuple(cells)

    def cell(self, cell_id: str) -> CellSpec:
        parts = cell_id.split("_")
        if len(parts) != 2 or parts[0] not in self.e_variants or parts[1] not in self.q_variants:
            raise ConfigError(f"unknown E×Q cell {cell_id!r}")
        e_id, q_id = parts
        return CellSpec(e_id, q_id, self.e_variants[e_id], self.q_variants[q_id])

    def all_cells(self) -> tuple[CellSpec, ...]:
        return tuple(
            self.cell(f"{e_id}_{q_id}")
            for e_id in self.e_variants
            for q_id in self.q_variants
        )
