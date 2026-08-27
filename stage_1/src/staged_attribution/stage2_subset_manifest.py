from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig


SCHEMA_VERSION = 1
SELECTION_METHOD = "sha256_rank_fixed_city_v1"


class Stage2SubsetManifestError(ValueError):
    """Raised when a frozen Stage2 training subset cannot be reproduced exactly."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_id_sha256(traj_ids: Sequence[int]) -> str:
    values = sorted(int(value) for value in traj_ids)
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def select_fixed_traj_ids(
    source_ids: Sequence[int] | np.ndarray,
    requested: int,
    subset_seed: int,
    city: str,
) -> list[int]:
    population = [int(value) for value in source_ids]
    if not city:
        raise Stage2SubsetManifestError("city must be non-empty")
    if len(set(population)) != len(population):
        raise Stage2SubsetManifestError("source traj_id values must be unique")
    if (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested <= 0
        or requested > len(population)
    ):
        raise Stage2SubsetManifestError(
            f"requested population {requested!r} is invalid for {len(population)} IDs"
        )
    if not isinstance(subset_seed, int) or isinstance(subset_seed, bool):
        raise Stage2SubsetManifestError("subset_seed must be an integer")
    ranked = sorted(
        population,
        key=lambda traj_id: (
            hashlib.sha256(
                f"{subset_seed}:stage2_train:{city}:{traj_id}".encode()
            ).digest(),
            traj_id,
        ),
    )
    return sorted(ranked[:requested])


def build_stage2_subset_manifest(
    source_ids: Sequence[int] | np.ndarray,
    city: str,
    requested: int,
    subset_seed: int,
    source_path: str | Path,
) -> dict[str, Any]:
    source = Path(source_path)
    selected_ids = select_fixed_traj_ids(source_ids, requested, subset_seed, city)
    population = [int(value) for value in source_ids]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "frozen_stage2_train_subset",
        "method": SELECTION_METHOD,
        "city": city,
        "requested": requested,
        "subset_seed": subset_seed,
        "source": {
            "path": str(source.resolve()),
            "file_sha256": file_sha256(source),
            "population": len(population),
            "traj_id_sha256": trajectory_id_sha256(population),
        },
        "selected": {
            "population": len(selected_ids),
            "traj_id_sha256": trajectory_id_sha256(selected_ids),
            "traj_ids": selected_ids,
        },
        "selection_depends_on_cell": False,
        "selection_depends_on_training_seed": False,
    }


def validate_stage2_subset_manifest(
    manifest: Mapping[str, Any],
    source_ids: Sequence[int] | np.ndarray,
    city: str,
    requested: int,
    subset_seed: int,
) -> list[int]:
    population = [int(value) for value in source_ids]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "kind": "frozen_stage2_train_subset",
        "method": SELECTION_METHOD,
        "city": city,
        "requested": requested,
        "subset_seed": subset_seed,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise Stage2SubsetManifestError(
                f"manifest {field} mismatch: {manifest.get(field)!r} != {value!r}"
            )
    source = manifest.get("source")
    selected = manifest.get("selected")
    if not isinstance(source, Mapping) or not isinstance(selected, Mapping):
        raise Stage2SubsetManifestError("manifest source and selected must be objects")
    if source.get("population") != len(population):
        raise Stage2SubsetManifestError("manifest source population mismatch")
    if source.get("traj_id_sha256") != trajectory_id_sha256(population):
        raise Stage2SubsetManifestError("manifest source traj_id hash mismatch")
    source_path_raw = source.get("path")
    if not isinstance(source_path_raw, str) or not source_path_raw:
        raise Stage2SubsetManifestError("manifest source path must be non-empty")
    source_path = Path(source_path_raw)
    if not source_path.is_file():
        raise Stage2SubsetManifestError(f"manifest source path does not exist: {source_path}")
    if source.get("file_sha256") != file_sha256(source_path):
        raise Stage2SubsetManifestError("manifest source file hash mismatch")
    frozen_source_ids = np.load(source_path, mmap_mode="r")
    if len(frozen_source_ids) != len(population) or trajectory_id_sha256(
        frozen_source_ids
    ) != trajectory_id_sha256(population):
        raise Stage2SubsetManifestError(
            "manifest source file IDs do not match the current Stage2 population"
        )
    selected_ids_raw = selected.get("traj_ids")
    if not isinstance(selected_ids_raw, list) or any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in selected_ids_raw
    ):
        raise Stage2SubsetManifestError("manifest selected traj_ids must be an integer list")
    selected_ids = [int(value) for value in selected_ids_raw]
    if selected_ids != sorted(selected_ids) or len(set(selected_ids)) != len(selected_ids):
        raise Stage2SubsetManifestError("manifest selected traj_ids must be sorted and unique")
    if selected.get("population") != requested or len(selected_ids) != requested:
        raise Stage2SubsetManifestError("manifest selected population mismatch")
    if selected.get("traj_id_sha256") != trajectory_id_sha256(selected_ids):
        raise Stage2SubsetManifestError("manifest selected traj_id hash mismatch")
    if not set(selected_ids).issubset(population):
        raise Stage2SubsetManifestError("manifest contains traj_id outside source population")
    expected_ids = select_fixed_traj_ids(population, requested, subset_seed, city)
    if selected_ids != expected_ids:
        raise Stage2SubsetManifestError("manifest selected traj_ids do not reproduce method")
    if manifest.get("selection_depends_on_cell") is not False:
        raise Stage2SubsetManifestError("manifest must be cell independent")
    if manifest.get("selection_depends_on_training_seed") is not False:
        raise Stage2SubsetManifestError("manifest must be training-seed independent")
    return selected_ids


def load_stage2_subset_manifest(
    path: str | Path,
    source_ids: Sequence[int] | np.ndarray,
    city: str,
    requested: int,
    subset_seed: int,
) -> tuple[list[int], dict[str, Any]]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise Stage2SubsetManifestError("manifest root must be an object")
    selected_ids = validate_stage2_subset_manifest(
        manifest,
        source_ids,
        city,
        requested,
        subset_seed,
    )
    return selected_ids, dict(manifest)


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = Path(path)
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to replace different frozen manifest: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    return output


def freeze_configured_subsets(
    config: ExperimentConfig,
    budget_name: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    if budget_name not in config.budgets:
        raise Stage2SubsetManifestError(f"unknown budget {budget_name!r}")
    budget = config.budgets[budget_name]
    if not isinstance(budget.stage2_train_trajectories, int):
        raise Stage2SubsetManifestError(
            "freezing requires a finite stage2_train_trajectories budget"
        )
    output = Path(output_dir)
    records: dict[str, Any] = {}
    for city in config.cities:
        source_path = config.paths[f"{city}_aligned_root"] / "train_traj_ids.npy"
        source_ids = np.load(source_path, mmap_mode="r")
        manifest = build_stage2_subset_manifest(
            source_ids,
            city,
            budget.stage2_train_trajectories,
            budget.stage2_subset_seed,
            source_path,
        )
        filename = (
            f"{city}_stage2_train_{budget.stage2_train_trajectories}_"
            f"seed{budget.stage2_subset_seed}.json"
        )
        path = write_immutable_json(output / filename, manifest)
        records[city] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "source_population": manifest["source"]["population"],
            "selected_population": manifest["selected"]["population"],
            "selected_traj_id_sha256": manifest["selected"]["traj_id_sha256"],
        }
    index = {
        "schema_version": SCHEMA_VERSION,
        "kind": "frozen_stage2_train_subset_index",
        "experiment_id": config.experiment_id,
        "budget": budget_name,
        "stage2_train_trajectories": budget.stage2_train_trajectories,
        "stage2_subset_seed": budget.stage2_subset_seed,
        "cities": records,
    }
    index_path = write_immutable_json(output / "index.json", index)
    return {"index": str(index_path.resolve()), "index_sha256": file_sha256(index_path), **index}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and verify a city-level Stage2 training trajectory subset."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget", default="controlled")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = freeze_configured_subsets(
        ExperimentConfig.load(args.config),
        args.budget,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
