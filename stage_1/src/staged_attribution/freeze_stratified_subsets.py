from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .stratified_subset import stratified_subset, trajectory_id_sha256


SCHEMA_VERSION = 1
REGISTERED_BUDGET = "controlled"


class FrozenSubsetError(RuntimeError):
    """Raised when fixed corrected-reevaluation subsets cannot be frozen safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git_state() -> dict[str, object]:
    anchor = Path(__file__).resolve().parent
    try:
        root = subprocess.run(
            ["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        revision = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise FrozenSubsetError("cannot resolve generator Git provenance") from error
    if len(revision) != 40:
        raise FrozenSubsetError("generator Git revision is not a full commit hash")
    return {"worktree": str(Path(root).resolve()), "revision": revision, "dirty": bool(status)}


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _write_immutable_json(path: Path, payload: object) -> Path:
    content = _canonical_json(payload)
    if path.is_file():
        if path.read_bytes() != content:
            raise FrozenSubsetError(f"refusing to replace immutable subset artifact: {path}")
        return path
    if path.exists():
        raise FrozenSubsetError(f"subset artifact path is not a regular file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FrozenSubsetError(f"stale subset temporary file exists: {temporary}") from error
    try:
        os.link(temporary, path)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != content:
            raise FrozenSubsetError(f"concurrent immutable subset mismatch: {path}")
    finally:
        temporary.unlink()
    return path


def _aligned_population(
    aligned_root: Path,
) -> tuple[list[int], list[int], dict[str, object]]:
    traj_path = aligned_root / "test_traj_ids.npy"
    offsets_path = aligned_root / "test_offsets.npy"
    if not traj_path.is_file() or not offsets_path.is_file():
        raise FrozenSubsetError(f"aligned test identity is incomplete: {aligned_root}")
    traj_array = np.load(traj_path, allow_pickle=False)
    offsets = np.load(offsets_path, allow_pickle=False)
    if traj_array.ndim != 1 or offsets.ndim != 1 or len(offsets) != len(traj_array) + 1:
        raise FrozenSubsetError("aligned test IDs/offsets have invalid shapes")
    if not np.issubdtype(traj_array.dtype, np.integer) or not np.issubdtype(
        offsets.dtype, np.integer
    ):
        raise FrozenSubsetError("aligned test IDs/offsets must use integer dtypes")
    if int(offsets[0]) != 0:
        raise FrozenSubsetError("aligned test offsets must begin at zero")
    ids = [int(value) for value in traj_array.tolist()]
    lengths = [int(value) for value in np.diff(offsets).tolist()]
    if (
        len(set(ids)) != len(ids)
        or not ids
        or any(length <= 0 for length in lengths)
    ):
        raise FrozenSubsetError(
            "aligned test IDs must be nonempty/unique and lengths positive"
        )
    return ids, lengths, {
        "test_traj_ids": _artifact(traj_path),
        "test_offsets": _artifact(offsets_path),
        "population": len(ids),
        "traj_id_order_sha256": trajectory_id_sha256(ids),
        "traj_id_sha256": trajectory_id_sha256(sorted(ids)),
        "minimum_points": min(lengths),
        "maximum_points": max(lengths),
    }


def freeze_stratified_subsets(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    budget: str = REGISTERED_BUDGET,
    expected_revision: str | None = None,
    expected_config_sha256: str | None = None,
    require_clean: bool = False,
) -> dict[str, object]:
    if budget != REGISTERED_BUDGET:
        raise FrozenSubsetError(
            "fixed stratified reevaluation subsets are registered only for controlled"
        )
    resolved_config = Path(config_path).resolve()
    config = ExperimentConfig.load(resolved_config)
    config_artifact = _artifact(resolved_config)
    generator_git = _git_state()
    if (
        expected_config_sha256 is not None
        and config_artifact["sha256"] != expected_config_sha256
    ):
        raise FrozenSubsetError("config SHA-256 differs from the expected formal config")
    if (
        expected_revision is not None
        and generator_git["revision"] != expected_revision
    ):
        raise FrozenSubsetError("generator revision differs from the expected revision")
    if require_clean and generator_git["dirty"]:
        raise FrozenSubsetError("generator worktree must be clean")
    stage3_population = config.budgets[budget].stage3_trajectories
    if not isinstance(stage3_population, int) or isinstance(stage3_population, bool):
        raise FrozenSubsetError("controlled Stage3 population must be a fixed integer")
    root = Path(output_dir).resolve()
    records = []
    for city in config.cities:
        aligned_key = f"{city}_aligned_root"
        if aligned_key not in config.paths:
            raise FrozenSubsetError(f"config lacks {aligned_key}")
        ids, lengths, source = _aligned_population(config.paths[aligned_key])
        for seed in config.seeds:
            selected = stratified_subset(ids, lengths, stage3_population, seed)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": config.experiment_id,
                "budget": budget,
                "city": city,
                "selection_depends_on_cell": False,
                "config": config_artifact,
                "generator_git": generator_git,
                "source": source,
                **selected,
            }
            path = _write_immutable_json(
                root / f"{budget}_{city}_seed_{seed}.json", manifest
            )
            records.append(
                {
                    "city": city,
                    "seed": seed,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "selected_population": stage3_population,
                    "selected_traj_id_sha256": selected[
                        "selected_traj_id_sha256"
                    ],
                }
            )
    index = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "kind": "fixed_stratified_reevaluation_subsets",
        "budget": budget,
        "config": config_artifact,
        "generator_git": generator_git,
        "selection_depends_on_cell": False,
        "manifests": records,
    }
    index_path = _write_immutable_json(root / "index.json", index)
    return {
        **index,
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze immutable city/seed stratified reevaluation subsets"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", default=REGISTERED_BUDGET)
    parser.add_argument("--expected-revision")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--require-clean", action="store_true")
    arguments = parser.parse_args()
    result = freeze_stratified_subsets(
        arguments.config,
        arguments.output_dir,
        budget=arguments.budget,
        expected_revision=arguments.expected_revision,
        expected_config_sha256=arguments.expected_config_sha256,
        require_clean=arguments.require_clean,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
