from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from .config import ExperimentConfig
from .run_experiment import (
    build_jobs,
    completed_attempt,
    query_gpus,
    run_parallel_experiment,
    safe_gpu_indices,
)


RESOURCE_OPTIONS = ("A", "B", "C")


@dataclass(frozen=True)
class LaunchContract:
    worktree: Path
    config_path: Path
    artifact_root: Path
    expected_branch: str
    expected_head: str
    expected_config_sha256: str
    resource_option: str
    budget: str = "controlled"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(worktree: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(worktree), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_resource_option(config: ExperimentConfig, option: str, budget: str) -> None:
    if option not in RESOURCE_OPTIONS:
        raise RuntimeError(f"unsupported Stage2 resource option: {option}")
    marker = config.provenance.get("stage2_resource_option")
    if marker != option:
        raise RuntimeError(
            "formal config provenance.stage2_resource_option must equal "
            f"{option!r}; found {marker!r}"
        )
    if config.provenance.get("candidate_only") is True:
        raise RuntimeError("candidate_only config cannot launch formal experiments")

    selected = config.budgets[budget]
    actual = (
        selected.stage2_train_trajectories,
        selected.stage2_batch_size,
        selected.stage2_gradient_accumulation,
    )
    expected = {
        "A": ("full", 8, 2),
        "B": (64000, 8, 2),
        "C": ("full", 16, 1),
    }[option]
    if actual != expected:
        raise RuntimeError(
            f"Stage2 resource option {option} requires population/batch/accumulation "
            f"{expected}, found {actual}"
        )
    manifests = selected.stage2_subset_manifests
    if option == "B":
        if set(manifests) != set(config.cities):
            raise RuntimeError(
                "option B requires explicit immutable Stage2 subset manifests for every city"
            )
        missing = [path for path in manifests.values() if not Path(path).is_file()]
        if missing:
            raise RuntimeError(f"option B subset manifest files are missing: {sorted(missing)}")
    elif manifests:
        raise RuntimeError(f"option {option} must not use Stage2 subset manifests")


def validate_launch_contract(contract: LaunchContract) -> ExperimentConfig:
    worktree = contract.worktree.resolve()
    config_path = contract.config_path.resolve()
    if not config_path.is_relative_to(worktree):
        raise RuntimeError("formal config must be inside the locked execution worktree")
    if _git(worktree, "branch", "--show-current") != contract.expected_branch:
        raise RuntimeError("execution branch drifted from the launch contract")
    if _git(worktree, "rev-parse", "HEAD") != contract.expected_head:
        raise RuntimeError("execution revision drifted from the launch contract")
    if _git(worktree, "status", "--porcelain"):
        raise RuntimeError("execution worktree is dirty; refusing to launch")
    observed_sha256 = sha256_file(config_path)
    if observed_sha256 != contract.expected_config_sha256:
        raise RuntimeError(
            "formal config SHA256 drifted: "
            f"expected {contract.expected_config_sha256}, found {observed_sha256}"
        )
    config = ExperimentConfig.load(config_path)
    if contract.budget not in config.budgets:
        raise RuntimeError(f"unknown launch budget: {contract.budget}")
    _validate_resource_option(config, contract.resource_option, contract.budget)
    artifact_root = contract.artifact_root.resolve()
    if contract.artifact_root.name != config.experiment_id:
        raise RuntimeError(
            "artifact root basename must equal the formal experiment_id: "
            f"expected {config.experiment_id}, found {contract.artifact_root.name}"
        )
    if artifact_root.is_relative_to(worktree):
        raise RuntimeError("formal artifacts must remain outside the Git execution worktree")
    return config


def active_formal_processes(contract: LaunchContract) -> list[dict[str, object]]:
    completed = subprocess.run(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    config_token = str(contract.config_path.resolve())
    artifact_token = str(contract.artifact_root.resolve())
    active = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        pid, command = int(fields[0]), fields[1]
        is_cell = "staged_attribution.run_cell" in command and artifact_token in command
        is_runner = "staged_attribution.run_experiment" in command and config_token in command
        if is_cell or is_runner:
            active.append({"pid": pid, "command": command})
    return active


@contextmanager
def single_instance_lock(path: str | Path) -> Iterator[None]:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another GPU watcher holds {lock_path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _event(event: str, **fields: object) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def inspect_queue(
    config: ExperimentConfig,
    contract: LaunchContract,
    cell_ids: Sequence[str] | None,
) -> dict[str, object]:
    jobs = build_jobs(config, contract.budget, cell_ids)
    completed = []
    pending = []
    for job in jobs:
        attempt = completed_attempt(contract.artifact_root / job.run_key)
        if attempt is None:
            pending.append(job.run_key)
        else:
            completed.append({"run_key": job.run_key, "attempt": str(attempt)})
    return {"completed": completed, "pending": pending}


def watch_and_launch(
    contract: LaunchContract,
    maximum_parallel: int = 4,
    poll_seconds: float = 15.0,
    minimum_free_mib: int = 16000,
    maximum_utilization: int = 20,
    cell_ids: Sequence[str] | None = None,
    validate_only: bool = False,
) -> dict[str, object]:
    config = validate_launch_contract(contract)
    queue = inspect_queue(config, contract, cell_ids)
    if validate_only:
        rows = query_gpus()
        safe = safe_gpu_indices(rows, minimum_free_mib, maximum_utilization)
        return {
            "contract_valid": True,
            "resource_option": contract.resource_option,
            "config_sha256": contract.expected_config_sha256,
            "safe_gpus": safe,
            "gpu_rows": rows,
            "completed_count": len(queue["completed"]),
            "pending_count": len(queue["pending"]),
        }

    lock_path = contract.artifact_root / "provenance" / "gpu_watch_and_launch.lock"
    with single_instance_lock(lock_path):
        validate_launch_contract(contract)
        active = active_formal_processes(contract)
        if active:
            raise RuntimeError(
                "formal experiment processes are already active; refusing a duplicate launch: "
                f"{active}"
            )
        _event(
            "watcher_started",
            resource_option=contract.resource_option,
            pending_count=len(queue["pending"]),
            minimum_free_mib=minimum_free_mib,
            maximum_utilization=maximum_utilization,
            poll_seconds=poll_seconds,
        )

        def guarded_query() -> list[dict[str, int]]:
            validate_launch_contract(contract)
            observed = query_gpus()
            eligible = safe_gpu_indices(
                observed,
                minimum_free_mib=minimum_free_mib,
                maximum_utilization=maximum_utilization,
            )
            _event("gpu_poll", safe_gpus=eligible, gpu_rows=observed)
            return observed

        result = run_parallel_experiment(
            contract.config_path,
            contract.budget,
            artifact_root=contract.artifact_root,
            maximum_parallel=maximum_parallel,
            wait_for_gpu=True,
            poll_seconds=poll_seconds,
            minimum_free_mib=minimum_free_mib,
            maximum_utilization=maximum_utilization,
            query_gpu_fn=guarded_query,
            cell_ids=cell_ids,
        )
        _event(
            "watcher_finished",
            completed_count=len(result["completed"]),
            failed_count=len(result["failed"]),
            pending_count=len(result["pending"]),
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously watch GPU gates and launch the immutable formal experiment"
    )
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--resource-option", choices=RESOURCE_OPTIONS, required=True)
    parser.add_argument("--budget", choices=("controlled",), default="controlled")
    parser.add_argument("--maximum-parallel", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--minimum-free-mib", type=int, default=16000)
