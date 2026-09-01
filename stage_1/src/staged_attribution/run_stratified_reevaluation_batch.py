from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> tuple[str, bool]:
    anchor = Path(__file__).resolve().parent
    revision = subprocess.run(
        ["git", "-C", str(anchor), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(anchor), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def _validate_digest(value: str, length: int, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ReevaluationBatchError(f"{label} must be a full lowercase hex digest")
    return normalized


def _run_key_sha256(run_keys: Sequence[str]) -> str:
    content = "".join(f"{run_key}\n" for run_key in sorted(run_keys)).encode()
    return hashlib.sha256(content).hexdigest()


def _read_source_audit(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReevaluationBatchError(f"invalid completed-source audit: {path}") from error
    if not isinstance(payload, Mapping):
        raise ReevaluationBatchError("completed-source audit must be a JSON object")
    return payload


@contextmanager
def _exclusive_batch_lock(root: Path) -> Iterator[tuple[Path, int]]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".stratified_reevaluation_batch.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.lseek(descriptor, 0, os.SEEK_SET)
            holder = os.read(descriptor, 4096).decode(errors="replace").strip()
            detail = f": {holder}" if holder else ""
            raise ReevaluationBatchError(
                f"another corrected reevaluation batch holds {path}{detail}"
            ) from error
        payload = {
            "schema_version": 1,
            "kind": "stratified_reevaluation_batch_lock",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, content)
        os.fsync(descriptor)
        yield path, descriptor
    finally:
        # Do not explicitly unlock: children inherit this open-file description.
        # Closing the parent's reference preserves the lock until the last child exits.
        os.close(descriptor)


def completed_reevaluation_attempt(
    job_root: str | Path,
    expected_identity: Mapping[str, object] | None = None,
) -> Path | None:
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
        if manifest.get("status") != "completed":
            continue
        if expected_identity is not None and any(
            manifest.get(key) != value for key, value in expected_identity.items()
        ):
            continue
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
    expected_revision: str,
    expected_config_sha256: str,
    expected_source_revision: str,
    source_audit_path: Path,
    expected_source_audit_sha256: str,
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
        "--expected-revision", expected_revision,
        "--expected-config-sha256", expected_config_sha256,
        "--expected-source-revision", expected_source_revision,
        "--source-audit", str(source_audit_path),
        "--expected-source-audit-sha256", expected_source_audit_sha256,
        "--device", "cuda:0",
    ]


def _validate_source_audit(
    audit: Mapping[str, object],
    *,
    resolved_config: Path,
    config_sha256: str,
    config: ExperimentConfig,
    source_root: Path,
    subset_dir: Path,
    jobs: Sequence[ExperimentJob],
    source_attempts: Mapping[str, Path],
    expected_revision: str,
) -> dict[str, str]:
    if audit.get("schema_version") != 1:
        raise ReevaluationBatchError("completed-source audit schema mismatch")
    if audit.get("kind") != "completed_formal_source_evidence_audit":
        raise ReevaluationBatchError("completed-source audit kind mismatch")
    if audit.get("experiment_id") != config.experiment_id:
        raise ReevaluationBatchError("completed-source audit experiment mismatch")
    if audit.get("budget") != "controlled":
        raise ReevaluationBatchError("completed-source audit budget mismatch")
    if Path(str(audit.get("source_root", ""))).resolve() != source_root:
        raise ReevaluationBatchError("completed-source audit source root mismatch")

    audit_config = audit.get("config")
    if not isinstance(audit_config, Mapping):
        raise ReevaluationBatchError("completed-source audit lacks config evidence")
    if Path(str(audit_config.get("path", ""))).resolve() != resolved_config:
        raise ReevaluationBatchError("completed-source audit config path mismatch")
    if audit_config.get("sha256") != config_sha256:
        raise ReevaluationBatchError("completed-source audit config SHA-256 mismatch")

    generator = audit.get("generator_git")
    if not isinstance(generator, Mapping):
        raise ReevaluationBatchError("completed-source audit lacks generator provenance")
    if generator.get("revision") != expected_revision or generator.get("dirty") is not False:
        raise ReevaluationBatchError("completed-source audit generator identity mismatch")

    subset_index_path = (subset_dir / "index.json").resolve()
    if not subset_index_path.is_file():
        raise ReevaluationBatchError("fixed-subset index is missing")
    subset_index = audit.get("fixed_subset_index")
    if not isinstance(subset_index, Mapping):
        raise ReevaluationBatchError("completed-source audit lacks fixed-subset index")
    if Path(str(subset_index.get("path", ""))).resolve() != subset_index_path:
        raise ReevaluationBatchError("completed-source audit fixed-subset index path mismatch")
    if subset_index.get("sha256") != _sha256(subset_index_path):
        raise ReevaluationBatchError("completed-source audit fixed-subset index SHA-256 mismatch")

    run_keys = [job.run_key for job in jobs]
    requested_cells = sorted({job.cell_id for job in jobs})
    if audit.get("requested_cell_ids") != requested_cells:
        raise ReevaluationBatchError("completed-source audit cell population mismatch")
    if audit.get("requested_job_count") != len(jobs):
        raise ReevaluationBatchError("completed-source audit job count mismatch")
    if audit.get("requested_run_key_sha256") != _run_key_sha256(run_keys):
        raise ReevaluationBatchError("completed-source audit run-key population mismatch")
    if audit.get("all_requested_sources_completed") is not True:
        raise ReevaluationBatchError("completed-source audit is not complete")
    if audit.get("missing_completed_source_count") != 0:
        raise ReevaluationBatchError("completed-source audit records missing sources")
    if audit.get("missing_completed_sources") != []:
        raise ReevaluationBatchError("completed-source audit missing-source list is non-empty")
    if audit.get("audited_completed_source_count") != len(jobs):
        raise ReevaluationBatchError("completed-source audit completed count mismatch")

    raw_rows = audit.get("audited_completed_sources")
    if not isinstance(raw_rows, list) or any(
        not isinstance(row, Mapping) for row in raw_rows
    ):
        raise ReevaluationBatchError("completed-source audit rows are invalid")
    rows_by_key: dict[str, Mapping[str, object]] = {}
    for raw_row in raw_rows:
        row = raw_row
        run_key = str(row.get("run_key", ""))
        if run_key in rows_by_key:
            raise ReevaluationBatchError("completed-source audit has duplicate run keys")
        rows_by_key[run_key] = row
    if set(rows_by_key) != set(run_keys):
        raise ReevaluationBatchError("completed-source audit row population mismatch")

    source_revisions: dict[str, str] = {}
    for job in jobs:
        row = rows_by_key[job.run_key]
        if (
            row.get("cell_id") != job.cell_id
            or row.get("city") != job.city
            or row.get("seed") != job.seed
            or row.get("budget") != job.budget
        ):
            raise ReevaluationBatchError(
                f"completed-source audit job identity mismatch: {job.run_key}"
            )
        expected_attempt = source_attempts[job.run_key].resolve()
        if Path(str(row.get("source_attempt", ""))).resolve() != expected_attempt:
            raise ReevaluationBatchError(
                f"completed-source audit attempt mismatch: {job.run_key}"
            )
        source_revision = _validate_digest(
            str(row.get("source_revision", "")),
            40,
            f"source revision for {job.run_key}",
        )
        fixed_subset = row.get("fixed_subset")
        if not isinstance(fixed_subset, Mapping):
            raise ReevaluationBatchError(
                f"completed-source audit lacks subset evidence: {job.run_key}"
            )
        subset_path = _subset_path(subset_dir, job).resolve()
        if Path(str(fixed_subset.get("path", ""))).resolve() != subset_path:
            raise ReevaluationBatchError(
                f"completed-source audit subset path mismatch: {job.run_key}"
            )
        if fixed_subset.get("sha256") != _sha256(subset_path):
            raise ReevaluationBatchError(
                f"completed-source audit subset SHA-256 mismatch: {job.run_key}"
            )
        source_revisions[job.run_key] = source_revision
    return source_revisions


def run_parallel_stratified_reevaluations(
    config_path: str | Path,
    *,
    source_root: str | Path,
    reevaluation_root: str | Path,
    subset_dir: str | Path,
    source_audit_path: str | Path,
    expected_revision: str,
    expected_config_sha256: str,
    expected_source_audit_sha256: str,
    maximum_parallel: int = 4,
    wait_for_gpu: bool = True,
    poll_seconds: float = 60.0,
    minimum_free_mib: int = 16000,
    maximum_utilization: int = 20,
    cell_ids: Sequence[str] | None = None,
    query_gpu_fn: Callable[[], list[dict[str, int]]] = query_gpus,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    git_identity_fn: Callable[[], tuple[str, bool]] = _git_identity,
) -> dict[str, object]:
    if maximum_parallel <= 0 or poll_seconds < 0:
        raise ValueError("invalid parallel or polling settings")
    expected_revision = _validate_digest(
        expected_revision, 40, "expected reevaluation revision"
    )
    expected_config_sha256 = _validate_digest(
        expected_config_sha256, 64, "expected config SHA-256"
    )
    expected_source_audit_sha256 = _validate_digest(
        expected_source_audit_sha256, 64, "expected source-audit SHA-256"
    )
    revision, dirty = git_identity_fn()
    if revision != expected_revision:
        raise ReevaluationBatchError("reevaluation implementation revision mismatch")
    if dirty:
        raise ReevaluationBatchError("reevaluation implementation worktree must be clean")
    resolved_config = Path(config_path).resolve()
    actual_config_sha256 = _sha256(resolved_config)
    if actual_config_sha256 != expected_config_sha256:
        raise ReevaluationBatchError("reevaluation config SHA-256 mismatch")
    audit_path = Path(source_audit_path).resolve()
    if (
        not audit_path.is_file()
        or audit_path.is_symlink()
        or _sha256(audit_path) != expected_source_audit_sha256
    ):
        raise ReevaluationBatchError("completed-source audit SHA-256 mismatch")
    source_audit = _read_source_audit(audit_path)
    config = ExperimentConfig.load(resolved_config)
    jobs = list(build_jobs(config, "controlled", cell_ids))
    source = Path(source_root).resolve()
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
    source_revisions = _validate_source_audit(
        source_audit,
        resolved_config=resolved_config,
        config_sha256=actual_config_sha256,
        config=config,
        source_root=source,
        subset_dir=subsets.resolve(),
        jobs=jobs,
        source_attempts=source_attempts,
        expected_revision=expected_revision,
    )

    def _attempt_identity(job: ExperimentJob) -> dict[str, object]:
        return {
            "experiment_id": config.experiment_id,
            "reevaluation": "fixed_stratified_population_aware_v1",
            "cell_id": job.cell_id,
            "city": job.city,
            "seed": job.seed,
            "budget": job.budget,
            "config_sha256": actual_config_sha256,
            "expected_source_revision": source_revisions[job.run_key],
            "source_attempt": str(source_attempts[job.run_key].resolve()),
            "subset_manifest": str(_subset_path(subsets, job).resolve()),
            "source_audit_path": str(audit_path),
            "source_audit_sha256": expected_source_audit_sha256,
            "git": {"revision": expected_revision, "dirty": False},
        }

    def _run_locked(lock_path: Path, lock_fd: int) -> dict[str, object]:
        completed: list[dict[str, object]] = []
        pending: list[ExperimentJob] = []
        for job in jobs:
            existing = completed_reevaluation_attempt(
                target / job.run_key, _attempt_identity(job)
            )
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
                    expected_revision, actual_config_sha256,
                    source_revisions[job.run_key],
                    audit_path, expected_source_audit_sha256,
                )
                process = popen_factory(
                    command,
                    cwd=Path.cwd(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    pass_fds=(lock_fd,),
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
            "evaluation_revision": expected_revision,
            "config_sha256": actual_config_sha256,
            "source_audit_path": str(audit_path),
            "source_audit_sha256": expected_source_audit_sha256,
            "batch_lock": str(lock_path),
            "completed": completed,
            "failed": failed,
            "pending": [job.run_key for job in pending],
        }

    with _exclusive_batch_lock(target) as (lock_path, lock_fd):
        return _run_locked(lock_path, lock_fd)



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the corrected stratified reevaluation queue")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reevaluation-root", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-source-audit-sha256", required=True)
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
        source_audit_path=arguments.source_audit,
        expected_revision=arguments.expected_revision,
        expected_config_sha256=arguments.expected_config_sha256,
        expected_source_audit_sha256=arguments.expected_source_audit_sha256,
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
