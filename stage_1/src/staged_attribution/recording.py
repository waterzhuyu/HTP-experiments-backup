from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


class RecordError(RuntimeError):
    """Raised when a run record would violate provenance or state invariants."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class RunRecorder:
    MANIFEST_NAME = "run_manifest.json"
    FORMAL_BUDGETS = frozenset({"controlled", "confirmation"})

    def __init__(self, run_dir: Path, required_artifacts: Sequence[str]) -> None:
        self.run_dir = run_dir
        self.manifest_path = run_dir / self.MANIFEST_NAME
        self.required_artifacts = tuple(required_artifacts)

    @classmethod
    def create(
        cls,
        run_dir: str | Path,
        experiment_id: str,
        cell_id: str,
        city: str,
        seed: int,
        budget: str,
        command: Sequence[str],
        git_revision: str,
        git_dirty: bool,
        required_artifacts: Sequence[str],
        launch_identity: Mapping[str, object] | None = None,
    ) -> RunRecorder:
        directory = Path(run_dir)
        manifest_path = directory / cls.MANIFEST_NAME
        if manifest_path.exists():
            raise FileExistsError(f"refusing to replace existing run_manifest: {manifest_path}")
        if budget in cls.FORMAL_BUDGETS and git_dirty:
            raise RecordError(f"{budget} runs require a clean Git worktree")
        if not experiment_id or not cell_id or not city:
            raise RecordError("experiment_id, cell_id and city must be non-empty")
        if not isinstance(seed, int):
            raise RecordError("seed must be an integer")
        if not command or not all(isinstance(value, str) and value for value in command):
            raise RecordError("command must be a non-empty string sequence")
        if len(git_revision) != 40:
            raise RecordError("git_revision must be a full 40-character revision")
        required = tuple(required_artifacts)
        if not required or len(set(required)) != len(required):
            raise RecordError("required_artifacts must be a non-empty unique sequence")
        if launch_identity is not None:
            try:
                json.dumps(launch_identity, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise RecordError("launch_identity must be JSON serializable") from error
            if launch_identity.get("verified") is not True:
                raise RecordError("launch_identity must be verified")
        directory.mkdir(parents=True, exist_ok=True)
        recorder = cls(directory, required)
        created = _now()
        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "cell_id": cell_id,
            "city": city,
            "seed": seed,
            "budget": budget,
            "status": "planned",
            "command": list(command),
            "git": {"revision": git_revision, "dirty": bool(git_dirty)},
            "required_artifacts": list(required),
            "artifacts": {},
            "failure": None,
            "created_at": created,
            "started_at": None,
            "finished_at": None,
        }
        if launch_identity is not None:
            manifest["launch_identity"] = dict(launch_identity)
        recorder._write(manifest)
        return recorder

    @classmethod
    def resume(cls, run_dir: str | Path) -> RunRecorder:
        directory = Path(run_dir)
        path = directory / cls.MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(f"run_manifest does not exist: {path}")
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        return cls(directory, manifest["required_artifacts"])

    def read(self) -> dict[str, object]:
        with self.manifest_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, manifest: Mapping[str, object]) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.manifest_path)

    @staticmethod
    def _require_transition(current: str, target: str, expected: str) -> None:
        if current != expected:
            raise RecordError(f"invalid run transition {current} -> {target}")

    @staticmethod
    def _artifact(path: str | Path) -> dict[str, object]:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise RecordError(f"artifact must be an existing regular file: {resolved}")
        return {
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        }

    def start(self) -> None:
        manifest = self.read()
        self._require_transition(str(manifest["status"]), "running", "planned")
        manifest["status"] = "running"
        manifest["started_at"] = _now()
        self._write(manifest)

    def complete(self, artifacts: Mapping[str, str | Path]) -> None:
        manifest = self.read()
        self._require_transition(str(manifest["status"]), "completed", "running")
        missing = sorted(set(self.required_artifacts) - set(artifacts))
        if missing:
            raise RecordError(f"missing required artifacts: {missing}")
        recorded = {
            name: self._artifact(path)
            for name, path in artifacts.items()
        }
        manifest["artifacts"] = recorded
        manifest["status"] = "completed"
        manifest["finished_at"] = _now()
        self._write(manifest)

    def fail(
        self,
        error_type: str,
        message: str,
        traceback_text: str,
        partial_artifacts: Mapping[str, str | Path] | None = None,
    ) -> None:
        manifest = self.read()
        self._require_transition(str(manifest["status"]), "failed", "running")
        recorded = dict(manifest.get("artifacts", {}))
        for name, path in (partial_artifacts or {}).items():
            recorded[name] = self._artifact(path)
        manifest["artifacts"] = recorded
        manifest["failure"] = {
            "type": error_type,
            "message": message,
            "traceback": traceback_text,
        }
        manifest["status"] = "failed"
        manifest["finished_at"] = _now()
        self._write(manifest)
