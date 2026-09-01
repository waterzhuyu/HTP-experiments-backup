from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path


class ImmutableArtifactError(RuntimeError):
    """Raised when reproducible evidence would overwrite different content."""


def canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_immutable_artifact_set(
    artifacts: Mapping[str | Path, bytes],
    *,
    label: str,
) -> tuple[Path, ...]:
    """Create a related evidence set without replacing any differing artifact.

    Existing byte-identical files are an idempotent replay. Every destination is
    preflighted before a missing file is created, so an ordinary rerun cannot
    leave half of a report updated. Each new file is published with a hard link
    from an exclusive, fsynced temporary file; a concurrent differing writer is
    rejected.
    """

    if not artifacts:
        raise ImmutableArtifactError(f"{label} artifact set is empty")
    normalized: list[tuple[Path, bytes]] = []
    seen: set[Path] = set()
    for raw_path, content in artifacts.items():
        path = Path(raw_path)
        if path in seen:
            raise ImmutableArtifactError(f"duplicate {label} artifact path: {path}")
        if not isinstance(content, bytes):
            raise TypeError(f"{label} artifact content must be bytes: {path}")
        seen.add(path)
        normalized.append((path, content))

    missing: list[tuple[Path, bytes]] = []
    for path, content in normalized:
        if path.is_symlink():
            raise ImmutableArtifactError(
                f"immutable {label} artifact must not be a symbolic link: {path}"
            )
        if path.is_file():
            if path.read_bytes() != content:
                raise ImmutableArtifactError(
                    f"refusing to replace different immutable {label} artifact: {path}"
                )
            continue
        if path.exists():
            raise ImmutableArtifactError(
                f"immutable {label} artifact is not a regular file: {path}"
            )
        missing.append((path, content))

    for path, content in missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise ImmutableArtifactError(
                f"stale immutable {label} temporary file exists: {temporary}"
            ) from error
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != content:
                raise ImmutableArtifactError(
                    f"concurrent immutable {label} artifact mismatch: {path}"
                )
        finally:
            temporary.unlink()
        _fsync_directory(path.parent)

    return tuple(path for path, _ in normalized)
