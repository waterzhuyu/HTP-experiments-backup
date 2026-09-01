from __future__ import annotations

from pathlib import Path

import pytest

from staged_attribution.immutable_artifacts import (
    ImmutableArtifactError,
    write_immutable_artifact_set,
)


def test_immutable_artifact_set_is_idempotent_and_repairs_missing_peer(
    tmp_path: Path,
) -> None:
    first = tmp_path / "report" / "analysis.json"
    second = tmp_path / "report" / "analysis.md"
    artifacts = {first: b'{"ok": true}\n', second: b"# audit\n"}

    assert write_immutable_artifact_set(artifacts, label="test report") == (
        first,
        second,
    )
    assert write_immutable_artifact_set(artifacts, label="test report") == (
        first,
        second,
    )

    second.unlink()
    write_immutable_artifact_set(artifacts, label="test report")
    assert first.read_bytes() == artifacts[first]
    assert second.read_bytes() == artifacts[second]


def test_immutable_artifact_set_preflights_every_peer_before_creation(
    tmp_path: Path,
) -> None:
    first = tmp_path / "analysis.json"
    second = tmp_path / "analysis.md"
    second.write_bytes(b"old evidence\n")

    with pytest.raises(ImmutableArtifactError, match="refusing to replace"):
        write_immutable_artifact_set(
            {first: b"new json\n", second: b"new markdown\n"},
            label="test report",
        )

    assert not first.exists()
    assert second.read_bytes() == b"old evidence\n"


def test_immutable_artifact_set_rejects_stale_temporary_and_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "analysis.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_bytes(b"stale")
    with pytest.raises(ImmutableArtifactError, match="stale"):
        write_immutable_artifact_set(
            {output: b"new\n"}, label="test report"
        )
    assert not output.exists()
    temporary.unlink()

    target = tmp_path / "target.json"
    target.write_bytes(b"same\n")
    output.symlink_to(target)
    with pytest.raises(ImmutableArtifactError, match="symbolic link"):
        write_immutable_artifact_set(
            {output: b"same\n"}, label="test report"
        )
