from __future__ import annotations

import hashlib
from dataclasses import replace
import json
from pathlib import Path

import pytest

from staged_attribution.watch_gpu_and_launch import (
    LaunchContract,
    _validate_resource_option,
    main,
    inspect_queue,
    single_instance_lock,
    validate_launch_contract,
    watch_and_launch,
)


SOURCE_CONFIG = Path("stage_1/configs/staged_attribution/experiment.json")


def _config_for_option(tmp_path: Path, option: str) -> Path:
    raw = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    raw["provenance"]["stage2_resource_option"] = option
    raw["provenance"]["candidate_only"] = False
    budget = raw["budgets"]["controlled"]
    budget.pop("stage2_subset_manifests", None)
    if option == "A":
        budget.update(
            stage2_train_trajectories="full",
            stage2_batch_size=8,
            stage2_gradient_accumulation=2,
        )
    elif option == "B":
        manifests = {}
        for city in raw["cities"]:
            path = tmp_path / f"{city}.json"
            path.write_text("{}", encoding="utf-8")
            manifests[city] = str(path)
        budget.update(
            stage2_train_trajectories=64000,
            stage2_batch_size=8,
            stage2_gradient_accumulation=2,
            stage2_subset_manifests=manifests,
        )
    elif option == "C":
        budget.update(
            stage2_train_trajectories="full",
            stage2_batch_size=16,
            stage2_gradient_accumulation=1,
        )
    else:
        raise AssertionError(option)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    return config_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(tmp_path: Path, config_path: Path, resource_option: str) -> LaunchContract:
    artifact_root = Path(json.loads(config_path.read_text())["paths"]["artifact_root"])
    experiment_id = json.loads(config_path.read_text())["experiment_id"]
    return LaunchContract(
        worktree=tmp_path,
        config_path=config_path,
        artifact_root=artifact_root / "staged_attribution" / experiment_id,
        expected_branch="locked-branch",
        expected_head="a" * 40,
        expected_config_sha256=_sha256(config_path),
        resource_option=resource_option,
    )


@pytest.mark.parametrize("option", ["A", "B", "C"])
def test_resource_option_contract_accepts_only_matching_formal_config(
    tmp_path: Path, option: str
) -> None:
    config_path = _config_for_option(tmp_path, option)
    from staged_attribution.config import ExperimentConfig

    config = ExperimentConfig.load(config_path)
    _validate_resource_option(config, option, "controlled")


def test_resource_option_contract_rejects_candidate_and_old_default(tmp_path: Path) -> None:
    from staged_attribution.config import ExperimentConfig

    raw = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    raw["provenance"].pop("stage2_resource_option", None)
    raw["provenance"].pop("stage2_resource_decision", None)
    old_budget = raw["budgets"]["controlled"]
    for field in (
        "stage2_train_trajectories",
        "stage2_subset_seed",
        "stage2_batch_size",
        "stage2_gradient_accumulation",
        "stage2_subset_manifests",
    ):
        old_budget.pop(field, None)
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stage2_resource_option"):
        _validate_resource_option(ExperimentConfig.load(old_path), "B", "controlled")

    candidate = _config_for_option(tmp_path, "B")
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    raw["provenance"]["candidate_only"] = True
    candidate.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="candidate_only"):
        _validate_resource_option(ExperimentConfig.load(candidate), "B", "controlled")


def test_option_b_requires_explicit_existing_city_manifests(tmp_path: Path) -> None:
    from staged_attribution.config import ExperimentConfig

    config_path = _config_for_option(tmp_path, "B")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["budgets"]["controlled"]["stage2_subset_manifests"].pop("chengdu")
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="cover exactly"):
        ExperimentConfig.load(config_path)

    config_path = _config_for_option(tmp_path, "B")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    Path(raw["budgets"]["controlled"]["stage2_subset_manifests"]["porto"]).unlink()
    with pytest.raises(RuntimeError, match="manifest files are missing"):
        _validate_resource_option(ExperimentConfig.load(config_path), "B", "controlled")


def test_launch_contract_rejects_git_config_and_dirty_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config_for_option(tmp_path, "A")
    contract = _contract(tmp_path, config_path, "A")
    responses = {
        ("branch", "--show-current"): "locked-branch",
        ("rev-parse", "HEAD"): "a" * 40,
        ("status", "--porcelain"): "",
    }
    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch._git",
        lambda _worktree, *arguments: responses[arguments],
    )

    validate_launch_contract(contract)

    responses[("status", "--porcelain")] = " M experiment.json"
    with pytest.raises(RuntimeError, match="dirty"):
        validate_launch_contract(contract)
    responses[("status", "--porcelain")] = ""
    config_path.write_text(config_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA256 drifted"):
        validate_launch_contract(contract)


def test_single_instance_lock_rejects_duplicate_holder(tmp_path: Path) -> None:
    lock_path = tmp_path / "watcher.lock"
    with single_instance_lock(lock_path):
        with pytest.raises(RuntimeError, match="another GPU watcher"):
            with single_instance_lock(lock_path):
                pass


def test_inspect_queue_classifies_manifest_evidence_without_process_claims(
    tmp_path: Path,
) -> None:
    from staged_attribution.config import ExperimentConfig

    config_path = _config_for_option(tmp_path, "A")
    config = ExperimentConfig.load(config_path)
    artifact_root = tmp_path / "artifacts"
    contract = replace(
        _contract(tmp_path, config_path, "A"),
        artifact_root=artifact_root,
        budget="smoke",
    )
    cells = ["e0_q0", "e1_q0", "e2_q0", "e3_q0", "e4_q0"]

    completed_root = artifact_root / "smoke/porto/seed_42/e0_q0"
    failed_old = completed_root / "attempt_001"
    (failed_old / "stage1").mkdir(parents=True)
    (failed_old / "run_manifest.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    (failed_old / "stage1/complete.json").write_text("{}", encoding="utf-8")
    completed = completed_root / "attempt_002"
    for stage in ("stage1", "stage2", "stage4"):
        (completed / stage).mkdir(parents=True)
        (completed / stage / "complete.json").write_text("{}", encoding="utf-8")
    (completed / "run_manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )

    running = artifact_root / "smoke/porto/seed_42/e1_q0/attempt_001"
    (running / "stage1").mkdir(parents=True)
    (running / "run_manifest.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    (running / "stage1/complete.json").write_text("{}", encoding="utf-8")

    failed = artifact_root / "smoke/porto/seed_42/e2_q0/attempt_001"
    (failed / "stage1").mkdir(parents=True)
    failure = {"type": "RuntimeError", "message": "fixture failure"}
    (failed / "run_manifest.json").write_text(
        json.dumps({"status": "failed", "failure": failure}), encoding="utf-8"
    )
    (failed / "stage1/complete.json").write_text("{}", encoding="utf-8")

    incomplete = artifact_root / "smoke/porto/seed_42/e3_q0/attempt_001"
    incomplete.mkdir(parents=True)

    result = inspect_queue(config, contract, cells)

    assert len(result["completed"]) == 1
    assert result["completed"][0]["attempt"].endswith("attempt_002")
    assert result["pending"] == [
        "smoke/porto/seed_42/e1_q0",
        "smoke/porto/seed_42/e2_q0",
        "smoke/porto/seed_42/e3_q0",
        "smoke/porto/seed_42/e4_q0",
    ]
    assert [item["run_key"] for item in result["running_manifests"]] == [
        "smoke/porto/seed_42/e1_q0"
    ]
    assert result["failed_manifests"][0]["failure"] == failure
    assert result["incomplete_attempts"][0]["reason"] == "missing_run_manifest"
    assert result["uncreated"] == ["smoke/porto/seed_42/e4_q0"]
    assert result["attempt_count"] == 5
    assert result["stage_marker_counts"] == {"stage1": 3, "stage2": 1, "stage4": 1}


def test_validate_only_never_calls_runner_or_creates_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config_for_option(tmp_path, "A")
    contract = replace(
        _contract(tmp_path, config_path, "A"),
        artifact_root=tmp_path / "artifacts",
    )
    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.validate_launch_contract",
        lambda _contract: __import__(
            "staged_attribution.config", fromlist=["ExperimentConfig"]
        ).ExperimentConfig.load(config_path),
    )
    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.query_gpus",
        lambda: [
            {"index": 0, "memory_total": 24576, "memory_used": 1000, "utilization": 5}
        ],
    )
    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.run_parallel_experiment",
        lambda *args, **kwargs: pytest.fail("runner must not be called"),
    )
    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.active_formal_processes",
        lambda _contract: pytest.fail("validate-only must not inspect launch processes"),
    )

    result = watch_and_launch(contract, validate_only=True)

    assert result["safe_gpus"] == [0]
    assert result["pending_count"] == 60
    assert result["running_manifest_count"] == 0
    assert result["failed_manifest_count"] == 0
    assert result["incomplete_attempt_count"] == 0
    assert result["uncreated_count"] == 60
    assert result["attempt_count"] == 0
    assert result["stage1_complete_count"] == 0
    assert result["stage2_complete_count"] == 0
    assert result["stage4_complete_count"] == 0


def test_watcher_revalidates_contract_on_every_gpu_poll_and_delegates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config_for_option(tmp_path, "A")
    contract = _contract(tmp_path, config_path, "A")
    config_module = __import__("staged_attribution.config", fromlist=["ExperimentConfig"])
    config = config_module.ExperimentConfig.load(config_path)
    validation_count = 0

    def validate(_contract):
        nonlocal validation_count
        validation_count += 1
        return config

    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.validate_launch_contract", validate
    )
    rows = [{"index": 2, "memory_total": 24576, "memory_used": 1000, "utilization": 5}]
    monkeypatch.setattr("staged_attribution.watch_gpu_and_launch.query_gpus", lambda: rows)
    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.active_formal_processes",
        lambda _contract: [],
    )
    captured = {}

    def fake_runner(*args, **kwargs):
        captured.update(kwargs)
        assert kwargs["query_gpu_fn"]() == rows
        return {"completed": [], "failed": [], "pending": []}

    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.run_parallel_experiment", fake_runner
    )

    result = watch_and_launch(contract, poll_seconds=7, maximum_parallel=3)

    assert result["failed"] == []
    assert validation_count == 3
    assert captured["wait_for_gpu"] is True
    assert captured["poll_seconds"] == 7
    assert captured["maximum_parallel"] == 3


def test_watcher_refuses_duplicate_formal_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config_for_option(tmp_path, "A")
    contract = _contract(tmp_path, config_path, "A")
    config_module = __import__("staged_attribution.config", fromlist=["ExperimentConfig"])
    config = config_module.ExperimentConfig.load(config_path)
    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.validate_launch_contract",
        lambda _contract: config,
    )
    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.active_formal_processes",
        lambda _contract: [{"pid": 123, "command": "staged_attribution.run_cell"}],
    )
    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.query_gpus",
        lambda: pytest.fail("must reject duplicate before querying GPUs"),
    )
    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.run_parallel_experiment",
        lambda *args, **kwargs: pytest.fail("must reject duplicate before runner"),
    )

    with pytest.raises(RuntimeError, match="already active"):
        watch_and_launch(contract)


def test_cli_main_delegates_the_complete_launch_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    config_path = _config_for_option(tmp_path, "A")
    contract = _contract(tmp_path, config_path, "A")
    captured = {}

    def fake_watch(observed_contract, **kwargs):
        captured["contract"] = observed_contract
        captured.update(kwargs)
        return {"contract_valid": True, "failed": []}

    monkeypatch.setattr(
        "staged_attribution.watch_gpu_and_launch.watch_and_launch", fake_watch
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch_gpu_and_launch",
            "--worktree", str(contract.worktree),
            "--config", str(contract.config_path),
            "--artifact-root", str(contract.artifact_root),
            "--expected-branch", contract.expected_branch,
            "--expected-head", contract.expected_head,
            "--expected-config-sha256", contract.expected_config_sha256,
            "--resource-option", contract.resource_option,
            "--poll-seconds", "7",
            "--validate-only",
        ],
    )

    main()

    assert captured["contract"] == contract
    assert captured["poll_seconds"] == 7
    assert captured["validate_only"] is True
    assert json.loads(capsys.readouterr().out)["contract_valid"] is True
