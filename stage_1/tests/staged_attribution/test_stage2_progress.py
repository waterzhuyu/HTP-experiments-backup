from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from staged_attribution.train_stage2 import (
    AssistantOnlyCollator,
    Stage2DataError,
    _TrainingProgressRecorder,
    _TokenizedDataset,
    _training_progress_callback,
    evaluate_teacher_forced_model,
)
from staged_attribution.run_cell import build_cell_dag


class ConstantModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, labels):
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


def test_stage2_progress_is_a_hashed_stage_artifact() -> None:
    stage2 = next(node for node in build_cell_dag("e0_q0") if node.name == "stage2")

    assert "training_progress.json" in stage2.required_artifacts
    assert "evaluation_progress.json" in stage2.required_artifacts


def test_training_progress_records_running_and_complete_state(tmp_path) -> None:
    progress = tmp_path / "training_progress.json"
    recorder = _TrainingProgressRecorder(progress, steps_total=4, progress_interval=2)

    initial = json.loads(progress.read_text())
    assert initial["status"] == "running"
    assert initial["steps_completed"] == 0
    assert initial["eta_seconds"] is None

    assert not recorder.update(1, 0.25, {"loss": 1.5, "ignored": "value"})
    assert recorder.update(2, 0.5, {"learning_rate": 0.0003})
    running = json.loads(progress.read_text())
    assert running["status"] == "running"
    assert running["steps_total"] == 4
    assert running["steps_completed"] == 2
    assert running["epoch"] == 0.5
    assert running["steps_per_second"] > 0
    assert running["eta_seconds"] is not None
    assert running["last_metrics"] == {"learning_rate": 0.0003, "loss": 1.5}

    recorder.complete(4, 1.0, {"loss": 0.75})
    completed = json.loads(progress.read_text())
    assert completed["status"] == "completed"
    assert completed["steps_completed"] == 4
    assert completed["epoch"] == 1.0
    assert completed["eta_seconds"] == 0.0
    assert completed["error"] is None


def test_training_progress_preserves_failure_and_rejects_early_complete(tmp_path) -> None:
    progress = tmp_path / "training_progress.json"
    recorder = _TrainingProgressRecorder(progress, steps_total=4, progress_interval=2)
    recorder.update(1, 0.25, {"loss": 1.5})

    with pytest.raises(Stage2DataError, match="planned optimizer-step count"):
        recorder.complete(1, 0.25)

    error = RuntimeError("synthetic training failure")
    recorder.fail(error)
    failed = json.loads(progress.read_text())
    assert failed["status"] == "failed"
    assert failed["steps_completed"] == 1
    assert failed["last_metrics"] == {"loss": 1.5}
    assert failed["error"] == {
        "type": "RuntimeError",
        "message": "synthetic training failure",
    }


def test_training_progress_callback_records_trainer_lifecycle(tmp_path) -> None:
    progress = tmp_path / "training_progress.json"
    recorder = _TrainingProgressRecorder(progress, steps_total=2, progress_interval=1)
    callback = _training_progress_callback(recorder)
    state = SimpleNamespace(global_step=0, epoch=0.0)
    control = SimpleNamespace()

    assert callback.on_train_begin(None, state, control) is control
    state.global_step = 1
    state.epoch = 0.5
    assert callback.on_step_end(None, state, control) is control
    assert callback.on_log(None, state, control, logs={"loss": 1.25}) is control
    state.global_step = 2
    state.epoch = 1.0
    assert callback.on_train_end(None, state, control) is control

    completed = json.loads(progress.read_text())
    assert completed["status"] == "completed"
    assert completed["steps_completed"] == 2
    assert completed["epoch"] == 1.0
    assert completed["last_metrics"] == {"loss": 1.25}
    assert completed["eta_seconds"] == 0.0


def test_teacher_forced_evaluation_writes_complete_progress(tmp_path) -> None:
    rows = [
        {
            "traj_id": index,
            "input_ids": [0, 1, 2],
            "attention_mask": [1, 1, 1],
            "label_start": 1,
        }
        for index in range(5)
    ]
    progress = tmp_path / "evaluation_progress.json"

    metrics = evaluate_teacher_forced_model(
        ConstantModel(),
        _TokenizedDataset(rows),
        AssistantOnlyCollator(SimpleNamespace(pad_token_id=0)),
        {"sampling": {1}, "layer_0": {2}},
        torch.device("cpu"),
        batch_size=2,
        progress_path=progress,
        progress_interval=1,
    )

    payload = json.loads(progress.read_text())
    assert metrics["population"] == 5
    assert payload["status"] == "completed"
    assert payload["batches_total"] == 3
    assert payload["batches_completed"] == 3
    assert payload["population_total"] == 5
    assert payload["population_completed"] == 5
    assert payload["supervised_tokens"] == metrics["tokens"]
    assert payload["eta_seconds"] == 0.0
    assert payload["error"] is None


class FailingModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, labels):
        raise RuntimeError("synthetic evaluation failure")


def test_teacher_forced_evaluation_preserves_failed_progress(tmp_path) -> None:
    rows = [
        {
            "traj_id": 1,
            "input_ids": [0, 1],
            "attention_mask": [1, 1],
            "label_start": 1,
        }
    ]
    progress = tmp_path / "evaluation_progress.json"

    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        evaluate_teacher_forced_model(
            FailingModel(),
            _TokenizedDataset(rows),
            AssistantOnlyCollator(SimpleNamespace(pad_token_id=0)),
            {"sampling": {1}},
            torch.device("cpu"),
            progress_path=progress,
            progress_interval=1,
        )

    payload = json.loads(progress.read_text())
    assert payload["status"] == "failed"
    assert payload["batches_completed"] == 0
    assert payload["error"] == {
        "type": "RuntimeError",
        "message": "synthetic evaluation failure",
    }
