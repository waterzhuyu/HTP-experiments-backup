from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from staged_attribution.train_stage2 import (
    AssistantOnlyCollator,
    _TokenizedDataset,
    evaluate_teacher_forced_model,
)
from staged_attribution.run_cell import build_cell_dag


class ConstantModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, labels):
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


def test_stage2_progress_is_a_hashed_stage_artifact() -> None:
    stage2 = next(node for node in build_cell_dag("e0_q0") if node.name == "stage2")

    assert "evaluation_progress.json" in stage2.required_artifacts


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
