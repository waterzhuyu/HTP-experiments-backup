from __future__ import annotations

import json
from pathlib import Path

import torch

from staged_attribution.model import AttributionRQVAE
from staged_attribution.train_stage1 import save_stage1_artifacts, train_epochs, train_step


def config() -> dict[str, object]:
    return {
        "dropout": 0.0,
        "Unet": {
            "ch_multi": [1, 2, 2, 4],
            "hidden_chs": 8,
            "num_res_blocks": 1,
            "road_dim": 8,
            "road_layers": 1,
        },
        "RQ_quant": {
            "n_codebooks": 4,
            "base_vocab_size": 256,
            "vocab_multi": [8, 4, 2, 1],
            "d_model": 8,
            "beta": 0.25,
        },
    }


def full_batch() -> dict[str, torch.Tensor]:
    return {
        "traj_id": torch.tensor([10, 20]),
        "gps": torch.tensor(
            [
                [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6],
                 [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0]],
                [[0.9, 0.8], [0.8, 0.7], [0.7, 0.6], [0.6, 0.5], [0.5, 0.4],
                 [0.4, 0.3], [0.3, 0.2], [0.2, 0.1], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "grid": torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4], [4, 4, 3, 3, 2, 2, 1, 1, -1]]),
        "point_road": torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4], [4, 4, 3, 3, 2, 2, 1, 1, -1]]),
        "time": torch.arange(9).float().repeat(2, 1),
        "lengths": torch.tensor([9, 8]),
        "route": torch.tensor([[0, 1], [4, -1]], dtype=torch.long),
        "route_lengths": torch.tensor([2, 1]),
        "route_geometry": torch.tensor(
            [
                [[0.0, 0.0], [0.1, 0.1]],
                [[0.1, 0.1], [0.2, 0.2]],
                [[0.8, 0.8], [0.9, 0.9]],
            ],
            dtype=torch.float32,
        ),
        "route_geometry_lengths": torch.tensor([2, 2, 2]),
        "dxy": torch.zeros(2, 9, 2),
        "road_percent": torch.zeros(2, 9),
        "route_lengths_raw": torch.tensor([2, 1]),
    }


def make_model(q_weights: dict[str, float] | None = None) -> AttributionRQVAE:
    torch.manual_seed(17)
    return AttributionRQVAE(
        config(),
        torch.randn(6, 8),
        num_grids=10,
        active_views=frozenset({"gps", "road", "grid"}),
        q_weights=q_weights,
    )


def q3_weights() -> dict[str, float]:
    return {
        "usage": 0.05,
        "invariance": 0.05,
        "variance": 0.05,
        "covariance": 0.005,
        "auxiliary": 0.1,
    }


def test_train_step_updates_model_and_reports_all_losses() -> None:
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.fusion.gps_adapter.weight.detach().clone()

    metrics = train_step(
        model,
        full_batch(),
        optimizer,
        device=torch.device("cpu"),
        sampling_seed=123,
        speed_edges=torch.tensor([0.01, 0.03, 0.1]),
        turn_edges=torch.tensor([0.1, 0.5]),
        meters_per_coordinate=(1.0, 1.0),
    )

    assert not torch.equal(before, model.fusion.gps_adapter.weight)
    assert set(metrics["q_loss_components"]) == set(q3_weights())
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics["q_loss_components"].values())
    assert metrics["loss"] > 0
    assert metrics["gradient_norm"] > 0


def test_q3_train_step_constructs_required_sampling_and_targets() -> None:
    model = make_model(q3_weights())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = train_step(
        model,
        full_batch(),
        optimizer,
        device=torch.device("cpu"),
        sampling_seed=321,
        speed_edges=torch.tensor([0.01, 0.03, 0.1]),
        turn_edges=torch.tensor([0.1, 0.5]),
        meters_per_coordinate=(1.0, 1.0),
    )

    assert metrics["sampling_enabled"] is True
    assert metrics["auxiliary_enabled"] is True
    assert metrics["q_loss_components"]["invariance"] >= 0
    assert metrics["q_loss_components"]["auxiliary"] > 0


def test_train_epochs_moves_model_to_requested_device_once() -> None:
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    requested_devices: list[torch.device] = []
    original_to = model.to

    def tracked_to(device: torch.device) -> AttributionRQVAE:
        requested_devices.append(torch.device(device))
        return original_to(device)

    model.to = tracked_to  # type: ignore[method-assign]

    train_epochs(
        model,
        [full_batch()],
        optimizer,
        device=torch.device("cpu"),
        epochs=1,
        seed=7,
        speed_edges=torch.tensor([0.01, 0.03, 0.1]),
        turn_edges=torch.tensor([0.1, 0.5]),
        meters_per_coordinate=(1.0, 1.0),
    )

    assert requested_devices == [torch.device("cpu")]


def test_train_epochs_steps_scheduler_and_records_learning_rate() -> None:
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 if step == 0 else 0.5
    )

    history = train_epochs(
        model,
        [full_batch()],
        optimizer,
        device=torch.device("cpu"),
        epochs=2,
        seed=7,
        speed_edges=torch.tensor([0.01, 0.03, 0.1]),
        turn_edges=torch.tensor([0.1, 0.5]),
        meters_per_coordinate=(1.0, 1.0),
        scheduler=scheduler,
    )

    assert scheduler.last_epoch == 2
    assert [row["learning_rate"] for row in history] == [5e-4, 5e-4]
    assert [row["global_step"] for row in history] == [1, 2]


def test_save_stage1_artifacts_is_complete_and_reloadable(tmp_path: Path) -> None:
    model = make_model()
    history = [{"epoch": 1, "loss": 1.5}]
    metrics = {"gate": {"passed": True, "reasons": []}}
    resolved = {
        "experiment_id": "exp_v1",
        "cell_id": "e6_q0",
        "city": "porto",
        "seed": 42,
    }

    artifacts = save_stage1_artifacts(
        output_dir=tmp_path,
        model=model,
        resolved_config=resolved,
        history=history,
        metrics=metrics,
    )

    assert set(artifacts) == {"resolved_config", "checkpoint", "stage1_metrics"}
    assert all(path.is_file() for path in artifacts.values())
    assert json.loads(artifacts["resolved_config"].read_text()) == resolved
    assert json.loads(artifacts["stage1_metrics"].read_text()) == metrics
    checkpoint = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["history"] == history
    assert checkpoint["resolved_config"] == resolved
    assert set(checkpoint["model"]) == set(model.state_dict())
