from __future__ import annotations

from copy import deepcopy

import pytest

from staged_attribution.config import ConfigError, ExperimentConfig


def valid_mapping() -> dict[str, object]:
    return {
        "experiment_id": "staged_attribution_v1",
        "cities": ["porto", "chengdu"],
        "seeds": [42, 314159, 20260819],
        "rq_vocab_sizes": [32, 64, 128, 256],
        "e_variants": {
            "e0": ["gps"],
            "e1": ["road"],
            "e2": ["grid"],
            "e3": ["gps", "road"],
            "e4": ["gps", "grid"],
            "e5": ["road", "grid"],
            "e6": ["gps", "road", "grid"],
        },
        "q_variants": {
            "q0": {
                "usage": 0.0,
                "invariance": 0.0,
                "variance": 0.0,
                "covariance": 0.0,
                "auxiliary": 0.0,
            },
            "q1": {
                "usage": 0.05,
                "invariance": 0.0,
                "variance": 0.0,
                "covariance": 0.0,
                "auxiliary": 0.0,
            },
            "q2": {
                "usage": 0.05,
                "invariance": 0.05,
                "variance": 0.05,
                "covariance": 0.005,
                "auxiliary": 0.0,
            },
            "q3": {
                "usage": 0.05,
                "invariance": 0.05,
                "variance": 0.05,
                "covariance": 0.005,
                "auxiliary": 0.10,
            },
        },
        "budgets": {
            "smoke": {
                "stage1_epochs": 1,
                "stage1_train_trajectories": 32,
                "stage1_test_trajectories": 16,
                "stage1_batch_size": 8,
                "stage2_epochs": 1,
                "stage2_max_steps": 2,
                "stage3_trajectories": 8,
            },
            "controlled": {
                "stage1_epochs": 20,
                "stage1_train_trajectories": "full",
                "stage1_test_trajectories": "full",
                "stage1_batch_size": 512,
                "stage2_epochs": 2,
                "stage2_max_steps": -1,
                "stage3_trajectories": 1024,
            },
            "confirmation": {
                "stage1_epochs": 100,
                "stage1_train_trajectories": "full",
                "stage1_test_trajectories": "full",
                "stage1_batch_size": 512,
                "stage2_epochs": 2,
                "stage2_max_steps": -1,
                "stage3_trajectories": "full",
            },
        },
        "paths": {
            "model_root": "/workspace/guozuyu/HTP-main/models/Qwen3-1.7B",
            "data_root": "/workspace/guozuyu/HTP-main/traj_dataset",
            "artifact_root": "/workspace/guozuyu/HTP-staged-attribution-artifacts",
        },
    }


def test_core_cells_deduplicate_e6_q0() -> None:
    config = ExperimentConfig.from_dict(valid_mapping())

    assert [cell.cell_id for cell in config.core_cells()] == [
        "e0_q0",
        "e1_q0",
        "e2_q0",
        "e3_q0",
        "e4_q0",
        "e5_q0",
        "e6_q0",
        "e6_q1",
        "e6_q2",
        "e6_q3",
    ]
    assert config.core_cells()[-1].views == frozenset({"gps", "road", "grid"})


def test_adaptive_cross_product_cells_reuse_frozen_e_and_q_specs() -> None:
    config = ExperimentConfig.from_dict(valid_mapping())

    cell = config.cell("e3_q2")

    assert cell.cell_id == "e3_q2"
    assert cell.views == frozenset({"gps", "road"})
    assert cell.q_weights == config.q_variants["q2"]
    assert len(config.all_cells()) == 7 * 4
    with pytest.raises(ConfigError, match="unknown E×Q cell"):
        config.cell("e7_q2")
    smoke = config.budgets["smoke"]
    assert smoke.stage1_train_trajectories == 32
    assert smoke.stage1_test_trajectories == 16
    assert smoke.stage1_batch_size == 8
    assert smoke.stage2_train_trajectories == "full"
    assert smoke.stage2_subset_seed == 20260826
    assert smoke.stage2_batch_size == 16
    assert smoke.stage2_gradient_accumulation == 1
    assert smoke.stage2_effective_batch_size == 16
    assert config.budgets["controlled"].stage1_train_trajectories == "full"


def test_stage2_resource_controls_are_explicit_and_preserve_effective_batch() -> None:
    mapping = valid_mapping()
    controlled = mapping["budgets"]["controlled"]
    controlled.update(
        {
            "stage2_train_trajectories": 64_000,
            "stage2_subset_seed": 20260826,
            "stage2_batch_size": 8,
            "stage2_gradient_accumulation": 2,
        }
    )

    budget = ExperimentConfig.from_dict(mapping).budgets["controlled"]

    assert budget.stage2_train_trajectories == 64_000
    assert budget.stage2_subset_seed == 20260826
    assert budget.stage2_batch_size == 8
    assert budget.stage2_gradient_accumulation == 2
    assert budget.stage2_effective_batch_size == 16


@pytest.mark.parametrize("bad_view", ["poi", "category", "text"])
def test_rejects_out_of_scope_views(bad_view: str) -> None:
    mapping = valid_mapping()
    mapping["e_variants"] = {**mapping["e_variants"], "e7": [bad_view]}

    with pytest.raises(ConfigError, match="unsupported view"):
        ExperimentConfig.from_dict(mapping)


def test_rejects_changed_rq_protocol() -> None:
    mapping = valid_mapping()
    mapping["rq_vocab_sizes"] = [256]

    with pytest.raises(ConfigError, match="RQ vocabulary"):
        ExperimentConfig.from_dict(mapping)


def test_rejects_missing_downstream_stage_budget() -> None:
    mapping = valid_mapping()
    del mapping["budgets"]["controlled"]["stage2_epochs"]

    with pytest.raises(ConfigError, match="stage2_epochs"):
        ExperimentConfig.from_dict(mapping)


@pytest.mark.parametrize("field", ["stage1_train_trajectories", "stage1_test_trajectories"])
def test_rejects_invalid_stage1_population_budget(field: str) -> None:
    mapping = valid_mapping()
    mapping["budgets"]["smoke"][field] = 0

    with pytest.raises(ConfigError, match=field):
        ExperimentConfig.from_dict(mapping)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage2_train_trajectories", 0),
        ("stage2_train_trajectories", "partial"),
        ("stage2_subset_seed", True),
        ("stage2_subset_seed", 1.5),
        ("stage2_batch_size", 0),
        ("stage2_batch_size", True),
        ("stage2_gradient_accumulation", 0),
        ("stage2_gradient_accumulation", True),
    ],
)
def test_rejects_invalid_stage2_resource_control(field: str, value: object) -> None:
    mapping = valid_mapping()
    mapping["budgets"]["controlled"][field] = value

    with pytest.raises(ConfigError, match=field):
        ExperimentConfig.from_dict(mapping)


def test_rejects_empty_or_duplicate_seeds() -> None:
    for seeds in ([], [42, 42]):
        mapping = valid_mapping()
        mapping["seeds"] = seeds
        with pytest.raises(ConfigError, match="seeds"):
            ExperimentConfig.from_dict(mapping)


def test_rejects_duplicate_view_masks() -> None:
    mapping = deepcopy(valid_mapping())
    mapping["e_variants"]["e1"] = ["gps"]

    with pytest.raises(ConfigError, match="duplicate view mask"):
        ExperimentConfig.from_dict(mapping)
