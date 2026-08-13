import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "src" / "p1_fixed_time_pilot.py"
SPEC = importlib.util.spec_from_file_location("p1_fixed_time_pilot", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_track():
    times = np.arange(0, 361, 10, dtype=np.int64)
    points = np.column_stack([104.0 + times * 1e-5, np.full(len(times), 30.7)])
    return times, points


def test_fixed_time_views_share_physical_boundaries():
    times, points = synthetic_track()
    windows = MODULE.fixed_time_views(times, points, 120)
    assert len(windows) == 3
    assert [(window["start"], window["stop"]) for window in windows] == [(0, 120), (120, 240), (240, 360)]
    for window in windows:
        assert window["views"]["native"][0, 2] == 0.0
        assert window["views"]["native"][-1, 2] == 1.0
        assert window["views"]["downsample_2x"][-1, 2] == 1.0
        assert window["views"]["downsample_3x"][-1, 2] == 1.0


def test_views_have_variable_point_counts_but_one_target():
    times, points = synthetic_track()
    window = MODULE.fixed_time_views(times, points, 120)[0]
    assert len(window["views"]["native"]) == 13
    assert len(window["views"]["downsample_2x"]) == 7
    assert len(window["views"]["downsample_3x"]) == 5
    assert window["target"].shape == (18,)


def test_model_produces_one_code_tuple_per_window():
    model = MODULE.FixedTimeRQVAE()
    points = torch.randn(3, 13, 3)
    lengths = torch.tensor([13, 7, 5])
    output = model(points, lengths)
    assert output["pre"].shape == (3, 64)
    assert output["codes"].shape == (3, 2)
    assert output["reconstruction"].shape == (3, 18)


def test_consistency_loss_is_zero_for_identical_embeddings():
    embedding = torch.randn(4, 64)
    assert torch.allclose(MODULE.cosine_loss(embedding, embedding), torch.tensor(0.0), atol=1e-6)
