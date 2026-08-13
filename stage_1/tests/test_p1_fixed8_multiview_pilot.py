import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "p1_fixed8_multiview_pilot.py"
SPEC = importlib.util.spec_from_file_location("p1_fixed8_multiview_pilot", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_record():
    n = 49
    return {
        "traj_id": 1,
        "user_id": 1,
        "flag": 0,
        "path": np.arange(4),
        "time_seq": np.arange(n),
        "gps_seq": np.column_stack([np.arange(n), np.arange(n)]),
        "dxy": np.column_stack([np.arange(n), np.arange(n)]),
        "percent_dist": np.linspace(0, 1, n),
    }


def test_view_removes_points_without_interpolation():
    view, indices = MODULE.make_view(synthetic_record(), 3)
    assert np.array_equal(indices, np.arange(0, 49, 3))
    assert np.array_equal(view["time_seq"], indices)


def test_fixed8_centers_use_original_indices():
    native = MODULE.token_centers(np.arange(49), 7)
    downsampled = MODULE.token_centers(np.arange(0, 49, 2), 4)
    assert np.array_equal(native, [0, 8, 16, 24, 32, 40, 48])
    assert np.array_equal(downsampled, [0, 16, 32, 48])


def test_alignment_compares_exact_common_centers():
    native = [torch.eye(7, 8)]
    sampled = [native[0][[0, 2, 4, 6]].clone()]
    loss = MODULE.aligned_cosine_loss(
        native, [np.arange(49)], sampled, [np.arange(0, 49, 2)]
    )
    assert torch.allclose(loss, torch.tensor(0.0))


if __name__ == "__main__":
    test_view_removes_points_without_interpolation()
    test_fixed8_centers_use_original_indices()
    test_alignment_compares_exact_common_centers()
    print("3 fixed8 multiview tests passed")
