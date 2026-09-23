import torch

from independent_latent_sid.diagnose_motion_probes import labels, run_probe


def test_event_endpoint_labels_exclude_singletons_and_missing_roads():
    cache = {
        "mask": torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
        "targets": {
            "gps": torch.tensor([[[1., 2.], [2., 4.]] + [[0., 0.]] * 6, [[3., 4.]] + [[0., 0.]] * 7, [[2., 3.], [3., 4.]] + [[0., 0.]] * 6]),
            "road": torch.tensor([[2, 3] + [0] * 6, [4] + [0] * 7, [0, 3] + [0] * 6]),
        },
    }
    result = labels(cache, (2., 3.))
    assert result["displacement"].tolist() == [[2., 6.], [0., 0.], [2., 3.]]
    assert result["valid_motion"].tolist() == [True, False, True]
    assert result["valid_road"].tolist() == [True, False, False]
    assert result["transition"].tolist() == [1., 0., 1.]


def test_motion_probe_runs_with_train_only_normalization():
    x = torch.arange(40, dtype=torch.float32).reshape(20, 2) / 40
    y = x * 2
    mask = torch.ones(20, dtype=torch.bool)
    result = run_probe(x, x, y, y, mask, mask, "motion", y.mean(0), y.std(0), torch.device("cpu"))
    assert result["events"] == 20
    assert result["mae_vector_km"] >= 0
