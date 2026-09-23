import torch

from independent_latent_sid.diagnose_quantization import summarize, targets


def test_conditional_targets_follow_actual_prefixes():
    output = {
        "view_latents": {
            "grid": torch.tensor([[1.0, 0.0]]),
            "road": torch.tensor([[2.0, 0.0]]),
            "gps": torch.tensor([[4.0, 0.0]]),
        },
        "prefix_vectors": torch.tensor([[[1.0, 0.0], [3.0, 0.0], [3.5, 0.0], [4.0, 0.0]]]),
    }
    actual = targets(output, "conditional")
    assert [float(item[0, 0]) for item in actual] == [1.0, 1.0, 1.0, 0.5]


def test_quantization_summary_checks_nearest_code():
    target = torch.tensor([[0.1, 0.0], [1.1, 0.0]])
    code = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    book = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    result = summarize(target, code, book)
    assert result["events"] == 2
    assert abs(result["residual_mse_per_dimension"] - 0.005) < 1e-6
