from __future__ import annotations

import torch

from model.rq_quant import RQBottleneck
from staged_attribution.quantizer import AttributionRQBottleneck


def make_quantizer() -> AttributionRQBottleneck:
    torch.manual_seed(3)
    return AttributionRQBottleneck(
        base_n_embed=256,
        codebook_dim=4,
        rq_layer=4,
        rq_multi=[8, 4, 2, 1],
        beta=0.25,
        assignment_temperature=0.2,
    )


def test_quantizer_preserves_four_level_protocol_and_decode_roundtrip() -> None:
    quantizer = make_quantizer()
    values = torch.randn(13, 4)

    output = quantizer(values)

    assert output.codes.shape == (13, 4)
    assert output.quantized.shape == values.shape
    assert torch.isfinite(output.quant_loss)
    assert torch.isfinite(output.usage_loss)
    assert 0.0 <= float(output.usage_loss.detach()) <= 1.0
    for layer, size in enumerate([32, 64, 128, 256]):
        assert int(output.codes[:, layer].min()) >= 0
        assert int(output.codes[:, layer].max()) < size
    decoded = quantizer.encode(output.codes.T)
    assert torch.allclose(decoded, output.quantized.detach())


def test_quantizer_loads_original_htp_codebook_state() -> None:
    original = RQBottleneck(
        base_n_embed=256,
        codebook_dim=4,
        rq_layer=4,
        rq_multi=[8, 4, 2, 1],
        beta=0.25,
    )
    quantizer = make_quantizer()

    incompatible = quantizer.load_state_dict(original.state_dict(), strict=False)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for new, old in zip(quantizer.codebooks, original.codebooks):
        assert torch.equal(new.codebook.weight, old.codebook.weight)


def test_collapsed_assignment_has_positive_usage_penalty() -> None:
    quantizer = make_quantizer()
    values = torch.zeros(64, 4)

    output = quantizer(values)

    assert float(output.usage_loss.detach()) > 0.0
    assert len(output.soft_marginals) == 4
    assert all(torch.allclose(marginal.sum(), torch.tensor(1.0), atol=1e-5) for marginal in output.soft_marginals)
