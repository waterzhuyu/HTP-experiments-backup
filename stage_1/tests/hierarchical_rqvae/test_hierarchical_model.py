import importlib.util
from pathlib import Path

import pytest
import torch

from hierarchical_rqvae.model import HierarchicalRQVAE
from staged_attribution.model import AttributionRQVAE

fixture_path = Path(__file__).parents[1] / 'staged_attribution' / 'test_model.py'
spec = importlib.util.spec_from_file_location('hierarchical_fixtures', fixture_path)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def model(weights=.1, depths=(1, 2, 4), chunk=256):
    torch.manual_seed(7)
    return HierarchicalRQVAE(fixtures.tiny_config(), torch.randn(6, 8), 10,
        {'gps', 'road', 'grid'}, dict(zip(['grid', 'road', 'gps'], depths)),
        dict.fromkeys(['grid', 'road', 'gps'], weights), {'grid': 5, 'road': 5},
        chunk_positions=chunk)


def test_zero_weight_matches_inherited_and_preserves_random_state():
    m = model(0).eval()
    torch.manual_seed(7)
    old = AttributionRQVAE(fixtures.tiny_config(), torch.randn(6, 8), 10,
                          frozenset({'gps', 'road', 'grid'})).eval()
    for key, value in old.state_dict().items():
        assert torch.equal(value, m.state_dict()[key]), key
    a, b = old(fixtures.complete_batch()), m(fixtures.complete_batch())
    assert torch.equal(a['codes'], b['codes'])
    assert torch.allclose(a['loss'], b['loss'])
    actual = b['prefix_vectors'][:, -1]
    decoded = m.rq_quant.encode(b['codes'][b['latent_mask']].T)
    assert torch.allclose(actual, decoded, atol=1e-7)


def test_chunk_losses_and_gradients_equal_dense():
    small, dense = model(chunk=3), model(chunk=10000)
    dense.load_state_dict(small.state_dict())
    x, y = small(fixtures.complete_batch()), dense(fixtures.complete_batch())
    assert torch.allclose(x['loss'], y['loss'], atol=1e-6)
    x['loss'].backward(); y['loss'].backward()
    for (name, a), (_, b) in zip(small.named_parameters(), dense.named_parameters()):
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=2e-5, rtol=1e-4), name


def test_auxiliary_gradient_reaches_encoder_but_not_codebooks():
    m = model().train()
    out = m.encode_views(fixtures.complete_batch())
    loss = sum(m.view_losses(out, fixtures.complete_batch()).values())
    loss.backward()
    assert m.rq_pre_quant_linear.weight.grad.abs().sum() > 0
    assert m.view_heads['road'][0].weight.grad.abs().sum() > 0
    assert all(book.codebook.weight.grad is None for book in m.rq_quant.codebooks)


def test_grid_prefix_cannot_read_future_codes():
    m = model().eval()
    batch = fixtures.complete_batch()
    out = m.encode_views(batch)
    before = m.view_losses(out, batch)
    changed = dict(out)
    changed['prefix_vectors'] = out['prefix_vectors'].clone()
    changed['prefix_vectors'][:, 1:] += 5
    after = m.view_losses(changed, batch)
    assert torch.equal(before['grid'], after['grid'])
    assert not torch.allclose(before['road'], after['road'])


def test_padding_is_ignored_by_head_targets():
    m = model().eval()
    batch = fixtures.complete_batch()
    out = m.encode_views(batch)
    before = m.view_losses(out, batch)
    for key in ['gps', 'grid', 'point_road']:
        batch[key] = batch[key].clone()
        batch[key][1, 8] = 999
    after = m.view_losses(out, batch)
    for key in before:
        assert torch.equal(before[key], after[key])


@pytest.mark.parametrize('depths', [(0, 2, 4), (1, 2, 5)])
def test_invalid_depth_rejected(depths):
    with pytest.raises(ValueError, match='prefix depth'):
        model(depths=depths)
