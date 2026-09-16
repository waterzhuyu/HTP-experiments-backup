import importlib.util
from pathlib import Path
import pytest
import torch
from independent_latent_sid.model import IndependentLatentSID

spec=importlib.util.spec_from_file_location('fixtures',Path(__file__).parent/'staged_attribution/test_model.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

def model(construction='conditional'):
    torch.manual_seed(7)
    return IndependentLatentSID(f.tiny_config(),torch.randn(6,8),10,{'gps','grid','road'},
        {'grid':1,'road':2,'gps':4},dict.fromkeys(['grid','road','gps'],.1),{'grid':5,'road':5},construction=construction)

def test_independent_parameters_and_control_initialization():
    a,b=model(),model('late_fusion')
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert len({next(e.parameters()).data_ptr() for e in a.view_encoders.values()})==3

@pytest.mark.parametrize('changed,unchanged',[('gps',['grid','road']),('point_road',['grid','gps']),('grid',['road','gps'])])
def test_view_isolation(changed,unchanged):
    m=model().eval();batch=f.complete_batch();a=m.encode_views(batch)
    batch[changed]=batch[changed].clone()
    if changed=='gps':batch[changed]+=2
    else:batch[changed]=(batch[changed]+1)%(6 if changed=='point_road' else 10)
    b=m.encode_views(batch)
    for v in unchanged:assert torch.equal(a['view_latents'][v],b['view_latents'][v])
    if changed=='gps':assert torch.equal(a['codes'][...,:2],b['codes'][...,:2])
    if changed=='point_road':assert torch.equal(a['codes'][...,0],b['codes'][...,0])

@pytest.mark.parametrize('view',['grid','road','gps'])
def test_aux_gradient_only_own_encoder(view):
    m=model();batch=f.complete_batch();out=m.encode_views(batch);m.view_losses(out,batch)[view].backward()
    for v,p in m.view_projections.items():
        if v==view:assert p.weight.grad.abs().sum()>0
        else:assert p.weight.grad is None or p.weight.grad.abs().sum()==0
    assert all(b.codebook.weight.grad is None for b in m.rq_quant.codebooks)

@pytest.mark.parametrize('construction',['conditional','late_fusion'])
def test_forward_backward_decode_roundtrip(construction):
    m=model(construction);batch=f.complete_batch();out=m(batch)
    assert torch.isfinite(out['loss']);out['loss'].backward()
    assert all(b.codebook.weight.grad.abs().sum()>0 for b in m.rq_quant.codebooks)
    m.eval();out=m.encode_views(batch)
    q=m.rq_quant.encode(out['codes'][out['latent_mask']].T)
    assert torch.allclose(q,out['prefix_vectors'][:,-1],atol=1e-6)
    a=m.decode(out['codes'][out['latent_mask']],out['latent_lengths'],out['lens_type'],batch)
    b=m.decode_quantized(out['prefix_vectors'][:,-1],out['latent_lengths'],out['lens_type'],batch)
    for x,y in zip(a,b):assert torch.allclose(x,y,atol=1e-6)
    clone=model(construction);clone.load_state_dict(m.state_dict(),strict=True);clone.eval()
    assert torch.equal(out['codes'],clone.encode_views(batch)['codes'])

def test_padding_not_embedded():
    m=model().eval();batch=f.complete_batch();before=m.encode_views(batch)
    for k in ('gps','grid','point_road'):
        batch[k]=batch[k].clone();batch[k][1,8]=999
    after=m.encode_views(batch)
    assert torch.equal(before['codes'],after['codes'])


def test_conditional_local_vq_has_no_cross_view_gradient():
    m=model();z={v:torch.randn(3,8,requires_grad=True) for v in ('gps','grid','road')}
    # Match fixture latent dimension, then isolate q1 sensitivity to later views.
    d=m.rq_quant.codebooks[0].codebook.embedding_dim
    z={v:torch.randn(3,d,requires_grad=True) for v in z}
    out=m.conditional_quantize(z);out.quant_loss.backward()
    assert all(v.grad.abs().sum()>0 for v in z.values())
