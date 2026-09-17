import torch
import pytest
from test_independent_latent_sid import model, f

@pytest.mark.parametrize('route',['gps_only','all_views'])
def test_reconstruction_gradient_routes_without_codebook_gradient(route):
    m=model(); m.reconstruction_route=route
    out=m(f.complete_batch()); out['reconstruction_loss'].backward()
    for view in ('grid','road','gps'):
        norm=sum(float(p.grad.abs().sum()) for p in m.view_encoders[view].parameters() if p.grad is not None)
        assert (norm>0) == (route=='all_views' or view=='gps')
    assert all(p.grad is None or p.grad.abs().sum()==0 for p in m.rq_quant.parameters())

@pytest.mark.parametrize('training',[True,False])
def test_forward_codes_losses_and_initialization_unchanged(training):
    a,b=model(),model(); b.reconstruction_route='all_views'
    assert all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items())
    a.train(training); b.train(training); batch=f.complete_batch()
    torch.manual_seed(23); x=a(batch)
    torch.manual_seed(23); y=b(batch)
    assert torch.equal(x['codes'],y['codes'])
    for key in ('loss','reconstruction_loss','quant_loss','decoder_latent'):
        torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)

@pytest.mark.parametrize('view',['grid','road','gps'])
def test_allrecon_aux_still_updates_only_own_branch(view):
    m=model();m.reconstruction_route='all_views';batch=f.complete_batch()
    out=m.encode_views(batch);m.view_losses(out,batch)[view].backward()
    for v,p in m.view_projections.items():
        assert (p.weight.grad is not None and p.weight.grad.abs().sum()>0)==(v==view)
    assert all(p.grad is None for p in m.rq_quant.parameters())


def test_local_quantization_loss_unchanged_gradients():
    a,b=model(),model();b.reconstruction_route='all_views'
    d=a.rq_quant.codebooks[0].codebook.embedding_dim
    z={v:torch.randn(5,d,requires_grad=True) for v in ('grid','road','gps')}
    zz={v:x.detach().clone().requires_grad_() for v,x in z.items()}
    a.conditional_quantize(z).quant_loss.backward();b.conditional_quantize(zz).quant_loss.backward()
    for v in z:torch.testing.assert_close(z[v].grad,zz[v].grad,rtol=0,atol=0)
    for p,q in zip(a.rq_quant.parameters(),b.rq_quant.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
