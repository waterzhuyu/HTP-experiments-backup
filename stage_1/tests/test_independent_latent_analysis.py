import torch
from independent_latent_sid.inspect_latents import geometry,summarize

def test_geometry_detects_constant_and_rank_one():
    assert geometry(torch.ones(20,4))['effective_rank']==0
    x=torch.arange(20.)[:,None].expand(-1,4)
    assert abs(geometry(x)['effective_rank']-1)<1e-7

def test_effective_usage_and_residual_cancellation():
    q=torch.tensor([1.,-1.,1.,-1.])[None,:,None].expand(20,-1,4)
    s=q.cumsum(1);z={v:torch.ones(20,4) for v in ('grid','road','gps')}
    r=summarize(z,s,q,torch.zeros(20,4,dtype=torch.long),[32,64,128,256])
    assert all(l['perplexity']==1 for l in r['layers'])
    assert r['layers'][1]['cosine_code_previous_prefix']==-1
    assert r['prefix_target_mse']['grid']==[0.,1.,0.,1.]
