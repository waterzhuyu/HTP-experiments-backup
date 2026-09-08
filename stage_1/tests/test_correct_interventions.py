import importlib.util
from pathlib import Path
import torch
from hierarchical_rqvae.correct_interventions import prefix_values

spec=importlib.util.spec_from_file_location('hierarchical_model_fixtures',Path(__file__).parent/'hierarchical_rqvae/test_hierarchical_model.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def test_corrected_prefix_sum_matches_production_and_omits_future():
    m=f.model().eval(); batch=f.fixtures.complete_batch()
    with torch.no_grad():
        out=m.encode_views(batch); c=out['codes'][out['latent_mask']]
        assert torch.equal(prefix_values(m,c,4),m.rq_quant.encode(c.T))
        a=m.decode(c,out['latent_lengths'],out['lens_type'],batch)
        b=m.decode_quantized(prefix_values(m,c,4),out['latent_lengths'],out['lens_type'],batch)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
        changed=c.clone(); changed[:,1:]=0
        assert torch.equal(prefix_values(m,c,1),prefix_values(m,changed,1))
