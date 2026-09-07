import numpy as np
import torch
from hierarchical_rqvae.diagnostics import progress_gps_errors, restored_view


def test_progress_alignment_handles_different_sampling_lengths():
    a=np.array([[-8.6,41.1],[-8.5,41.2]])
    b=np.array([a[0],a.mean(0),a[1]])
    result=progress_gps_errors(a,b)
    assert result['matched_progress_ade_m']<1e-7
    assert result['dtw_sum_m']<1e-5
    assert result['length_absolute_error']==1


def test_restored_view_keeps_endpoints_and_earlier_tie():
    batch={'gps':torch.arange(10).reshape(1,5,2).float(),'time':torch.tensor([[1_600_000_000+i*15 for i in range(5)]],dtype=torch.float64),
           'grid':torch.tensor([[0,1,2,3,4]]),'point_road':torch.tensor([[0,1,2,3,4]]),'lengths':torch.tensor([5])}
    out=restored_view(batch,2)
    assert torch.equal(out['gps'],batch['gps'])
    assert out['grid'].tolist()==[[0,0,2,2,4]]
    assert torch.equal(out['time'],batch['time'])


def test_frozen_probe_preserves_codes_and_masks_partial_events(tmp_path):
    from hierarchical_rqvae.diagnostics import probe_view
    torch.manual_seed(1)
    mask=torch.tensor([[True]*8,[True,True]+[False]*6])
    features=torch.randn(2,4,8)
    cache={'prefixes':features,'mask':mask,'weights':mask.float()/10,
           'targets':{'gps':torch.randn(2,8,2)}}
    original=features.clone()
    result=probe_view(cache,cache,'prefixes',0,'gps',2,tmp_path/'probe.pt',torch.device('cpu'),epochs=1,batch_events=1)
    assert result['steps']==2 and np.isfinite(result['loss'])
    assert torch.equal(features,original)
    assert result['accuracy'] is None
