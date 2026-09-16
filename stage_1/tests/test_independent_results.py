import numpy as np
import pytest
from independent_latent_sid.analyze_results import require_population
from hierarchical_rqvae.analyze import arrays,paired,shapley


def test_population_rejects_duplicate_and_missing():
    require_population([{'traj_id':2},{'traj_id':1}],[1,2])
    with pytest.raises(ValueError):require_population([{'traj_id':1},{'traj_id':1}],[1,2])
    with pytest.raises(ValueError):require_population([{'traj_id':1}],[1,2])


def test_invalid_population_penalty_not_success_filter():
    metric={'population':2,'population_mean_penalized_error':.55,'per_trajectory':[{'traj_id':2,'valid':False,'penalized_error':1.},{'traj_id':1,'valid':True,'penalized_error':.1}]}
    assert arrays(metric,[1,2]).tolist()==[.1,1.]
    assert arrays(metric,[1,2],10).tolist()==[.1,10.]


def test_paired_direction_and_shapley():
    w=np.array([[1,1],[2,0],[0,2]])
    r=paired([1,2],[.5,1],w)
    assert r['relative_improvement']==.5
    assert r['relative_improvement_ci95']==[.5,.5]
    out=shapley(dict(zip(('g00','g10','g01','g11'),map(np.array,([1,2],[2,4],[3,5],[5,8])))))
    assert np.allclose(out['sid']+out['sampling'],out['total'])


def test_completion_rejects_short_training_budget(tmp_path,monkeypatch):
    import json
    import independent_latent_sid.analyze_results as a
    m={'cell_id':'HV','config_sha256':'c','preflight_sha256':'p','seed':42,'budget':'pilot','stages':dict.fromkeys(a.STAGES,{})}
    monkeypatch.setattr(a,'verify_attempt',lambda path,reference:(m,3))
    (tmp_path/'stage1').mkdir();(tmp_path/'stage2').mkdir()
    (tmp_path/'stage1/training_progress.json').write_text(json.dumps({'status':'completed','step':4,'epochs':1}))
    (tmp_path/'stage2/training_progress.json').write_text(json.dumps({'status':'completed','steps_completed':2}))
    with pytest.raises(ValueError,match='tokenizer budget'):a.verify_completed(tmp_path,None,'c','p','HV')
    (tmp_path/'stage1/training_progress.json').write_text(json.dumps({'status':'completed','step':1280,'epochs':20}))
    with pytest.raises(ValueError,match='LLM budget'):a.verify_completed(tmp_path,None,'c','p','HV')
    (tmp_path/'stage2/training_progress.json').write_text(json.dumps({'status':'completed','steps_completed':1024}))
    assert a.verify_completed(tmp_path,None,'c','p','HV')[1]==3
