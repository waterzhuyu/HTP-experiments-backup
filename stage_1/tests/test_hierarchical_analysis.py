import numpy as np
import pytest
from hierarchical_rqvae.analyze import paired, shapley, holm, arrays


def test_exact_paired_improvement_and_identical_control():
    w=np.random.default_rng(12).multinomial(4,[.25]*4,size=2000)
    base=np.array([1.,2.,4.,8.])
    result=paired(base,.9*base,w)
    assert result['relative_improvement']==pytest.approx(.1)
    assert result['relative_improvement_ci95']==pytest.approx([.1,.1])
    assert paired(base,base,w)['difference_ci95']==[0.,0.]


def test_shapley_interaction_allocation():
    a=np.array([.1,.2]); out=shapley(dict(g00=a,g10=a+1,g01=a+2,g11=a+7))
    assert out['sid']==pytest.approx([3,3])
    assert out['sampling']==pytest.approx([4,4])
    assert out['interaction']==pytest.approx([4,4])


def test_failure_penalty_and_population_checks():
    m={'population':2,'population_mean_penalized_error':.6,'per_trajectory':[
        {'traj_id':8,'valid':False,'penalized_error':1}, {'traj_id':2,'valid':True,'penalized_error':.2}]}
    assert arrays(m,[2,8],10)==pytest.approx([.2,10])
    with pytest.raises(ValueError): arrays(m,[2,9])
    m['per_trajectory'][0]['traj_id']=2
    with pytest.raises(ValueError): arrays(m,[2,8])


def test_holm_known_family():
    assert holm([.04,.001,.02])==pytest.approx([.04,.003,.04])
