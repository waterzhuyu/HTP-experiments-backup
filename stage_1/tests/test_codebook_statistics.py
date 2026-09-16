import numpy as np
import pytest
from codebook_ablation.statistics import capacity_effects,summarize

def test_identity_has_zero_effect_and_zero_interaction():
    b=np.arange(1,25,dtype=float).reshape(2,3,4)
    r=summarize(b,b,replicates=100)
    assert r['mean_relative_improvement']==[0.,0.]
    assert r['interaction_vs_base']==[0.]
    assert r['all_tested_interaction_intervals_inside_margin']

def test_known_capacity_interaction_survives_paired_sampling():
    b=np.arange(1,25,dtype=float).reshape(2,3,4)
    h=b*np.array([.99,.90])[:,None,None]
    r=summarize(b,h,replicates=200)
    assert np.allclose(r['mean_relative_improvement'],[.01,.10])
    assert np.allclose(r['interaction_simultaneous_ci95'],[[.09,.09]])
    assert not r['all_tested_capacity_benefit_upper_bounds_below_margin']
    assert not r['all_tested_interaction_intervals_inside_margin']

def test_seed_disagreement_not_hidden_by_trip_replication():
    b=np.ones((2,3,30));h=b.copy();h[:,0]*=.8;h[:,1]*=1.2
    r=summarize(b,h,replicates=1000)
    assert r['ci95'][0][0]<-.05 and r['ci95'][0][1]>.05

def test_invalid_arrays_fail_closed():
    with pytest.raises(ValueError):summarize(np.zeros((2,3,4)),np.ones((2,3,4)),replicates=10)
    with pytest.raises(ValueError):summarize(np.ones((2,3,4)),np.ones((2,3,5)),replicates=10)
