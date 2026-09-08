import numpy as np
import pytest
from hierarchical_rqvae.analyze_domains import training_edges, centered_p
from staged_attribution.population_bootstrap import build_metric_contributions, population_regrets, HISTOGRAM_METRICS


def fixture():
    gps={1:np.array([[.001,.001],[.004,.001],[.007,.001]])}
    roads={1:[0,0,0]}
    rows=[{'traj_id':1,'status':'reconstructed','gps_traj':gps[1].tolist(),'road_traj':[0,0,0]}]
    geom={0:np.array([[0.,0.],[.01,0.]])}
    kwargs={'mbr':{'min_lon':0.,'max_lon':.02,'min_lat':0.,'max_lat':.02},
            'grid':{'grid_size_m':100.,'meters_per_lon':111320.,'meters_per_lat':110540.,'n_grid_x':23,'n_grid_y':23}}
    return gps,roads,rows,geom,kwargs


def test_fixed_support_retains_tails_and_failed_population():
    gps,roads,rows,geom,kwargs=fixture()
    supports={k:training_edges([0.,.001,.002]) for k in HISTOGRAM_METRICS}
    c=build_metric_contributions(gps,roads,rows,geom,'porto',histogram_edges=supports,**kwargs)
    assert all(population_regrets(c)[k][0]==pytest.approx(0.) for k in HISTOGRAM_METRICS)
    assert c.histograms['segment_distance_jsd'][0].sum()==2
    assert c.histograms['segment_distance_jsd'][0][0,-1]==2 # above train maximum
    failed=build_metric_contributions(gps,roads,[{'traj_id':1,'status':'failed'}],geom,'porto',histogram_edges=supports,**kwargs)
    assert failed.supports==c.supports
    assert all(v[0]==1. for v in population_regrets(failed).values())


def test_constant_training_support_can_detect_new_values():
    e=training_edges([0.,0.])
    h=np.histogram([0.,.001],bins=e)[0]
    assert np.count_nonzero(h)==2
    with pytest.raises(ValueError): training_edges([np.nan])


def test_dropped_values_rejected():
    gps,roads,rows,geom,kwargs=fixture()
    with pytest.raises(ValueError,match='drops observations'):
        build_metric_contributions(gps,roads,rows,geom,'porto',histogram_edges={k:[0.,1e-9] for k in HISTOGRAM_METRICS},**kwargs)


def test_centered_null_test():
    assert centered_p(np.zeros(2000),0.)==1.
    assert centered_p(np.ones(2000),1.)==pytest.approx(1/2001)
