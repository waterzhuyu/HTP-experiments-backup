import numpy as np
import pytest
from codebook_ablation.domains import compare,DOMAIN_METRICS,validate_support

def test_domain_direction_is_candidate_minus_baseline():
    points={'B2':{m:np.array([.2]) for m in DOMAIN_METRICS},'H1':{m:np.array([.3]) for m in DOMAIN_METRICS}}
    draws={'B2':{m:np.full(20,.2) for m in DOMAIN_METRICS},'H1':{m:np.full(20,.3) for m in DOMAIN_METRICS}}
    rows=compare(points,draws)
    assert len(rows)==8
    assert all(np.isclose(r['difference_regret'],.1) and np.allclose(r['ci95'],[.1,.1]) for r in rows)

def test_changed_frozen_support_refused(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{}')
    with pytest.raises(ValueError,match='support changed'):validate_support(p)
