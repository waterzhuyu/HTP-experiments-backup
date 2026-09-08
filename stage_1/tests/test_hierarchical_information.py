import numpy as np
import pytest
from hierarchical_rqvae.analyze_information import count_summary,unigram_cross_entropy


def test_collision_entropy_and_unseen_validation_codes():
    codes=np.array([[0,0],[0,0],[1,1],[1,1]])
    x=count_summary(codes,[2,2])
    assert x['unique_tuples']==2
    assert x['duplicate_event_fraction']==.5
    assert x['pair_collision_probability']==pytest.approx(1/3)
    assert x['tuple_entropy_nats']==pytest.approx(np.log(2))
    assert unigram_cross_entropy(np.array([[0],[0]]),np.array([[1]]),[2])[0]==pytest.approx(np.log(6))
    with pytest.raises(ValueError): count_summary(np.array([[2,0]]),[2,2])
