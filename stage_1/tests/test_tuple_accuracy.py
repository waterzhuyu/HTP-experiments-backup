import numpy as np
import pytest
from hierarchical_rqvae.audit_tuple_accuracy import extract_correctness,correctness_summary


def test_tuple_metric_is_not_sequence_or_per_layer_accuracy():
    c=np.array([[1,1,1,1],[1,0,1,1],[0,1,1,0]],dtype=bool)
    s=correctness_summary(c)
    assert s['tuple_accuracy']==pytest.approx(1/3)
    assert s['first_wrong_layer_counts']=={'1':1,'2':1,'3':0,'4':0,'all_correct':1}
    assert s['later_layer_conditional_accuracy']['4']['earlier_any_wrong']['accuracy']==.5


def test_shifted_sid_extraction_ignores_delimiters_but_enforces_order():
    groups=[{10},{20},{30},{40}]
    gt=[-100,99,10,20,30,40,98,10,20,30,40]
    pred=[8,99,10,20,30,40,17,10,0,30,40]
    c=extract_correctness(pred,gt,groups)
    assert c.tolist()==[[True]*4,[True,False,True,True]]
    with pytest.raises(ValueError): extract_correctness([10,20],[10,20],groups)
    with pytest.raises(ValueError): extract_correctness([10,30,20,40],[10,30,20,40],groups)
