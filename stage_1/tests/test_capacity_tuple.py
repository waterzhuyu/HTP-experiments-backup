import pytest
from codebook_ablation.tuple_accuracy import build_groups,extract_correctness

class Tokenizer:
    unk_token_id=-1
    def convert_tokens_to_ids(self,tokens):
        return [10000*('abcd'.index(t[1])+1)+int(t[3:-1]) for t in tokens]

def test_largest_capacity_includes_high_sid_values():
    groups=build_groups(Tokenizer(),[4096,8192,128,256])
    labels=[14095,28191,30127,40255]
    assert extract_correctness(labels,labels,groups).all()
    pred=labels.copy();pred[1]=20000
    assert extract_correctness(pred,labels,groups).tolist()==[[True,False,True,True]]

def test_missing_and_aliased_tokens_refused():
    class Bad(Tokenizer):
        def convert_tokens_to_ids(self,t):return [-1]*len(t)
    with pytest.raises(ValueError):build_groups(Bad(),[2]*4)
    class Colliding(Tokenizer):
        def convert_tokens_to_ids(self,t):return list(range(len(t)))
    with pytest.raises(ValueError):build_groups(Colliding(),[2]*4)

def test_wrong_layer_order_not_counted_as_tuple():
    groups=build_groups(Tokenizer(),[2]*4)
    labels=[20000,10000,30000,40000]
    with pytest.raises(ValueError):extract_correctness(labels,labels,groups)
