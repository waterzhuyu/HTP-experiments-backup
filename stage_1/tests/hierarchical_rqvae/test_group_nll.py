from types import SimpleNamespace
import math
import torch
from staged_attribution.train_stage2 import evaluate_teacher_forced_model, AssistantOnlyCollator, _TokenizedDataset


def test_sid_nll_excludes_easy_delimiters():
    class FixedModel(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            logits=torch.tensor([0.,1.,2.,8.,0.]).expand(*input_ids.shape,5)
            return SimpleNamespace(logits=logits)
    rows=[{'input_ids':[0,1,3,2,3], 'attention_mask':[1]*5,'label_start':1}]
    result=evaluate_teacher_forced_model(FixedModel(),_TokenizedDataset(rows),
        AssistantOnlyCollator(SimpleNamespace(pad_token_id=0)),{'layer_0':{1},'sampling':{2}},torch.device('cpu'))
    expected=float(-torch.log_softmax(torch.tensor([0.,1.,2.,8.,0.]),0)[1])
    assert math.isclose(result['sid_nll'],expected,abs_tol=1e-6)
    assert result['sid_nll']>result['nll']
    assert result['group_nll']['layer_0']==result['sid_nll']
    assert result['denominators']=={'layer_0':1,'sampling':1}
