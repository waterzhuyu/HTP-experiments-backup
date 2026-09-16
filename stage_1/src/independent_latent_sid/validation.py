import json
from pathlib import Path

def validate_plan(plan):
    reference=json.loads(Path('stage_1/configs/codebook_ablation/base.json').read_text())
    for key in ('data','tokenizer_training','llm_training','generation','model'):
        if plan[key] != reference[key]:
            raise ValueError(f'unregistered configuration change: {key}')
    expected=dict(reference['cells'])
    for name,construction in [('LF','late_fusion'),('HV','conditional')]:
        expected[name]={**reference['cells']['H1'],'construction':construction}
    if plan['cells'] != expected:
        raise ValueError('unregistered cells')
    if plan['ablation']['preflight_sha256'] != reference['ablation']['preflight_sha256']:
        raise ValueError('split identity changed')
    return True
