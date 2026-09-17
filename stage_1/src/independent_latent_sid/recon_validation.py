import json
from pathlib import Path
from independent_latent_sid.validation import validate_plan as validate_original

def validate_plan(plan):
    original=json.loads(Path('stage_1/configs/independent_latent_sid/pilot.json').read_text())
    expected={**original['cells']['HV'],'reconstruction_route':'all_views'}
    if plan['cells'].get('HVAllRecon')!=expected or set(plan['cells'])!={'HVAllRecon'}:
        raise ValueError('unregistered reconstruction-route cell')
    checked={**plan,'cells':original['cells']}
    validate_original(checked)
    if plan['ablation']['baseline_revision']!='70b82db195f600e3271030ddb5e6c228e4915f31':
        raise ValueError('baseline revision changed')
    return True
