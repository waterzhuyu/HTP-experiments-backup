import json
from pathlib import Path

SETTINGS = {
    'base': [32,64,128,256], 'quarter': [8,16,32,64],
    'scale4': [128,256,512,1024], 'front_matched': [256,128,64,32],
    'front_large': [4096,8192,128,256],
}

def validate_plan(plan):
    original=json.loads(Path('stage_1/configs/hierarchical_rqvae/pilot.json').read_text())
    for key in ('data','tokenizer_training','llm_training','generation','gates','evaluation'):
        if plan[key] != original[key]:
            raise ValueError(f'unapproved scientific drift: {key}')
    if plan['cells'] != {k:original['cells'][k] for k in ('B2','H1')}:
        raise ValueError('paired B2/H1 cell definitions changed')
    model=dict(plan['model']); model['vocab_sizes']=original['model']['vocab_sizes']
    if model != original['model']:
        raise ValueError('non-capacity model change')
    name=plan['ablation']['setting']
    if plan['model']['vocab_sizes'] != SETTINGS[name]:
        raise ValueError('unregistered capacity')
    if plan['ablation']['initialization'] != 'base_layer_scale_nested_common_codes_v1':
        raise ValueError('initialization changed')
    return True
