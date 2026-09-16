import json
import pytest
from independent_latent_sid.launch import validate_smoke
from hierarchical_rqvae.preflight import sha256


def fixture(root):
    for cell in ('LF','HV'):
        run=root/'smoke/porto/42'/cell/'attempt_001';(run/'decode').mkdir(parents=True)
        artifacts={}
        for v in ('g00','g10','g01','g11'):
            p=run/'decode'/f'{v}_reconstructed.jsonl'
            p.write_text('\n'.join(json.dumps({'traj_id':i,'status':'reconstructed'}) for i in range(8)))
            artifacts[str(p.relative_to(run))]={'sha256':sha256(p)}
        stages={k:{'artifacts':{}} for k in ('stage1','tokens','stage2_data','stage2','stage3','g_attribution','decode','stage4','diagnostics')}
        stages['decode']['artifacts']=artifacts
        (run/'run_manifest.json').write_text(json.dumps({'status':'completed','config_sha256':'abc','revision':'test','stages':stages}))


def test_smoke_gate_verifies_artifacts(tmp_path):
    fixture(tmp_path)
    assert set(validate_smoke(tmp_path,'abc'))=={'LF','HV'}
    path=next(tmp_path.glob('smoke/porto/42/HV/attempt_001/decode/g00*'))
    path.write_text('changed')
    with pytest.raises(ValueError,match='changed smoke artifact'):validate_smoke(tmp_path,'abc')


def test_smoke_gate_rejects_missing_or_other_config(tmp_path):
    with pytest.raises(ValueError,match='expected one verified'):validate_smoke(tmp_path,'abc')
    fixture(tmp_path)
    with pytest.raises(ValueError,match='expected one verified'):validate_smoke(tmp_path,'different')
