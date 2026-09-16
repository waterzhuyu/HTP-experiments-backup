import json
import pytest
from codebook_ablation import collect

def test_missing_runs_never_produce_inference(tmp_path,monkeypatch):
    pre=tmp_path/'pre';pre.mkdir();(pre/'manifest.json').write_text(json.dumps({'validation_ids':list(range(1024))}))
    monkeypatch.setattr(collect,'PREFLIGHT',pre)
    def forbidden(*a,**k):raise AssertionError('inference on missing runs')
    monkeypatch.setattr(collect,'summarize',forbidden)
    result=collect.collect(tmp_path/'empty')
    assert result['status']=='incomplete' and result['verified_runs']==0
    assert len(result['ledger'])==30 and 'primary' not in result

def test_duplicate_success_refused(tmp_path):
    paths=[]
    for i in range(2):
        p=tmp_path/str(i);p.write_text(json.dumps({'status':'completed'}));paths.append(p)
    with pytest.raises(ValueError,match='duplicate'):collect.unique_completed(paths)

def test_training_population_identity_checked(monkeypatch):
    monkeypatch.setattr(collect,'sha256',lambda p:'same')
    m={'seed':42,'cell_id':'B2','budget':'pilot','config_sha256':'same','preflight_sha256':'same','stages':dict.fromkeys(collect.STAGES)}
    collect.validate_identity(m,'base',42,'B2')
    for key,value in [('seed',7),('budget','smoke'),('config_sha256','other'),('preflight_sha256','other'),('stages',{})]:
        bad=dict(m);bad[key]=value
        with pytest.raises(ValueError):collect.validate_identity(bad,'base',42,'B2')
