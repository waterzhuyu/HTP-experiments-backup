import json
from pathlib import Path
import pytest
from codebook_ablation import queue

def setup_pair(tmp_path,monkeypatch):
    root=tmp_path/'artifacts';pre=tmp_path/'preflight';pre.mkdir()
    (pre/'manifest.json').write_text('{}')
    monkeypatch.setattr(queue,'ROOT',root);monkeypatch.setattr(queue,'PREFLIGHT',pre)
    monkeypatch.setattr(queue,'sha256',lambda path:'hash')
    monkeypatch.setattr(queue,'verify_attempt',lambda p:(queue.read(p/'run_manifest.json'),0))
    paths=[]
    for cell in ('B2','H1'):
        p=root/f'codebook_ablation_20260916_base/smoke/porto/42/{cell}/attempt_001'
        (p/'stage4').mkdir(parents=True)
        manifest={'status':'completed','stages':{s:{} for s in queue.STAGES},'config_sha256':'hash',
                  'preflight_sha256':'hash','cell_id':cell,'seed':42}
        (p/'run_manifest.json').write_text(json.dumps(manifest))
        (p/'stage4/stage4_metrics.json').write_text(json.dumps({g:{'population':8,'validity_rate':1.0 if g=='g00' else 0.0} for g in ('g00','g10','g01','g11')}))
        paths.append(p)
    return paths

def test_queue_allows_untrained_generation_failures_but_requires_oracle(tmp_path,monkeypatch):
    paths=setup_pair(tmp_path,monkeypatch)
    assert queue.verify_pair('base','smoke',42)
    p=paths[1]/'stage4/stage4_metrics.json';m=queue.read(p);m['g00']['validity_rate']=0.875;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='oracle'):queue.verify_pair('base','smoke',42)

def test_queue_refuses_incomplete_or_mismatched_evidence(tmp_path,monkeypatch):
    paths=setup_pair(tmp_path,monkeypatch)
    p=paths[0]/'run_manifest.json';original=queue.read(p)
    for field,value in [('seed',7),('config_sha256','other'),('stages',{})]:
        m=dict(original);m[field]=value;p.write_text(json.dumps(m))
        with pytest.raises(ValueError):queue.verify_pair('base','smoke',42)
    m=dict(original);m['status']='running';p.write_text(json.dumps(m))
    assert not queue.verify_pair('base','smoke',42)
