"""Public synthetic populations exercise the full analysis path, not experiment results."""
import json
from pathlib import Path
import numpy as np
import pytest
from hierarchical_rqvae.preflight import sha256
from independent_latent_sid.analyze_results import analyze,CELLS,STAGES


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def synthetic(root):
    ids=list(range(1024));pre=root/'preflight/manifest.json';write(pre,{'validation_ids':ids})
    config=root/'config.json';write(config,{'ablation':{'preflight_path':str(pre.parent),'preflight_sha256':sha256(pre)}})
    for c,scale in [('HV',.8),('LF',1.),('H1',1.1),('B2',1.2)]:
        run=root/'pilot/porto/42'/c/'attempt_001'
        write(run/'stage1/training_progress.json',{'status':'completed','step':1280,'epochs':20})
        write(run/'stage2/training_progress.json',{'status':'completed','steps_completed':1024})
        write(run/'stage1/stage1_metrics.json',{'population':1024,'codebook':[{'layer':i,'active_fraction':1.,'perplexity_fraction':1.,'max_share':.1} for i in range(4)]})
        write(run/'stage2/stage2_metrics.json',{'teacher_forced':{'population':1024,'denominators':dict.fromkeys([f'layer_{i}' for i in range(4)],1024),'group_nll':dict.fromkeys([f'layer_{i}' for i in range(4)],.5)}})
        write(run/'diagnostics/information.json',{'unigram_nll_per_layer':[1.]*4,'validation':{'events':1024}})
        write(run/'diagnostics/probes/metrics.json',[{'representation':r,'depth':d,'view':v,'epochs':10} for r in ('prefixes','layers') for d in range(1,5) for v in ('grid','road','gps')])
        write(run/'diagnostics/sampling.json',{})
        write(run/'diagnostics/gps_errors.json',{g:[{'traj_id':i,'matched_progress_ade_m':float(scale),'fde_m':float(scale),'length_absolute_error':0} for i in ids] for g in ('g00','g10','g01','g11')})
        metrics={}
        for g,offset in [('g00',0.),('g10',.01),('g01',.02),('g11',.03)]:
            rows=[{'traj_id':i,'valid':True,'penalized_error':scale*(.1+i/10000+offset)} for i in reversed(ids)]
            metrics[g]={'population':1024,'population_mean_penalized_error':float(np.mean([r['penalized_error'] for r in rows])),'per_trajectory':rows}
        write(run/'stage4/stage4_metrics.json',metrics)
        artifacts={str(p.relative_to(run)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in run.rglob('*') if p.is_file()}
        stages={s:{'status':'completed','artifacts':{}} for s in STAGES};stages['diagnostics']['artifacts']=artifacts
        write(run/'run_manifest.json',{'status':'completed','cell_id':c,'revision':'synthetic','config_sha256':sha256(config),'preflight_sha256':sha256(pre),'seed':42,'evaluation_split':'validation_from_official_train','budget':'pilot','stages':stages})
    return config


def test_full_known_effect_then_missing_run(tmp_path):
    config=synthetic(tmp_path)
    result=analyze(tmp_path,config,tmp_path/'result.json')
    assert result['status']=='complete_single_seed_analysis'
    assert abs(result['contrasts']['HV-LF']['g11']['relative_improvement']-.2)<1e-12
    assert result['contrasts']['HV-LF']['g11']['n_pairs']==1024
    assert result['cells']['HV']['nll_context_gain']['gain_over_unigram_nats_per_token']==.5
    assert result['cells']['HV']['gps']['g11']['successful']==1024
    p=tmp_path/'pilot/porto/42/B2/attempt_001/run_manifest.json';m=json.loads(p.read_text());m['status']='running';write(p,m)
    incomplete=analyze(tmp_path,config,tmp_path/'incomplete.json')
    assert incomplete['status']=='incomplete'
    assert 'contrasts' not in incomplete
    assert 'B2' in incomplete['pending']


def test_changed_artifact_rejected(tmp_path):
    config=synthetic(tmp_path)
    p=tmp_path/'pilot/porto/42/HV/attempt_001/stage4/stage4_metrics.json';p.write_text('{}')
    with pytest.raises(ValueError,match='artifact mismatch'):analyze(tmp_path,config,tmp_path/'invalid.json')
