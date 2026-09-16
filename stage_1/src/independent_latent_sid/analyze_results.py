"""Paired, complete-population single-seed analysis; never launches experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from hierarchical_rqvae.analyze import arrays, paired, shapley, read, verify_attempt, digest
from hierarchical_rqvae.preflight import atomic_json

CELLS=('HV','LF','H1','B2')
GROUPS=('g00','g10','g01','g11')
CONTRASTS=(('LF','HV'),('H1','LF'),('H1','HV'),('B2','H1'))
STAGES={'stage1','tokens','stage2_data','stage2','stage3','g_attribution','decode','stage4','diagnostics'}


def require_population(rows, ids):
    keys=[int(r['traj_id']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=set(ids):
        raise ValueError('missing or duplicate trajectory')


def verify_completed(path, reference, config_sha, preflight_sha, cell):
    m,count=verify_attempt(path,reference)
    if m['cell_id']!=cell or m['config_sha256']!=config_sha or m['preflight_sha256']!=preflight_sha:
        raise ValueError('run identity mismatch')
    if m['seed']!=42 or m['budget']!='pilot' or set(m['stages'])!=STAGES:
        raise ValueError('not the complete single-seed pilot')
    s1=read(path/'stage1/training_progress.json');s2=read(path/'stage2/training_progress.json')
    if s1.get('status')!='completed' or s1.get('step')!=1280 or s1.get('epochs')!=20:
        raise ValueError('incomplete tokenizer budget')
    if s2.get('status')!='completed' or s2.get('steps_completed')!=1024:
        raise ValueError('incomplete LLM budget')
    return m,count


def summarize_cell(path, ids, metrics, weights):
    s1=read(path/'stage1/stage1_metrics.json');s2=read(path/'stage2/stage2_metrics.json')['teacher_forced']
    info=read(path/'diagnostics/information.json');probes=read(path/'diagnostics/probes/metrics.json')
    expected={(r,d,v) for r in ('prefixes','layers') for d in range(1,5) for v in ('grid','road','gps')}
    observed=[(x['representation'],x['depth'],x['view']) for x in probes]
    if len(observed)!=24 or set(observed)!=expected or any(x['epochs']!=10 for x in probes):
        raise ValueError('missing or incomplete frozen probes')
    if s1['population']!=len(ids) or s2['population']!=len(ids):raise ValueError('stage metric population mismatch')
    for i in range(4):
        if s2['denominators'][f'layer_{i}']!=info['validation']['events']:
            raise ValueError('SID NLL event denominator mismatch')
    ce=np.asarray(info['unigram_nll_per_layer']);group=np.asarray([s2['group_nll'][f'layer_{i}'] for i in range(4)])
    gps=read(path/'diagnostics/gps_errors.json');gps_summary={}
    for g in GROUPS:
        require_population(gps[g],ids)
        success=[x for x in gps[g] if 'matched_progress_ade_m' in x]
        gps_summary[g]={'population':len(ids),'successful':len(success),'failed':len(ids)-len(success),
            'successful_only':{k:float(np.mean([x[k] for x in success])) if success else None
                for k in ('matched_progress_ade_m','fde_m','length_absolute_error')}}
    values={g:arrays(metrics[g],ids) for g in GROUPS};sv=shapley(values)
    def compact(x):
        if isinstance(x,dict):return {k:compact(v) for k,v in x.items() if k not in ('counts','per_trajectory')}
        if isinstance(x,list):return [compact(v) for v in x]
        return x
    return {'stage1':s1,'teacher_forced':s2,'information':compact(info),'probes':probes,'gps':gps_summary,
        'sampling':read(path/'diagnostics/sampling.json'),
        'g':{g:compact(metrics[g]) for g in GROUPS},
        'nll_context_gain':{'unigram_mean_nll':float(ce.mean()),'llm_mean_layer_nll':float(group.mean()),
            'gain_over_unigram_nats_per_token':float((ce-group).mean()),'per_layer_gain':(ce-group).tolist(),
            'interpretation':'descriptive; train unigram alpha=.5 ignores route/prior layers/events; not a matched conditional baseline'},
        'shapley':{k:{'mean':float(v.mean()),'ci95':np.quantile(weights@v/len(ids),[.025,.975]).tolist()} for k,v in sv.items()}},gps


def analyze(root, config, output):
    if output.exists():raise ValueError('refusing overwrite')
    plan=read(config);preflight=Path(plan['ablation']['preflight_path'])/'manifest.json'
    ps=digest(preflight);cs=digest(config)
    if ps!=plan['ablation']['preflight_sha256']:raise ValueError('preflight changed')
    ids=tuple(sorted(map(int,read(preflight)['validation_ids'])))
    if len(ids)!=1024 or len(set(ids))!=1024:raise ValueError('wrong frozen validation population')
    paths={};pending={};manifests={};counts={}
    for c in CELLS:
        candidates=sorted((root/'pilot/porto/42'/c).glob('attempt_*/run_manifest.json'))
        completed=[p.parent for p in candidates if read(p)['status']=='completed']
        if len(completed)>1:raise ValueError('duplicate completed cell')
        if not completed:
            pending[c]=[{'path':str(p),'status':read(p)['status'],'stage':read(p).get('current_stage')} for p in candidates] or [{'status':'not_started'}]
            continue
        paths[c]=completed[0]
        manifests[c],counts[c]=verify_completed(paths[c],next(iter(manifests.values()),None),cs,ps,c)
    result={'status':'incomplete' if pending else 'complete_single_seed_analysis','pending':pending,
        'analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'command':sys.argv,'config_sha256':cs,'preflight_sha256':ps,'verified_cells':list(manifests),
        'verified_artifact_counts':counts,'source_manifests':{c:{'path':str(paths[c]/'run_manifest.json'),'sha256':digest(paths[c]/'run_manifest.json'),'revision':m['revision']} for c,m in manifests.items()}}
    if pending:
        atomic_json(output,result);return result
    weights=np.random.default_rng(271828).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    metrics={c:read(p/'stage4/stage4_metrics.json') for c,p in paths.items()}
    values={c:{g:arrays(metrics[c][g],ids) for g in GROUPS} for c in CELLS}
    cells={};gps={}
    for c,p in paths.items():cells[c],gps[c]=summarize_cell(p,ids,metrics[c],weights)
    contrasts={}
    for b,c in CONTRASTS:
        item={g:paired(values[b][g],values[c][g],weights) for g in GROUPS}
        item['g11_penalty10']=paired(arrays(metrics[b]['g11'],ids,10.),arrays(metrics[c]['g11'],ids,10.),weights)
        bm=weights@values[b]['g11']/len(ids);cm=weights@values[c]['g11']/len(ids)
        item['primary_relative_improvement_simultaneous_ci95']=np.quantile(1-cm/bm,[.05/(2*len(CONTRASTS)),1-.05/(2*len(CONTRASTS))]).tolist() if (bm>0).all() else None
        item['gps_common_success_only']={}
        for g in GROUPS:
            rb={int(r['traj_id']):r for r in gps[b][g] if 'matched_progress_ade_m' in r}
            rc={int(r['traj_id']):r for r in gps[c][g] if 'matched_progress_ade_m' in r};common=sorted(rb.keys()&rc.keys())
            if not common:continue
            w=weights if len(common)==len(ids) else np.random.default_rng(271828).multinomial(len(common),np.full(len(common),1/len(common)),size=2000)
            item['gps_common_success_only'][g]={'excluded':len(ids)-len(common),**paired([rb[i]['matched_progress_ade_m'] for i in common],[rc[i]['matched_progress_ade_m'] for i in common],w)}
        contrasts[c+'-'+b]=item
    health={}
    for c in CELLS:
        failures=[]
        for row in cells[c]['stage1']['codebook']:
            for key,t,op in [('active_fraction',.1,'min'),('perplexity_fraction',.05,'min'),('max_share',.5,'max')]:
                if (op=='min' and row[key]<t) or (op=='max' and row[key]>t):failures.append(f'layer{row["layer"]+1}:{key}')
        health[c]={'passed':not failures,'failures':failures}
    result.update(cells=cells,contrasts=contrasts,codebook_health=health,
        bootstrap={'unit':'paired validation trajectory','replicates':2000,'seed':271828,'ci95':'percentile','simultaneous':'Bonferroni four predeclared G11 contrasts; descriptive'},
        limitations=['Single seed: trajectory bootstrap does not measure training variability.',
        'Shared roads/vehicles/dates and oracle route; no cross-city or route-choice claims.',
        'Probe losses descriptive without trajectory-wise uncertainty; no causal view-exclusivity claim.',
        'GPS coordinate comparison excludes failures, primary population error retains them.',
        'No automatic improvement gate: jointly interpret primary error, G00, validity, probes and entropy.'])
    atomic_json(output,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('dirty analysis workspace')
    result=analyze(a.root,a.config,a.output)
    print(json.dumps({'status':result['status'],'verified_cells':result['verified_cells'],'pending':result['pending']},indent=2))

if __name__=='__main__':main()
