"""Read-only, paired pilot analysis. Never launches training or opens held-out test."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

GROUPS = ('g00', 'g10', 'g01', 'g11')
CONTRASTS = (('B2','H1'), ('B0','B1'), ('B1','B2'), ('B2','H2'), ('B2','H3'))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def interval(draws):
    return np.quantile(draws, [.025, .975]).tolist()


def paired(base, candidate, weights):
    base, candidate = np.asarray(base), np.asarray(candidate)
    if base.shape != candidate.shape or base.ndim != 1 or not np.isfinite([base,candidate]).all():
        raise ValueError('invalid paired arrays')
    bm, cm = weights @ base / len(base), weights @ candidate / len(base)
    diff = cm - bm
    return {'baseline':float(base.mean()), 'candidate':float(candidate.mean()),
            'difference':float((candidate-base).mean()), 'difference_ci95':interval(diff),
            'relative_improvement':float(1-candidate.mean()/base.mean()) if base.mean()>0 else None,
            'relative_improvement_ci95':interval(1-cm/bm) if (bm>0).all() else None,
            'n_pairs':len(base)}


def shapley(values):
    a,b,c,d = (values[k] for k in GROUPS)
    out={'sid':.5*((b-a)+(d-c)), 'sampling':.5*((c-a)+(d-b)),
         'interaction':d-b-c+a, 'total':d-a}
    if not np.allclose(out['sid']+out['sampling'],out['total'],atol=1e-12):
        raise ValueError('Shapley efficiency violated')
    return out


def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    order=np.argsort(p)
    result=np.empty(len(p)); last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,(len(p)-rank)*p[index])); result[index]=last
    return result.tolist()


def arrays(metric, ids, penalty=1.):
    rows=metric['per_trajectory']; by={int(r['traj_id']):r for r in rows}
    if len(by)!=len(rows) or set(by)!=set(ids) or metric['population']!=len(ids):
        raise ValueError('duplicate/mismatched evaluation population')
    values=np.array([float(by[i]['penalized_error']) if by[i]['valid'] else penalty for i in ids])
    if not np.isfinite(values).all(): raise ValueError('nonfinite error')
    if penalty==1. and not np.isclose(values.mean(),metric['population_mean_penalized_error'],rtol=1e-10):
        raise ValueError('primary metric aggregation mismatch')
    return values


def verify_attempt(path, reference=None):
    m=read(path/'run_manifest.json')
    if m['status']!='completed': raise ValueError(f'{path}: incomplete')
    for key in ('revision','config_sha256','preflight_sha256','seed','evaluation_split'):
        if reference is not None and m[key]!=reference[key]: raise ValueError(f'mismatched {key}')
    count=0
    for stage in m['stages'].values():
        if stage['status']!='completed': raise ValueError('incomplete stage')
        for name, info in stage['artifacts'].items():
            f=path/name
            if not f.resolve().is_relative_to(path.resolve()): raise ValueError('escaping artifact')
            if f.stat().st_size!=info['bytes'] or digest(f)!=info['sha256']:
                raise ValueError(f'artifact mismatch: {f}')
            count+=1
    return m,count


def analyze(root, cells, n_resamples=2000, seed=271828, interventions_root=None):
    attempts={c:root/'pilot'/'porto'/'42'/c/'attempt_001' for c in cells}
    manifests={}; verified={}; metrics={}; output={}
    for c,p in attempts.items():
        manifests[c],verified[c]=verify_attempt(p,next(iter(manifests.values()),None))
        metrics[c]=read(p/'stage4/stage4_metrics.json')
    if not {'B2','H1'}<=set(cells): raise ValueError('primary comparison missing')
    preflight=read(root/'preflight/manifest.json')
    if digest(root/'preflight/manifest.json')!=manifests['B2']['preflight_sha256']:
        raise ValueError('preflight identity mismatch')
    ids=tuple(sorted(map(int,preflight['validation_ids'])))
    if len(ids)!=1024: raise ValueError('pilot requires complete 1024 validation population')
    weights=np.random.default_rng(seed).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=n_resamples)
    values={c:{g:arrays(metrics[c][g],ids) for g in GROUPS} for c in cells}
    gps={c:read(p/'diagnostics/gps_errors.json') for c,p in attempts.items()}
    for c,p in attempts.items():
        s1=read(p/'stage1/stage1_metrics.json'); s2=read(p/'stage2/stage2_metrics.json')
        gps_summary={}
        for g in GROUPS:
            rows=gps[c][g]
            if len(rows)!=len(ids) or {int(r['traj_id']) for r in rows}!=set(ids): raise ValueError('GPS population mismatch')
            ok=[r for r in rows if 'matched_progress_ade_m' in r]
            gps_summary[g]={'population':len(rows),'successful':len(ok),'failed':len(rows)-len(ok),
                'successful_only':{k:float(np.mean([r[k] for r in ok])) if ok else None for k in
                 ('matched_progress_ade_m','fde_m','dtw_sum_m','length_absolute_error')}}
        sv=shapley(values[c])
        output[c]={'elapsed_seconds':manifests[c]['elapsed_seconds'], 'codebook':s1['codebook'],
            'prefix_conditional_entropy':s1['prefix_conditional_entropy'], 'teacher_forced':s2['teacher_forced'],
            'g':{g:{k:v for k,v in metrics[c][g].items() if k!='per_trajectory'} for g in GROUPS},
            'gps':gps_summary, 'shapley':{k:{'mean':float(v.mean()),'ci95':interval(weights@v/len(ids))} for k,v in sv.items()},
            'probes':read(p/'diagnostics/probes/metrics.json'), 'sampling':read(p/'diagnostics/sampling.json')}
        if interventions_root is None:
            iv=read(p/'diagnostics/interventions/metrics.json')
            output[c]['intervention_status']='superseded_batch_dependent_do_not_interpret'
        else:
            corrected=interventions_root/c
            audit=read(corrected/'audit_manifest.json')
            if audit['status']!='completed' or audit['source_manifest_sha256']!=digest(p/'run_manifest.json'):
                raise ValueError('uncertified corrected interventions')
            for name,sha in audit['artifacts'].items():
                if digest(corrected/name)!=sha: raise ValueError('corrected artifact hash mismatch')
            iv=read(corrected/'metrics.json')
            output[c]['intervention_status']='corrected_production_singleton'
            output[c]['intervention_audit']=audit
        # Strip only per-trajectory rows and donor IDs; retain aggregate intervention evidence.
        def compact(x):
            if isinstance(x,dict):
                result={k:compact(v) for k,v in x.items() if k not in ('per_trajectory','donors','gps_successful_only')}
                if 'gps_successful_only' in x:
                    rows=x['gps_successful_only']
                    result['gps_successful_only']={'population':len(rows), 'means':{
                        key:float(np.mean([r[key] for r in rows])) if rows else None for key in
                        ('matched_progress_ade_m','fde_m','dtw_sum_m','length_absolute_error')}}
                return result
            if isinstance(x,list): return [compact(v) for v in x]
            return x
        output[c]['interventions']=compact(iv)
    contrasts={}
    for b,c in CONTRASTS:
        if b not in cells or c not in cells: continue
        contrasts[f'{c}-{b}']={'primary':paired(values[b]['g11'],values[c]['g11'],weights),
            'invalid_penalty_10':paired(arrays(metrics[b]['g11'],ids,10.),arrays(metrics[c]['g11'],ids,10.),weights),
            'g00':paired(values[b]['g00'],values[c]['g00'],weights)}
        contrasts[f'{c}-{b}']['gps_paired_successful_only']={}
        for g in GROUPS:
            rb={int(r['traj_id']):r for r in gps[b][g] if 'matched_progress_ade_m' in r}
            rc={int(r['traj_id']):r for r in gps[c][g] if 'matched_progress_ade_m' in r}
            common=sorted(rb.keys() & rc.keys())
            if not common: continue
            wg=(weights if len(common)==len(ids) else np.random.default_rng(seed).multinomial(
                len(common),np.full(len(common),1/len(common)),size=n_resamples))
            contrasts[f'{c}-{b}']['gps_paired_successful_only'][g]={
                'n_common_successes':len(common),'excluded_from_coordinate_comparison':len(ids)-len(common),
                'metrics':{k:paired([rb[i][k] for i in common],[rc[i][k] for i in common],wg) for k in
                    ('matched_progress_ade_m','fde_m','dtw_sum_m','length_absolute_error')}}
    gates={}
    base=output['B2']
    for c in cells:
        x=output[c]; failed=[]
        for row in x['codebook']:
            for key,threshold,op in [('active_fraction',.1,'min'),('perplexity_fraction',.05,'min'),('max_share',.5,'max')]:
                if (op=='min' and row[key]<threshold) or (op=='max' and row[key]>threshold):
                    failed.append(f'layer_{row["layer"]+1}:{key}={row[key]:.6g}')
        oracle=x['gps']['g00']; base_oracle=base['gps']['g00']
        if oracle['successful']!=len(ids) or base_oracle['successful']!=len(ids):
            failed.append('G00 GPS missing observations')
            ratio=None
        else:
            ratio=oracle['successful_only']['matched_progress_ade_m']/base_oracle['successful_only']['matched_progress_ade_m']
            if ratio>1.1: failed.append('G00 GPS ADE degradation >10%')
        valid=x['g']['g11']['validity_rate']; bv=base['g']['g11']['validity_rate']
        if valid<.99: failed.append('G11 validity <.99')
        if bv-valid>.01+1e-12: failed.append('G11 validity drop >.01')
        gates[c]={'eligible':not failed,'failures':failed,'g00_ade_ratio':ratio,
                  'sid_nll_ratio':x['teacher_forced']['sid_nll']/base['teacher_forced']['sid_nll']}
    primary=contrasts['H1-B2']['primary']
    repeat=gates['H1']['eligible'] and primary['relative_improvement']>=.05 and gates['H1']['sid_nll_ratio']<=1.02
    return {'status':'complete_six_cell_analysis' if set(cells)==set(('B0','B1','B2','H1','H2','H3')) else 'provisional_completed_cells',
        'analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'command':sys.argv,'training_revision':manifests['B2']['revision'], 'verified_artifact_counts':verified,
        'preflight_sha256':manifests['B2']['preflight_sha256'],'config_sha256':manifests['B2']['config_sha256'],
        'population':len(ids),'bootstrap':{'replicates':n_resamples,'seed':seed,'unit':'paired validation trajectory','confidence':.95},
        'cells':output,'contrasts':contrasts,'gates':gates,'repeat_gate':bool(repeat),
        'limitations':['single seed, held-out trips with shared roads/vehicles/dates','oracle route throughout',
                       'GPS summaries successful-only, failures separately retained','G00 GPS gate operationalized as matched-progress ADE',
                       'secondary common-support population bootstrap pending']}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cells',nargs='+',default=['B0','B1','B2','H1','H2','H3']); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interventions-root',type=Path)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('refusing overwrite of analysis record')
    result=analyze(args.root,args.cells,interventions_root=args.interventions_root); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'primary':result['contrasts']['H1-B2']['primary'],
                      'repeat_gate':result['repeat_gate'],'gates':result['gates']},indent=2))

if __name__=='__main__': main()
