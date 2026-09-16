"""Verified primary report for all five capacities and three paired training seeds."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from hierarchical_rqvae.analyze import arrays, paired, shapley, verify_attempt
from hierarchical_rqvae.preflight import atomic_json, sha256
from codebook_ablation.queue import ROOT, PREFLIGHT, ORDER, SEEDS, STAGES
from codebook_ablation.statistics import summarize

def read(path):return json.loads(Path(path).read_text())

def validate_identity(manifest,setting,seed,cell):
    config=Path(f'stage_1/configs/codebook_ablation/{setting}.json')
    expected={'seed':seed,'cell_id':cell,'budget':'pilot',
        'config_sha256':sha256(config),'preflight_sha256':sha256(PREFLIGHT/'manifest.json')}
    for key,value in expected.items():
        if manifest[key]!=value:raise ValueError(f'identity mismatch: {key}')
    if set(manifest['stages'])!=STAGES:raise ValueError('stage missing')

def unique_completed(paths):
    complete=[p for p in paths if read(p)['status']=='completed']
    if len(complete)>1:raise ValueError('ambiguous duplicate completed attempts')
    return complete[0] if complete else None

def collect(root=ROOT):
    ids=sorted(map(int,read(PREFLIGHT/'manifest.json')['validation_ids']))
    if len(ids)!=1024 or len(set(ids))!=1024:raise ValueError('wrong frozen validation population')
    ledger=[];rows=[];values={};revisions=set()
    for setting in ORDER:
        for seed in SEEDS:
            for cell in ('B2','H1'):
                paths=sorted((root/f'codebook_ablation_20260916_{setting}'/'pilot/porto'/str(seed)/cell).glob('attempt_*/run_manifest.json'))
                entry={'setting':setting,'seed':seed,'cell':cell,'attempts':[
                    {'path':str(p),'status':read(p)['status'],'error':read(p).get('error')} for p in paths]}
                path=unique_completed(paths)
                if path is None:
                    entry['verified']=False;ledger.append(entry);continue
                run=path.parent;m,count=verify_attempt(run);validate_identity(m,setting,seed,cell)
                stage1=read(run/'stage1/training_progress.json');stage2=read(run/'stage2/training_progress.json')
                if stage1['step']!=1280 or stage2['steps_completed']!=1024:
                    raise ValueError('training budget not completed')
                if stage1['status']!='completed' or stage2['status']!='completed':raise ValueError('training incomplete')
                metrics=read(run/'stage4/stage4_metrics.json')
                vectors={g:arrays(metrics[g],ids) for g in ('g00','g10','g01','g11')}
                values[(setting,seed,cell)]=vectors
                entry.update(verified=True,path=str(run),manifest_sha256=sha256(path),artifacts_verified=count,revision=m['revision'])
                ledger.append(entry);revisions.add(m['revision'])
                s1=read(run/'stage1/stage1_metrics.json');s2=read(run/'stage2/stage2_metrics.json')['teacher_forced']
                gps=read(run/'diagnostics/gps_errors.json')['g11']
                if len(gps)!=1024 or {int(r['traj_id']) for r in gps}!=set(ids):raise ValueError('ADE population mismatch')
                ade=[r['matched_progress_ade_m'] for r in gps if 'matched_progress_ade_m' in r]
                info=read(run/'diagnostics/information.json')
                row={'setting':setting,'seed':seed,'cell':cell,
                    'g11_error':float(vectors['g11'].mean()),'g00_error':float(vectors['g00'].mean()),
                    'g11_validity_rate':metrics['g11']['validity_rate'],
                    'ade_m_successful_only':float(np.mean(ade)) if ade else None,'ade_denominator':len(ade),
                    'sid_nll':s2['sid_nll'],'teacher_forced':s2,'codebooks':s1['codebook'],
                    'parameter_count':s1['parameter_count'],'information':info,
                    'probes':read(run/'diagnostics/probes/metrics.json'),
                    'shapley_mean':{k:float(v.mean()) for k,v in shapley(vectors).items()},
                    'run_path':str(run),'elapsed_seconds':m['elapsed_seconds']}
                rows.append(row)
    ready=len(rows)==30
    out={'status':'primary_analysis_ready' if ready else 'incomplete',
         'expected_runs':30,'verified_runs':len(rows),'ledger':ledger,'rows':rows,
         'training_revisions':sorted(revisions),'validation_ids':ids,
         'analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'remaining_final_report_requirements':['complete tuple restoration audit','common-training-bin domain quality analysis','full method and limitation synthesis','Notion and local final record synchronization']}
    if ready:
        if len(revisions)!=1:raise ValueError('training revisions differ; audit changes before combining')
        b=np.stack([[values[(setting,seed,'B2')]['g11'] for seed in SEEDS] for setting in ORDER])
        h=np.stack([[values[(setting,seed,'H1')]['g11'] for seed in SEEDS] for setting in ORDER])
        out['capacity_order']=ORDER;out['seed_order']=SEEDS;out['primary']=summarize(b,h)
        rng=np.random.default_rng(271828);weights=rng.multinomial(1024,np.full(1024,1/1024),size=2000)
        out['per_seed_paired']={setting:{str(seed):{g:paired(values[(setting,seed,'B2')][g],values[(setting,seed,'H1')][g],weights)
            for g in ('g00','g10','g01','g11')} for seed in SEEDS} for setting in ORDER}
        # Explicit failure-penalty sensitivity, using the same frozen population.
        sensitivity={}
        for setting in ORDER:
            for seed in SEEDS:
                for cell in ('B2','H1'):
                    run=next(Path(r['run_path']) for r in rows if (r['setting'],r['seed'],r['cell'])==(setting,seed,cell))
                    sensitivity[(setting,seed,cell)]=arrays(read(run/'stage4/stage4_metrics.json')['g11'],ids,penalty=10.)
        sb=np.stack([[sensitivity[(setting,seed,'B2')] for seed in SEEDS] for setting in ORDER])
        sh=np.stack([[sensitivity[(setting,seed,'H1')] for seed in SEEDS] for setting in ORDER])
        out['penalty10_sensitivity']=summarize(sb,sh)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('refusing overwrite')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('commit analysis before execution')
    result=collect();atomic_json(args.output,result)
    print(json.dumps({'status':result['status'],'verified_runs':result['verified_runs'],'output':str(args.output)}))

if __name__=='__main__':main()
