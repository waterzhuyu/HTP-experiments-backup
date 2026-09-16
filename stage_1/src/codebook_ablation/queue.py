"""Bounded two-GPU supervisor; no training result is inferred from a process exit alone."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from hierarchical_rqvae.preflight import atomic_json, sha256
from hierarchical_rqvae.analyze import verify_attempt
from codebook_ablation.validation import SETTINGS

ROOT = Path('/workspace/guozuyu/HTP-codebook-ablation-artifacts')
PREFLIGHT = Path('/workspace/guozuyu/HTP-hierarchical-rqvae-artifacts/hierarchical_rqvae_gps_grid_road_v1_20260907/preflight')
STAGES = {'stage1','tokens','stage2_data','stage2','stage3','g_attribution','decode','stage4','diagnostics'}
ORDER = ['base','front_large','front_matched','scale4','quarter']
SEEDS = [42,314159,20260907]

def read(p):
    return json.loads(Path(p).read_text())

def runs(setting,phase,seed,cell):
    return sorted((ROOT/f'codebook_ablation_20260916_{setting}'/phase/'porto'/str(seed)/cell).glob('attempt_*/run_manifest.json'))

def verify_pair(setting,phase,seed):
    for cell in ('B2','H1'):
        paths=runs(setting,phase,seed,cell)
        completed=[p for p in paths if read(p)['status']=='completed']
        if not completed:
            return False
        if len(completed)!=1:
            raise ValueError('ambiguous duplicate successful attempts')
        p=completed[0]; m,_=verify_attempt(p.parent)
        if set(m['stages'])!=STAGES:
            raise ValueError('required stage missing')
        config=Path(f'stage_1/configs/codebook_ablation/{setting}.json')
        if m['config_sha256']!=sha256(config) or m['preflight_sha256']!=sha256(PREFLIGHT/'manifest.json'):
            raise ValueError('run configuration or population changed')
        if m['seed']!=seed or m['cell_id']!=cell:
            raise ValueError('run identity mismatch')
        metrics=read(p.parent/'stage4/stage4_metrics.json')
        expected=8 if phase=='smoke' else 1024
        if any(metrics[g]['population']!=expected for g in ('g00','g10','g01','g11')):
            raise ValueError('evaluation population incomplete')
        if metrics['g00']['validity_rate']!=1:
            raise ValueError('oracle codec smoke/production failure')
    return True

def consumed_seconds():
    return sum(read(p).get('elapsed_seconds',0) for p in ROOT.glob('codebook_ablation_*/*/porto/*/*/attempt_*/run_manifest.json'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['smoke','pilot'],required=True)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1]);args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    lock=(ROOT/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('dirty worktree')
    if len(args.gpus)>2 or len(args.gpus)!=len(set(args.gpus)):raise ValueError('one or two distinct GPU indices required')
    if args.phase=='pilot':
        for setting in ORDER:
            if not verify_pair(setting,'smoke',42):raise RuntimeError(f'{setting}: smoke gate incomplete')
    tasks=[(s,seed) for seed in ([42] if args.phase=='smoke' else SEEDS) for s in ORDER]
    tasks=[(s,seed) for s,seed in tasks if not verify_pair(s,args.phase,seed)]
    # Never silently restart a nonterminal or failed attempt. Recovery is an explicit decision.
    for setting,seed in tasks:
        for cell in ('B2','H1'):
            for p in runs(setting,args.phase,seed,cell):
                if read(p)['status']!='completed':raise RuntimeError(f'Inspect previous attempt before retry: {p}')
    state={'status':'running','phase':args.phase,'pid':os.getpid(),'revision':revision,
           'started_at':time.time(),'pending':tasks,'active':[],'finished':[]}
    target=ROOT/f'supervisor_{args.phase}.json';active={};failed=False
    try:
        while tasks or active:
            for gpu,(proc,record,log) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                log.close()
                record.update(returncode=rc,finished_at=time.time())
                if rc==0:
                    record['verified']=verify_pair(record['setting'],args.phase,record['seed'])
                else:record['verified']=False
                state['finished'].append(record);del active[gpu]
                if not record['verified']:failed=True
            # Conservatively reserve the full per-pair cap for each live/new process.
            for gpu in args.gpus:
                if failed or not tasks or gpu in active:continue
                used=consumed_seconds()
                reserve=2*3600 if args.phase=='smoke' else 12*3600
                if used+(len(active)+1)*reserve>96*3600:
                    failed=True;state['error']='global GPU-hour budget reservation exhausted';break
                size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
                if size>100*1024**3:
                    failed=True;state['error']='global artifact cap exceeded';break
                if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=revision:
                    failed=True;state['error']='source revision changed while supervising';break
                if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
                    failed=True;state['error']='source dirty while supervising';break
                setting,seed=tasks.pop(0)
                cmd=['uv','run','python','-m','codebook_ablation.run','--config',f'stage_1/configs/codebook_ablation/{setting}.json',
                     '--preflight',str(PREFLIGHT),'--phase',args.phase,'--cells','B2','H1','--seed',str(seed)]
                logfile=ROOT/f'{args.phase}_{setting}_{seed}_{int(time.time())}.log'
                log=logfile.open('x')
                proc=subprocess.Popen(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdin=subprocess.DEVNULL,
                                      stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record={'setting':setting,'seed':seed,'gpu':gpu,'pid':proc.pid,'command':cmd,'log':str(logfile),'started_at':time.time()}
                active[gpu]=(proc,record,log)
            state.update(pending=tasks,active=[r for _,r,_ in active.values()],updated_at=time.time())
            atomic_json(target,state)
            if failed and not active:break
            if active:time.sleep(15)
        state['status']='needs_attention' if failed else 'completed'
        state['finished_at']=time.time();atomic_json(target,state)
        if failed:raise RuntimeError('Queue stopped; inspect supervisor and run manifests')
    except BaseException as error:
        state.update(status='needs_attention',error=repr(error),active=[r for _,r,_ in active.values()])
        atomic_json(target,state)
        # Live children are deliberately not restarted or killed on observer interruption.
        raise

if __name__=='__main__':main()
