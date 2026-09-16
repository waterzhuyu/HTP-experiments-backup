from __future__ import annotations
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import torch
from hierarchical_rqvae.run import BudgetExhausted, verify_preflight
from hierarchical_rqvae.preflight import atomic_json, sha256
from codebook_ablation.backend import Backend
from codebook_ablation.validation import validate_plan
from staged_attribution.token_protocol import codebook_protocol

def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True); parser.add_argument('--preflight',required=True)
    parser.add_argument('--phase',choices=['smoke','pilot'],required=True)
    parser.add_argument('--cells',nargs='+',required=True); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    plan=json.loads(Path(args.config).read_text())
    if args.phase not in plan.get('allowed_phases', []):
        raise RuntimeError('phase is not enabled in this execution config')
    if not plan['execution_enabled'] and not args.validate_only:
        raise RuntimeError('execution disabled')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('dirty worktree: commit before execution')
    validate_plan(plan)
    preflight_manifest=verify_preflight(args.preflight)
    if sha256(Path(args.preflight)/'manifest.json') != plan['ablation']['preflight_sha256']:
        raise ValueError('frozen preflight changed')
    if args.phase=='pilot':
        audit_path=Path(args.preflight)/'alignment_audit.json'
        if not json.loads(audit_path.read_text())['passed']:
            raise ValueError('pointwise data audit has not passed')
        model_manifest=json.loads((Path(args.preflight)/'model_manifest.json').read_text())
        for path,digest in model_manifest['hashes'].items():
            if sha256(path)!=digest: raise ValueError('base model artifact changed')
        for path,digest in preflight_manifest['source_hashes'].items():
            if sha256(path)!=digest: raise ValueError('source data artifact changed')
        validate_plan(plan)
    if any(c not in plan['cells'] for c in args.cells):
        raise ValueError('unknown cell')
    if args.validate_only:
        print(json.dumps({'valid':True,'phase':args.phase,'cells':args.cells})); return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    free, total = torch.cuda.mem_get_info()
    if free < 23 * 1024**3:
        raise RuntimeError('GPU has less than 23 GiB free; wait without launching')
    torch.set_num_threads(4)
    cap=plan['budgets'][args.phase]['gpu_hours_cap']*3600
    root=Path(plan['paths']['artifact_root'])/plan['experiment_id']/args.phase/'porto'/str(args.seed)
    prior_seconds=sum(json.loads(p.read_text()).get('elapsed_seconds',0) for p in root.glob('*/attempt_*/run_manifest.json'))
    deadline=time.monotonic()+max(0,cap-prior_seconds)
    for cell in args.cells:
        existing=sorted((root/cell).glob('attempt_*'))
        if any(json.loads((p/'run_manifest.json').read_text()).get('status')=='completed' for p in existing if (p/'run_manifest.json').exists()):
            raise RuntimeError(f'{cell} already completed; do not duplicate')
        run_dir=root/cell/f'attempt_{len(existing)+1:03d}'
        run_dir.mkdir(parents=True,exist_ok=False)
        backend=Backend(args.config,args.preflight,cell,args.phase,args.seed,deadline)
        manifest={**backend.resolved,'status':'running','command':sys.argv,'pid':os.getpid(),'started_at':time.time(),'elapsed_seconds':0,'stages':{}}
        started=time.monotonic()
        try:
            for stage in ('stage1','tokens','stage2_data','stage2','stage3','g_attribution','decode','stage4','diagnostics'):
                backend.check_budget()
                for completed_stage in manifest['stages'].values():
                    for relative, evidence in completed_stage['artifacts'].items():
                        if sha256(run_dir/relative) != evidence['sha256']:
                            raise ValueError(f'artifact changed before consuming: {relative}')
                torch.cuda.reset_peak_memory_stats()
                stage_started=time.monotonic()
                manifest['current_stage']=stage
                atomic_json(run_dir/'run_manifest.json',manifest)
                print(f'{cell}: {stage}',flush=True)
                paths=getattr(backend,'_'+stage)(run_dir)
                evidence={str(p.relative_to(run_dir)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in paths.values()}
                manifest['stages'][stage]={'status':'completed','artifacts':evidence,
                    'elapsed_seconds':time.monotonic()-stage_started,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                manifest['elapsed_seconds']=time.monotonic()-started
                atomic_json(run_dir/'run_manifest.json',manifest)
                gc.collect(); torch.cuda.empty_cache()
            manifest['status']='completed'
        except Exception as error:
            manifest['status']='budget_exhausted' if isinstance(error,BudgetExhausted) else 'failed'
            manifest['error']=repr(error)
            raise
        finally:
            manifest['elapsed_seconds']=time.monotonic()-started
            atomic_json(run_dir/'run_manifest.json',manifest)


def main():
    config = Path(sys.argv[sys.argv.index('--config') + 1])
    sizes = json.loads(config.read_text())['model']['vocab_sizes']
    with codebook_protocol(sizes):
        run()

if __name__ == '__main__':
    main()
