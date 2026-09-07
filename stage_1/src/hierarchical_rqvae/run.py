"""Isolated experiment runner. The historical staged-attribution queue is never called."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hierarchical_rqvae.model import HierarchicalRQVAE
from hierarchical_rqvae.preflight import atomic_json, sha256
from staged_attribution.data import AlignedHTPDataset, load_road_context, make_collator
from staged_attribution.pipeline import PipelineBackend, _set_seed
from staged_attribution.losses import LOSS_NAMES
from staged_attribution.train_stage1 import train_step, build_cosine_warmup_scheduler, save_stage1_artifacts
from staged_attribution.stage4_metrics import evaluate_population
from staged_attribution.generate_stage3 import run_hf_generation


class BudgetExhausted(RuntimeError):
    pass


def verify_preflight(path):
    path = Path(path)
    marker = json.loads((path / 'complete.json').read_text())
    if sha256(path / 'manifest.json') != marker['manifest_sha256']:
        raise ValueError('preflight manifest hash mismatch')
    manifest = json.loads((path / 'manifest.json').read_text())
    for name, digest in manifest['cohort_hashes'].items():
        if sha256(Path(manifest['cohort']) / name) != digest:
            raise ValueError(f'cohort artifact hash mismatch: {name}')
    if set(manifest['train_ids']) & set(manifest['validation_ids']):
        raise ValueError('train/validation leakage')
    return manifest


class Backend(PipelineBackend):
    def __init__(self, config_path, preflight_path, cell_id, phase, seed, deadline):
        self.config_path = Path(config_path).resolve()
        self.plan = json.loads(self.config_path.read_text())
        self.manifest = verify_preflight(preflight_path)
        self.phase, self.seed, self.cell_id, self.city = phase, seed, cell_id, 'porto'
        cell = self.plan['cells'][cell_id]
        self.cell = SimpleNamespace(views=frozenset(cell['views']), q_weights=dict.fromkeys(LOSS_NAMES, 0.))
        self.config = SimpleNamespace(paths={k:Path(v) for k,v in self.plan['paths'].items()}, rq_vocab_sizes=self.plan['model']['vocab_sizes'])
        smoke = phase == 'smoke'
        self.budget = SimpleNamespace(
            stage1_train_trajectories=32 if smoke else 8192,
            stage1_test_trajectories=16 if smoke else 1024,
            stage1_epochs=1 if smoke else 20, stage1_batch_size=8 if smoke else 128,
            stage2_train_trajectories='full', stage2_subset_seed=self.plan['data']['split_seed'],
            stage2_subset_manifests={}, stage2_epochs=1 if smoke else 2,
            stage2_max_steps=2 if smoke else -1, stage2_batch_size=1 if smoke else 2,
            stage2_gradient_accumulation=1 if smoke else 8, stage2_effective_batch_size=1 if smoke else 16,
            stage3_trajectories=8 if smoke else 1024)
        self.device = torch.device('cuda:0')
        self.data_root = self.config.paths['data_root'] / self.city
        self.aligned_root = Path(self.manifest['cohort'])
        self._datasets, self._road_context = {}, None
        self.deadline = deadline
        self.resolved = {'experiment_id':self.plan['experiment_id'], 'cell_id':cell_id, 'city':self.city,
            'seed':seed, 'budget':phase, 'evaluation_split':'validation_from_official_train',
            'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'config_sha256':sha256(self.config_path),'preflight_sha256':sha256(Path(preflight_path)/'manifest.json'),
            'cell':cell, 'plan':self.plan}

    def _dataset(self, split):
        if split not in self._datasets:
            base = AlignedHTPDataset(self.aligned_root, self.aligned_root/f'processed_{split}_traj.pkl', split)
            n = self._requested(split)
            base.indices = np.arange(min(int(n),len(base)),dtype=np.int64)
            if len(base) != n:
                raise ValueError('insufficient frozen cohort')
            self._datasets[split] = base
        return self._datasets[split]

    def _context(self):
        if self._road_context is None:
            self._road_context = load_road_context(self.data_root/'rn/edge_info.csv',
                json.loads((self.aligned_root/'metadata.json').read_text())['mbr'], self.aligned_root/'z_score.json')
        return self._road_context

    def _new_model(self):
        _set_seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        road_features = torch.load(self.data_root/'road_emb_128d.pt', weights_only=True)
        metadata = json.loads((self.aligned_root/'metadata.json').read_text())
        cell = self.plan['cells'][self.cell_id]
        return HierarchicalRQVAE(self._model_config(),road_features,metadata['num_grids'],cell['views'],
            cell['prefix_depths'],cell['aux_weights'],self.manifest['observed_vocab'],head_seed=self.seed)

    def _load_model(self, run_dir):
        model = self._new_model()
        checkpoint = torch.load(run_dir/'stage1/stage1_model.pt',weights_only=True)
        for key in ('cell_id','preflight_sha256','config_sha256','seed'):
            if checkpoint['resolved_config'][key] != self.resolved[key]:
                raise ValueError(f'checkpoint identity mismatch: {key}')
        model.load_state_dict(checkpoint['model'],strict=True)
        return model

    def check_budget(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExhausted('phase GPU-hours cap reached')

    def _stage1(self, run_dir):
        output = run_dir/'stage1'; output.mkdir(parents=True,exist_ok=True)
        loader = self._loader('train',shuffle=True)
        model = self._new_model().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        scheduler = build_cosine_warmup_scheduler(optimizer,len(loader),self.budget.stage1_epochs,
            warmup_epochs=10,warmup_ratio=.01,minimum_ratio=.1)
        speed,turn,meters = self._auxiliary_settings(self._dataset('train'))
        history=[]; started=time.monotonic(); step=0
        try:
            for epoch in range(self.budget.stage1_epochs):
                for batch in loader:
                    self.check_budget()
                    values = train_step(model,batch,optimizer,self.device,self.seed+step,speed,turn,meters)
                    scheduler.step(); step+=1
                    if step % 10 == 0 or step == 1:
                        atomic_json(output/'training_progress.json',{'status':'running','epoch':epoch+1,
                            'step':step,'total_steps':len(loader)*self.budget.stage1_epochs,'elapsed_seconds':time.monotonic()-started,**values})
                history.append({'epoch':epoch+1,'last_step':step,'last_metrics':values})
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'epoch':epoch+1,'resolved_config':self.resolved},output/'latest_checkpoint.pt.tmp')
                (output/'latest_checkpoint.pt.tmp').replace(output/'latest_checkpoint.pt')
            metrics = self._evaluate_stage1_model(model,self._loader('test'),speed,turn,meters)
            metrics['parameter_count'] = sum(p.numel() for p in model.parameters())
            paths = save_stage1_artifacts(output,model,self.resolved,history,metrics)
            atomic_json(output/'training_progress.json',{'status':'completed','step':step,
                'elapsed_seconds':time.monotonic()-started,'epochs':self.budget.stage1_epochs})
            return paths
        except Exception as error:
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(), 'step':step,
                'resolved_config':self.resolved},output/'interrupted_checkpoint.pt')
            atomic_json(output/'training_progress.json',{'status':'failed','step':step,'error':repr(error)})
            raise

    def _stage2_data(self, run_dir):
        paths = super()._stage2_data(run_dir)
        from transformers import AutoTokenizer
        from staged_attribution.token_protocol import special_tokens
        from staged_attribution.train_stage2 import tokenize_training_record
        tokenizer=AutoTokenizer.from_pretrained(str(self.config.paths['model_root']))
        tokenizer.add_special_tokens({'additional_special_tokens':special_tokens(json.loads((self.aligned_root/'metadata.json').read_text())['num_roads'])})
        audit={}
        for split in ('train','test'):
            frame=pd.read_parquet(run_dir/'stage2_data'/f'llm_{split}.parquet')
            lengths=[len(tokenize_training_record(tokenizer,r)['input_ids']) for r in frame.to_dict('records')]
            if max(lengths)>self.plan['data']['max_sequence_length']:
                raise ValueError('oversized prompt: revise common cohort before training LLM')
            audit[split]={'max_length':max(lengths),'population':len(lengths)}
        atomic_json(run_dir/'stage2_data'/'length_audit.json',audit)
        return paths

    def _stage3(self,run_dir):
        output=run_dir/'stage3'; output.mkdir(parents=True,exist_ok=True)
        frame=pd.read_parquet(run_dir/'stage2_data/llm_test.parquet')
        manifest=json.loads((run_dir/'stage2_data/subset_manifest.json').read_text())
        frame=frame[frame.traj_id.isin(manifest['generation']['traj_ids'])]
        metadata=json.loads((self.aligned_root/'metadata.json').read_text())
        result=run_hf_generation(frame,run_dir/'stage2/adapter',output/'generations.jsonl',metadata['num_roads'],
            self.seed,device=str(self.device),batch_size=1,max_new_tokens=256,progress_path=output/'generation_progress.json')
        atomic_json(output/'generation_audit.json',result['audit'])
        return {'generations':output/'generations.jsonl'}

    def _stage4(self,run_dir):
        output=run_dir/'stage4'; output.mkdir(parents=True,exist_ok=True)
        dataset=self._dataset('test')
        variants={k:[json.loads(line) for line in (run_dir/'decode'/f'{k}_reconstructed.jsonl').read_text().splitlines()]
                  for k in ('g00','g10','g01','g11')}
        ids={int(r['traj_id']) for r in variants['g00']}
        if any({int(r['traj_id']) for r in rows}!=ids for rows in variants.values()):
            raise ValueError('G populations differ')
        real_gps={}; real_roads={}
        for item in (dataset[i] for i in range(len(dataset))):
            tid=int(item['traj_id'])
            if tid in ids:
                real_gps[tid]=np.asarray(dataset.records_by_id[tid]['gps_seq'],dtype=np.float64)
                real_roads[tid]=item['point_road'].tolist()
        geometry=self._road_geometries()
        metrics={k:evaluate_population(real_gps,real_roads,rows,geometry,self.city,invalid_penalty=1.,
                    mbr=dataset.metadata['mbr'],grid=dataset.metadata['grid']) for k,rows in variants.items()}
        atomic_json(output/'stage4_metrics.json',metrics)
        return {'metrics':output/'stage4_metrics.json'}


def main():
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
    verify_preflight(args.preflight)
    if any(c not in plan['cells'] for c in args.cells):
        raise ValueError('unknown cell')
    if args.validate_only:
        print(json.dumps({'valid':True,'phase':args.phase,'cells':args.cells})); return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
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
            for stage in ('stage1','tokens','stage2_data','stage2','stage3','g_attribution','decode','stage4'):
                backend.check_budget()
                manifest['current_stage']=stage
                atomic_json(run_dir/'run_manifest.json',manifest)
                print(f'{cell}: {stage}',flush=True)
                paths=getattr(backend,'_'+stage)(run_dir)
                evidence={str(p.relative_to(run_dir)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in paths.values()}
                manifest['stages'][stage]={'status':'completed','artifacts':evidence}
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


if __name__=='__main__':
    main()
