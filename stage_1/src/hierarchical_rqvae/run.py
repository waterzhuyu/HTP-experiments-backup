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
from staged_attribution.train_stage2 import run_hf_stage2


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
        self._last_size_check = 0.0
        self.artifact_scope = self.config.paths["artifact_root"] / self.plan["experiment_id"] / phase
        model_manifest=Path(preflight_path)/'model_manifest.json'
        self.resolved = {'base_model_manifest_sha256':sha256(model_manifest) if model_manifest.exists() else None, 'experiment_id':self.plan['experiment_id'], 'cell_id':cell_id, 'city':self.city,
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

    def _loader(self, split, shuffle=False):
        from torch.nn.utils.rnn import pad_sequence
        dataset=self._dataset(split)
        base_collate=make_collator(self._context())
        def precise_collate(items):
            batch=base_collate(items)
            batch['time']=pad_sequence([torch.as_tensor(
                np.asarray(dataset.records_by_id[int(item['traj_id'])]['time_seq'],dtype=np.float64))
                for item in items],batch_first=True,padding_value=0.)
            return batch
        return DataLoader(dataset,batch_size=self.budget.stage1_batch_size,shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.seed),num_workers=0,collate_fn=precise_collate)

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
        if time.monotonic() - self._last_size_check > 30:
            self._last_size_check = time.monotonic()
            size = sum(p.stat().st_size for p in self.artifact_scope.rglob('*') if p.is_file())
            if size >= self.plan['budgets']['pilot']['artifact_gib_cap'] * 1024**3:
                raise BudgetExhausted('artifact byte cap reached')

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

    def _stage2(self, run_dir):
        from transformers import TrainerCallback
        backend = self
        class BudgetCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                try:
                    backend.check_budget()
                except BudgetExhausted:
                    # Save only trainable state plus optimizer, never another base-model copy.
                    model = kwargs['model']
                    torch.save({'trainable_parameters':{k:p.detach().cpu() for k,p in model.named_parameters() if p.requires_grad},
                        'optimizer':kwargs['optimizer'].state_dict(), 'step':state.global_step,
                        'resolved_config':backend.resolved}, run_dir/'stage2/interrupted_trainable_state.pt')
                    raise
                return control
        output = run_dir/'stage2'
        frame = pd.read_parquet(run_dir/'stage2_data/llm_train.parquet')
        eval_frame = pd.read_parquet(run_dir/'stage2_data/llm_test.parquet')
        metadata = json.loads((self.aligned_root/'metadata.json').read_text())
        result = run_hf_stage2(frame, self.config.paths['model_root'], output, metadata['num_roads'],
            self.seed, self.budget.stage2_epochs, self.budget.stage2_max_steps,
            batch_size=self.budget.stage2_batch_size, gradient_accumulation=self.budget.stage2_gradient_accumulation,
            eval_frame=eval_frame, runtime_guard=self.check_budget, extra_callbacks=[BudgetCallback()])
        atomic_json(output/'adapter_manifest.json', {'adapter':result['adapter'],
            'base_model':str(self.config.paths['model_root']), 'cell_id':self.cell_id,
            'tokenizer_sha256':sha256(run_dir/'stage1/stage1_model.pt'),
            'adapter_storage':result['adapter_storage'], 'seed':self.seed})
        return {str(p.relative_to(output)):p for p in output.rglob('*') if p.is_file()}

    def _stage3(self,run_dir):
        output=run_dir/'stage3'; output.mkdir(parents=True,exist_ok=True)
        frame=pd.read_parquet(run_dir/'stage2_data/llm_test.parquet')
        manifest=json.loads((run_dir/'stage2_data/subset_manifest.json').read_text())
        frame=frame[frame.traj_id.isin(manifest['generation']['traj_ids'])]
        metadata=json.loads((self.aligned_root/'metadata.json').read_text())
        result=run_hf_generation(frame,run_dir/'stage2/adapter',output/'generations.jsonl',metadata['num_roads'],
            self.seed,device=str(self.device),batch_size=1,max_new_tokens=256,progress_path=output/'generation_progress.json',
            progress_interval=1, runtime_guard=self.check_budget)
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


    def _diagnostics(self,run_dir):
        from hierarchical_rqvae.diagnostics import collect_codes, frozen_probes, decode_level_interventions, progress_gps_errors
        output=run_dir/'diagnostics';output.mkdir(parents=True,exist_ok=True)
        model=self._load_model(run_dir)
        train,_=collect_codes(model,self._loader('train'),self.device,self.check_budget)
        val,sampling=collect_codes(model,self._loader('test'),self.device,self.check_budget,sampling=True)
        torch.save({'train':train,'validation':val},output/'frozen_features.pt')
        atomic_json(output/'sampling.json',sampling)
        metadata=json.loads((self.aligned_root/'metadata.json').read_text())
        probes=frozen_probes(train,val,metadata['num_grids'],metadata['num_roads'],output/'probes',
            self.device,self.check_budget,epochs=1 if self.phase=='smoke' else 10)
        decode_level_interventions(self,model,val,output/'interventions')
        dataset=self._dataset('test')
        errors={}
        for variant in ('g00','g10','g01','g11'):
            rows=[json.loads(line) for line in (run_dir/'decode'/f'{variant}_reconstructed.jsonl').read_text().splitlines()]
            errors[variant]=[{'traj_id':r['traj_id'],**progress_gps_errors(dataset.records_by_id[int(r['traj_id'])]['gps_seq'],r['gps_traj'])}
                if r['status']=='reconstructed' else {'traj_id':r['traj_id'],'status':'failed','gps_errors':None} for r in rows]
        atomic_json(output/'gps_errors.json',errors)
        return {str(p.relative_to(output)):p for p in output.rglob('*') if p.is_file()}


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
    preflight_manifest=verify_preflight(args.preflight)
    if args.phase=='pilot':
        audit_path=Path(args.preflight)/'alignment_audit.json'
        if not json.loads(audit_path.read_text())['passed']:
            raise ValueError('pointwise data audit has not passed')
        model_manifest=json.loads((Path(args.preflight)/'model_manifest.json').read_text())
        for path,digest in model_manifest['hashes'].items():
            if sha256(path)!=digest: raise ValueError('base model artifact changed')
        for path,digest in preflight_manifest['source_hashes'].items():
            if sha256(path)!=digest: raise ValueError('source data artifact changed')
        original=Path(args.config).with_name('plan.json')
        if sha256(original)!=preflight_manifest['config_sha256']:
            raise ValueError('original scientific plan changed after preflight')
        frozen=json.loads(original.read_text())
        for key in ('data','model','cells','tokenizer_training','llm_training','generation','gates','evaluation'):
            if plan[key]!=frozen[key]: raise ValueError(f'scientific configuration drift: {key}')
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


if __name__=='__main__':
    main()
