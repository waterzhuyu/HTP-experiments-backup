"""Correct batch-dependent intervention decoding without changing training or main evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from hierarchical_rqvae.analyze import verify_attempt, digest, read
from hierarchical_rqvae.run import Backend
from hierarchical_rqvae.preflight import atomic_json
from hierarchical_rqvae.diagnostics import progress_gps_errors
from staged_attribution.data import make_collator
from staged_attribution.decode_stage3 import reconstruct_gps
from staged_attribution.stage4_metrics import evaluate_population


def prefix_values(model, codes, depth):
    if not 1<=depth<=4: raise ValueError('depth must be 1..4')
    # Same stack and reduction layout as production RQ.encode; zero omitted layers.
    vectors=[book.encode(codes[:,i]) for i,book in enumerate(model.rq_quant.codebooks)]
    return torch.stack([v if i<depth else torch.zeros_like(v) for i,v in enumerate(vectors)],dim=-1).sum(-1)


@torch.no_grad()
def run_cell(backend, source, output):
    if output.exists(): raise ValueError('refusing overwrite')
    manifest,count=verify_attempt(source)
    output.mkdir(parents=True)
    started=time.monotonic()
    model=backend._load_model(source).eval().to(backend.device)
    dataset=backend._dataset('test'); items={int(dataset[i]['traj_id']):dataset[i] for i in range(len(dataset))}
    tokens={int(r['traj_id']):r for r in map(json.loads,(source/'tokens/test_tokens.jsonl').read_text().splitlines())}
    cached=torch.load(source/'diagnostics/frozen_features.pt',map_location='cpu',weights_only=True)['validation']
    for tid,r in tokens.items():
        if not torch.equal(cached['codes'][cached['traj_id']==tid],torch.tensor(r['codes'])):
            raise ValueError('cache/export code mismatch')
    original={int(r['traj_id']):r for r in map(json.loads,(source/'decode/g00_reconstructed.jsonl').read_text().splitlines())}
    if set(tokens)!=set(original) or set(items)!=set(tokens): raise ValueError('population mismatch')
    buckets={}
    for tid,item in items.items(): buckets.setdefault((len(tokens[tid]['codes']),len(item['route'])//5),[]).append(tid)
    donors={}
    for group in buckets.values():
        group=sorted(group,key=lambda t:hashlib.sha256(f'42:{t}'.encode()).hexdigest())
        if len(group)>1: donors.update({t:group[(i+1)%len(group)] for i,t in enumerate(group)})
    collate=make_collator(backend._context()); geometry=backend._road_geometries(); scale=backend._context().z_score
    conditions={f'prefix_{i}':[] for i in range(1,5)}
    conditions.update({f'swap_{i}':[] for i in range(1,5)});conditions['swap_3_4']=[]
    parity={'percent_max_abs':0.,'dxy_max_abs':0.,'gps_degrees_max_abs':0.}
    for index,tid in enumerate(sorted(tokens)):
        backend.check_budget()
        batch={k:v.to(backend.device) for k,v in collate([items[tid]]).items()}
        r=tokens[tid]; codes=torch.tensor(r['codes'],device=backend.device); lengths=torch.tensor([len(codes)],device=backend.device)
        lens=torch.tensor([r['lens_type']],device=backend.device)
        for condition,rows in conditions.items():
            if condition.startswith('prefix_'):
                values=prefix_values(model,codes,int(condition.split('_')[1]))
            else:
                if tid not in donors: continue
                swapped=codes.clone(); other=torch.tensor(tokens[donors[tid]]['codes'],device=backend.device)
                for layer in condition.split('_')[1:]: swapped[:,int(layer)-1]=other[:,int(layer)-1]
                values=model.rq_quant.encode(swapped.T)
            percent,dxy,decoded_lengths=model.decode_quantized(values,lengths,lens,batch)
            n=int(decoded_lengths[0]); row=reconstruct_gps({'traj_id':tid,'status':'decoded','road_seq':r['road_seq'],
                'pred_percent':percent[0,:n].cpu().tolist(),'pred_dxy':dxy[0,:n].cpu().tolist(),'decoded_length':n},geometry,scale)
            rows.append(row)
            if condition=='prefix_4':
                ref=original[tid]
                if row['status']!=ref['status'] or n!=ref['decoded_length']: raise ValueError('prefix4 status/length mismatch')
                for field,key in [('pred_percent','percent_max_abs'),('pred_dxy','dxy_max_abs'),('gps_traj','gps_degrees_max_abs')]:
                    diff=float(np.max(np.abs(np.asarray(row[field])-np.asarray(ref[field]))))
                    parity[key]=max(parity[key],diff)
        if (index+1)%32==0:
            atomic_json(output/'progress.json',{'status':'running','completed':index+1,'population':len(tokens),
                'elapsed_seconds':time.monotonic()-started,'parity':parity})
    # Same-device production path expected to agree to numerical precision.
    # CPU is for debugging only; final corrected report requires CUDA and these tolerances.
    passed=(parity['percent_max_abs']<=1e-5 and parity['dxy_max_abs']<=1e-5 and parity['gps_degrees_max_abs']<=1e-7)
    result={'swap_eligible':len(donors),'total_validation':len(items),'donors':donors,'conditions':{},
        'truncation_interpretation':'OOD zero residual diagnostic; production singleton decoding',
        'swap_interpretation':'same event-count/route-length bucket; cached exported codes; association only'}
    for name,rows in conditions.items():
        ids={int(r['traj_id']) for r in rows}; real={t:np.asarray(dataset.records_by_id[t]['gps_seq']) for t in ids}
        roads={t:items[t]['point_road'].tolist() for t in ids}
        if not rows: result['conditions'][name]={'status':'not_applicable'};continue
        result['conditions'][name]={'metrics':evaluate_population(real,roads,rows,geometry,'porto',invalid_penalty=1.,
            mbr=dataset.metadata['mbr'],grid=dataset.metadata['grid']),
            'gps_successful_only':[{'traj_id':r['traj_id'],**progress_gps_errors(real[r['traj_id']],r['gps_traj'])}
                for r in rows if r['status']=='reconstructed']}
        atomic_json(output/f'{name}_reconstructed.json',rows)
    eligible=[r for r in conditions['prefix_4'] if r['traj_id'] in donors]
    if eligible:
        real={t:np.asarray(dataset.records_by_id[t]['gps_seq']) for t in donors};roads={t:items[t]['point_road'].tolist() for t in donors}
        result['swap_native_reference']=evaluate_population(real,roads,eligible,geometry,'porto',invalid_penalty=1.,mbr=dataset.metadata['mbr'],grid=dataset.metadata['grid'])
    atomic_json(output/'metrics.json',result)
    audit={'status':'completed' if passed and backend.device.type=='cuda' else 'parity_failed_or_cpu_debug_only',
        'analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_manifest_sha256':digest(source/'run_manifest.json'),'training_revision':manifest['revision'],
        'source_artifacts_verified':count,'device':str(backend.device),'parity':parity,'elapsed_seconds':time.monotonic()-started,
        'supersedes':str(source/'diagnostics/interventions'),'artifacts':{str(f.relative_to(output)):digest(f) for f in output.glob('*.json')}}
    atomic_json(output/'audit_manifest.json',audit)
    print(json.dumps(audit,indent=2),flush=True)
    if audit['status']!='completed': raise ValueError('corrected intervention parity not certified')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cells',nargs='+',default=['B2','H1','B0','B1','H2','H3']);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for cell in args.cells:
        b=Backend(args.config,args.root/'preflight',cell,'pilot',42,time.monotonic()+3600)
        run_cell(b,args.root/'pilot/porto/42'/cell/'attempt_001',args.output/cell)

if __name__=='__main__':main()
