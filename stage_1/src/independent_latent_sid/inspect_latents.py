"""Read-only frozen-tokenizer diagnostics; never updates checkpoint or optimizer."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from independent_latent_sid.backend import Backend
from hierarchical_rqvae.preflight import atomic_json, sha256


def geometry(x):
    x=x.double()
    centered=x-x.mean(0)
    eig=torch.linalg.eigvalsh(centered.T@centered/max(len(x)-1,1)).clamp_min(0)
    p=eig/eig.sum().clamp_min(1e-30)
    rank=float((-(p*p.clamp_min(1e-30).log()).sum()).exp()) if eig.sum()>1e-20 else 0.
    return {'events':len(x),'mean_vector_norm':float(x.mean(0).norm()),
        'mean_event_norm':float(x.norm(dim=-1).mean()),'rms_centered':float(centered.square().mean().sqrt()),
        'mean_dimension_std':float(centered.square().mean(0).sqrt().mean()),'effective_rank':rank}


def summarize(latents, prefixes, layers, codes, sizes):
    result={'latent_geometry':{v:geometry(z) for v,z in latents.items()},'layers':[],'prefix_target_mse':{}}
    for v,z in latents.items():
        result['prefix_target_mse'][v]=[float((prefixes[:,i]-z).square().mean()) for i in range(4)]
    for i,size in enumerate(sizes):
        count=codes[:,i].bincount(minlength=size).double();p=count/count.sum()
        before=torch.zeros_like(layers[:,i]) if i==0 else prefixes[:,i-1]
        q=layers[:,i]
        result['layers'].append({'layer':i+1,'active_codes':int((count>0).sum()),
            'perplexity':float((-(p*p.clamp_min(1e-30).log()).sum()).exp()),
            'max_share':float(p.max()),'dominant_code':int(p.argmax()),
            'mean_code_norm':float(q.norm(dim=-1).mean()),
            'prefix_before_norm':float(before.norm(dim=-1).mean()),
            'cosine_code_previous_prefix':None if i==0 else float(F.cosine_similarity(q,before).mean()),
            'prefix_after_norm':float(prefixes[:,i].norm(dim=-1).mean())})
    result['cross_view_cosine']={a+'_'+b:float(F.cosine_similarity(latents[a],latents[b]).mean())
        for a,b in [('grid','road'),('grid','gps'),('road','gps')]}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cell',choices=['HV','LF'],required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('refusing overwrite')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('dirty analysis workspace')
    torch.set_num_threads(4)
    plan=json.loads(Path(args.config).read_text());run=json.loads((args.run_dir/'run_manifest.json').read_text())
    if run['cell_id']!=args.cell or run['config_sha256']!=sha256(args.config):raise ValueError('run identity mismatch')
    stage=run['stages'].get('stage1',{})
    if stage.get('status')!='completed':raise ValueError('tokenizer not completed')
    for path,meta in stage['artifacts'].items():
        if sha256(args.run_dir/path)!=meta['sha256']:raise ValueError('stage1 artifact changed')
    backend=Backend(args.config,plan['ablation']['preflight_path'],args.cell,'pilot',run['seed'],time.monotonic()+1800)
    model=backend._load_model(args.run_dir).eval().to('cuda')
    result={'cell':args.cell,'status':'completed','command':sys.argv,'analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'training_revision':run['revision'],'config_sha256':run['config_sha256'],'preflight_sha256':run['preflight_sha256'],
        'checkpoint_sha256':sha256(args.run_dir/'stage1/stage1_model.pt'),
        'method':'FP32 eval; full frozen train and validation, equal event weighting; no optimizer or checkpoint writes', 'splits':{}}
    for split,n in [('train',8192),('test',1024)]:
        zs={v:[] for v in ('grid','road','gps')};ps=[];qs=[];cs=[];traj_ids=[]
        with torch.no_grad():
            for batch in backend._loader(split):
                batch={k:v.to('cuda') for k,v in batch.items()};out=model.encode_views(batch)
                for v in zs:zs[v].append(out['view_latents'][v].cpu())
                ps.append(out['prefix_vectors'].cpu());qs.append(out['layer_vectors'].cpu());cs.append(out['codes'][out['latent_mask']].cpu())
                traj_ids.extend(batch['traj_id'].tolist())
        if len(traj_ids)!=n or len(set(traj_ids))!=n:raise ValueError('wrong population')
        data=summarize({v:torch.cat(x) for v,x in zs.items()},torch.cat(ps),torch.cat(qs),torch.cat(cs),plan['model']['vocab_sizes'])
        data['trajectories']=n;result['splits']['validation' if split=='test' else split]=data
    batch=next(iter(backend._loader('train')));batch={k:v.to('cuda') for k,v in batch.items()}
    model.train();torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    out=model(batch)
    terms={'reconstruction':out['reconstruction_loss'],'vq':out['quant_loss'],
        **{f'weighted_aux_{v}':.1*out['view_losses'][v] for v in ('grid','road','gps')}}
    weights=[model.view_projections[v].weight for v in ('grid','road','gps')]
    result['gradient_diagnostic']={'scope':'first 128 train trajectories, FP32 train mode dropout seed42; projection-weight gradient norms, no optimizer', 'terms':{}}
    for name,loss in terms.items():
        grads=torch.autograd.grad(loss,weights,retain_graph=True,allow_unused=True)
        result['gradient_diagnostic']['terms'][name]={'loss':float(loss.detach()),'projection_grad_norm':dict(zip(('grid','road','gps'),[None if g is None else float(g.norm()) for g in grads]))}
    atomic_json(args.output,result);print(json.dumps({'output':str(args.output),'validation':result['splits']['validation'],'gradients':result['gradient_diagnostic']},indent=2))

if __name__=='__main__':main()
