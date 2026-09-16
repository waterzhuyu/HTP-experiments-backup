"""Measure weighted objective gradient conflicts without updating model parameters."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from hierarchical_rqvae.preflight import atomic_json,sha256
from independent_latent_sid.backend import Backend


def cosine(a,b):
    if a is None or b is None or a.norm()==0 or b.norm()==0:return None
    return float(torch.dot(a.flatten(),b.flatten())/(a.norm()*b.norm()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('refusing overwrite')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('dirty analysis workspace')
    torch.set_num_threads(4);plan=json.loads(Path(a.config).read_text());m=json.loads((a.run_dir/'run_manifest.json').read_text())
    if m['cell_id'] not in ('HV','LF') or m['config_sha256']!=sha256(a.config):raise ValueError('source identity mismatch')
    if m['stages']['stage1']['status']!='completed':raise ValueError('tokenizer unfinished')
    for file,meta in m['stages']['stage1']['artifacts'].items():
        if sha256(a.run_dir/file)!=meta['sha256']:raise ValueError('stage1 artifact changed')
    backend=Backend(a.config,plan['ablation']['preflight_path'],m['cell_id'],'pilot',m['seed'],time.monotonic()+600)
    model=backend._load_model(a.run_dir).to('cuda').train();records=[]
    for index,batch in enumerate(itertools.islice(backend._loader('train'),8)):
        torch.manual_seed(42+index);torch.cuda.manual_seed_all(42+index)
        batch={k:v.to('cuda') for k,v in batch.items()};out=model(batch)
        terms={'rec':out['reconstruction_loss'],'vq':out['quant_loss'],**{v:.1*out['view_losses'][v] for v in ('grid','road','gps')}}
        parameters=[model.view_projections[v].weight for v in ('grid','road','gps')]
        grads={k:torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) for k,loss in terms.items()}
        row={'batch_index':index,'trajectory_count':len(batch['traj_id']),'dropout_seed':42+index,'views':{}}
        for vi,view in enumerate(('grid','road','gps')):
            row['views'][view]={a+'__'+b:cosine(grads[a][vi],grads[b][vi]) for a,b in itertools.combinations(terms,2)}
        records.append(row)
    summary={}
    for view in ('grid','road','gps'):
        summary[view]={}
        for pair in records[0]['views'][view]:
            vals=[r['views'][view][pair] for r in records if r['views'][view][pair] is not None]
            summary[view][pair]={'n':len(vals),'mean':sum(vals)/len(vals) if vals else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None}
    result={'status':'completed','cell':m['cell_id'],'analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'training_revision':m['revision'],'checkpoint_sha256':sha256(a.run_dir/'stage1/stage1_model.pt'),'config_sha256':m['config_sha256'],'preflight_sha256':m['preflight_sha256'],
        'command':sys.argv,'method':'First eight sequential train batches of 128 (1024 unique trajectories); FP32 train mode; dropout seeds42..49; actual weighted loss gradients of each view projection weight; no optimizer or checkpoint update; cosine=None for absent/zero gradient.',
        'limitations':'Directional association at final checkpoint, not training history or causal ablation; only projection weights, no inference over all parameters.',
        'summary':summary,'batches':records}
    atomic_json(a.output,result);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
