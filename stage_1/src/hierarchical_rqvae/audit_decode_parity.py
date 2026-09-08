"""Actual-checkpoint CPU audit of cached/exported codes and batch decoder parity."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from hierarchical_rqvae.run import Backend
from hierarchical_rqvae.analyze import digest
from staged_attribution.data import make_collator


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cell',default='B2');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists(): raise ValueError('refusing overwrite')
    torch.set_num_threads(2)
    b=Backend(args.config,args.root/'preflight',args.cell,'pilot',42,float('inf'));b.device=torch.device('cpu')
    run=args.root/'pilot/porto/42'/args.cell/'attempt_001'
    cache=torch.load(run/'diagnostics/frozen_features.pt',map_location='cpu',weights_only=True)['validation']
    tokens={int(r['traj_id']):r for r in map(json.loads,(run/'tokens/test_tokens.jsonl').read_text().splitlines())}
    mismatches=[]
    for tid,r in tokens.items():
        if not torch.equal(cache['codes'][cache['traj_id']==tid],torch.tensor(r['codes'])): mismatches.append(tid)
    model=b._load_model(run).eval(); batch=next(iter(b._loader('test')))
    source={int(r['traj_id']):r for r in map(json.loads,(run/'decode/g00_reconstructed.jsonl').read_text().splitlines())}
    dataset=b._dataset('test');items={int(dataset[i]['traj_id']):dataset[i] for i in range(len(dataset))}
    collate=make_collator(b._context()); results=[]
    with torch.no_grad():
        out=model.encode_views(batch)
        bp,bd,bl=model.decode_quantized(out['prefix_vectors'][:,3],out['latent_lengths'],out['lens_type'],batch)
        for j,tid in enumerate(batch['traj_id'].tolist()[:8]):
            r=tokens[tid]; c=torch.tensor(r['codes']); lengths=torch.tensor([len(c)]); lens=torch.tensor([r['lens_type']])
            single=collate([items[tid]])
            sp,sd,sl=model.decode(c,lengths,lens,single)
            qp,qd,ql=model.decode_quantized(model.rq_quant.encode(c.T),lengths,lens,single)
            n=int(sl[0]); orig=source[tid]
            results.append({'traj_id':tid,'length':n,'cpu_single_vs_formal_gpu_percent_max':float(np.max(np.abs(sp[0,:n].numpy()-np.asarray(orig['pred_percent'])))),
                'cpu_single_vs_formal_gpu_dxy_max':float(np.max(np.abs(sd[0,:n].numpy()-np.asarray(orig['pred_dxy'])))),
                'same_single_prefix4_vs_decode_percent_max':float((qp-sp).abs().max()),
                'same_single_prefix4_vs_decode_dxy_max':float((qd-sd).abs().max()),
                'batch128_vs_single_percent_max':float((bp[j,:n]-sp[0,:n]).abs().max()),
                'batch128_vs_single_dxy_max':float((bd[j,:n]-sd[0,:n]).abs().max())})
    result={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cell':args.cell,
        'source_manifest_sha256':digest(run/'run_manifest.json'),'population':len(tokens),'cache_token_mismatch_count':len(mismatches),
        'mismatched_ids':mismatches,'samples':results}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
