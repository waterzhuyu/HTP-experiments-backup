"""Post-hoc information diagnostics from frozen train/validation SID assignments."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from hierarchical_rqvae.analyze import read, verify_attempt, digest


def count_summary(codes, sizes):
    codes=np.asarray(codes,dtype=np.int64)
    if codes.ndim!=2 or codes.shape[1]!=len(sizes): raise ValueError('invalid code matrix')
    n=len(codes)
    if not n: raise ValueError('empty population')
    layers=[]
    for i,k in enumerate(sizes):
        if np.any(codes[:,i]<0) or np.any(codes[:,i]>=k): raise ValueError('out of vocabulary')
        counts=np.bincount(codes[:,i],minlength=k); p=counts[counts>0]/n
        layers.append({'layer':i+1,'entropy_nats':float(-(p*np.log(p)).sum()),
            'active_codes':int((counts>0).sum()),'counts':counts.tolist()})
    _,counts=np.unique(codes,axis=0,return_counts=True); p=counts/n
    return {'events':n,'layers':layers,'unique_tuples':len(counts),
        'duplicate_event_fraction':float(1-len(counts)/n),
        'pair_collision_probability':float(np.sum(counts*(counts-1))/(n*(n-1))) if n>1 else None,
        'tuple_entropy_nats':float(-(p*np.log(p)).sum())}


def unigram_cross_entropy(train_codes,val_codes,sizes,alpha=.5):
    result=[]
    for i,k in enumerate(sizes):
        counts=np.bincount(train_codes[:,i],minlength=k)
        probability=(counts+alpha)/(len(train_codes)+alpha*k)
        result.append(float(-np.log(probability[val_codes[:,i]]).mean()))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cells',nargs='+',default=['B0','B1','B2','H1','H2','H3']);a=p.parse_args()
    if a.output.exists(): raise ValueError('refusing overwrite')
    output={}
    for cell in a.cells:
        run=a.root/'pilot/porto/42'/cell/'attempt_001';manifest,_=verify_attempt(run)
        cached=torch.load(run/'diagnostics/frozen_features.pt',map_location='cpu',weights_only=True)
        arrays={s:cached[s]['codes'].numpy() for s in ['train','validation']}
        sizes=manifest['plan']['model']['vocab_sizes']
        ce=unigram_cross_entropy(arrays['train'],arrays['validation'],sizes)
        llm=read(run/'stage2/stage2_metrics.json')['teacher_forced']
        output[cell]={s:count_summary(c,sizes) for s,c in arrays.items()}
        output[cell]['train_unigram_validation_cross_entropy_nats']=ce
        output[cell]['llm_sid_nll']=llm['sid_nll']
        output[cell]['unigram_mean_nll']=float(np.mean(ce))
        output[cell]['llm_gain_over_train_unigram_nats']=float(np.mean(ce)-llm['sid_nll'])
        output[cell]['llm_nll_fraction_of_unigram']=float(llm['sid_nll']/np.mean(ce))
        output[cell]['source_manifest_sha256']=digest(run/'run_manifest.json')
        output[cell]['training_revision']=manifest['revision']
    report={'status':'complete' if len(a.cells)==6 else 'provisional','analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'method':'post-hoc frozen SID count diagnostics; unigram fit exclusively to train, alpha=.5 per code',
        'units':'nats per event for tuple entropy; nats per layer token for SID/unigram NLL',
        'limitations':['duplicate_event_fraction counts repeated tuples including same trajectory',
            'pair collision is probability of identical tuples for two distinct uniformly chosen events',
            'unigram predictor ignores route, prior events and prior layers; comparison descriptive, not a matched LLM baseline',
            'lower code entropy alone does not establish why LLM NLL changed'],'cells':output}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({c:{k:v for k,v in x.items() if k not in ['train','validation']} for c,x in output.items()},indent=2))

if __name__=='__main__':main()
