"""Supplementary teacher-forced and ordinal free-running tuple accuracy; no training."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import numpy as np


def correctness_summary(correct):
    correct=np.asarray(correct,dtype=bool)
    if correct.ndim!=2 or correct.shape[1]!=4 or not len(correct): raise ValueError('expected nonempty events x 4')
    first=np.where(correct.all(1),4,correct.argmin(1))
    patterns=(correct*(1 << np.arange(4))).sum(1)
    conditional={}
    for layer in range(1,4):
        earlier_ok=correct[:,:layer].all(1)
        conditional[str(layer+1)]={}
        for name,mask in [('earlier_all_correct',earlier_ok),('earlier_any_wrong',~earlier_ok)]:
            conditional[str(layer+1)][name]={'events':int(mask.sum()),
                'correct':int(correct[mask,layer].sum()),
                'accuracy':float(correct[mask,layer].mean()) if mask.any() else None}
    return {'events':len(correct),'tuple_correct':int(correct.all(1).sum()),'tuple_accuracy':float(correct.all(1).mean()),
        'layer_correct':correct.sum(0).tolist(),'layer_accuracy':correct.mean(0).tolist(),
        'first_wrong_layer_counts':{str(i+1) if i<4 else 'all_correct':int((first==i).sum()) for i in range(5)},
        'correctness_pattern_counts':dict(Counter(map(str,patterns.tolist()))),'later_layer_conditional_accuracy':conditional}


def extract_correctness(predictions,labels,groups):
    reverse={value:layer for layer,values in enumerate(groups) for value in values}
    selected=[j for j,v in enumerate(labels) if int(v) in reverse]
    if not selected or len(selected)%4: raise ValueError('incomplete target SID tuples')
    order=[reverse[int(labels[j])] for j in selected]
    if order!=list(range(4))*(len(order)//4): raise ValueError('target SID order mismatch')
    return np.array([int(predictions[j])==int(labels[j]) for j in selected]).reshape(-1,4)


def build_groups(tokenizer,sizes):
    if len(sizes)!=4 or any(type(k) is not int or k<2 for k in sizes):raise ValueError('invalid capacities')
    groups=[set(tokenizer.convert_tokens_to_ids([f'<{prefix}_{j}>' for j in range(k)]))
            for prefix,k in zip(('a','b','c','d'),sizes)]
    if any(len(g)!=k or tokenizer.unk_token_id in g or None in g for g,k in zip(groups,sizes)):
        raise ValueError('missing or aliased SID token')
    if len(set.union(*groups))!=sum(sizes):raise ValueError('cross-layer token collision')
    return groups


def main():
    import torch
    import pandas as pd
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer,set_seed
    from hierarchical_rqvae.analyze import read,verify_attempt,digest
    from hierarchical_rqvae.preflight import atomic_json
    from staged_attribution.adapter_storage import restore_and_validate_loaded_compact_adapter
    from staged_attribution.train_stage2 import tokenize_training_record,AssistantOnlyCollator,_TokenizedDataset
    import staged_attribution.train_stage2 as stage2
    from staged_attribution.token_protocol import PREFIXES
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.output.exists(): raise ValueError('refusing overwrite')
    a.output.mkdir(parents=True); torch.set_num_threads(2)
    for run in [a.run]:
        started=time.monotonic();manifest,count=verify_attempt(run)
        if manifest['budget']!='pilot':raise ValueError('requires full pilot validation')
        cell=manifest['cell_id'];sizes=manifest['rq_vocab_sizes']
        protocol=read(run/'stage2/adapter/rq_protocol.json')
        if protocol!={'vocab_sizes':sizes,'config_sha256':manifest['config_sha256']}:raise ValueError('adapter protocol mismatch')
        output=a.output/cell;output.mkdir();adapter=run/'stage2/adapter';set_seed(manifest['seed'])
        tokenizer=AutoTokenizer.from_pretrained(str(adapter));tokenizer.pad_token=tokenizer.eos_token
        model=AutoPeftModelForCausalLM.from_pretrained(str(adapter),torch_dtype=torch.bfloat16).to(a.device).eval()
        model.config.use_cache=False
        restore_and_validate_loaded_compact_adapter(model,adapter)
        frame=pd.read_parquet(run/'stage2_data/llm_test.parquet').sort_values('traj_id')
        tokens={int(r['traj_id']):r for r in map(json.loads,(run/'tokens/test_tokens.jsonl').read_text().splitlines())}
        generated={int(r['traj_id']):r for r in map(json.loads,(run/'decode/g11_reconstructed.jsonl').read_text().splitlines())}
        ids=list(map(int,frame['traj_id']))
        if len(ids)!=1024 or len(set(ids))!=1024 or set(ids)!=set(tokens) or set(ids)!=set(generated): raise ValueError('population mismatch')
        groups=build_groups(tokenizer,sizes)
        allgroups={f'layer_{i}':v for i,v in enumerate(groups)}
        allgroups['sampling']=set(tokenizer.convert_tokens_to_ids([f'<t_{j}>' for j in range(8)]))
        tokenized=[tokenize_training_record(tokenizer,r) for r in frame.to_dict('records')]
        correctness=[];original_grouped=stage2.grouped_teacher_forced_accuracy
        def collect(pred,label,token_groups):
            for pr,gt in zip(pred.tolist(),label.tolist()): correctness.append(extract_correctness(pr,gt,groups))
            return original_grouped(pred,label,token_groups)
        # Local process-only instrumentation; restore callable even if evaluation raises.
        stage2.grouped_teacher_forced_accuracy=collect
        try:
            with torch.no_grad():
                metrics=stage2.evaluate_teacher_forced_model(model,_TokenizedDataset(tokenized),AssistantOnlyCollator(tokenizer),
                    allgroups,torch.device(a.device),batch_size=4,progress_path=output/'progress.json',progress_interval=8)
        finally: stage2.grouped_teacher_forced_accuracy=original_grouped
        original=read(run/'stage2/stage2_metrics.json')['teacher_forced']
        per_traj=[];free=[];extra_events=0;failed=0
        for tid,correct in zip(ids,correctness):
            truth=np.asarray(tokens[tid]['codes']);record=generated[tid]
            if len(correct)!=len(truth): raise ValueError('tuple denominator mismatch')
            predicted=np.asarray(record.get('codes') or [],dtype=int)
            valid=record['status']=='reconstructed' and predicted.ndim==2 and predicted.shape[1]==4
            fc=np.zeros_like(correct)
            if valid:
                n=min(len(predicted),len(truth));fc[:n]=predicted[:n]==truth[:n];extra_events+=max(0,len(predicted)-len(truth))
            else: failed+=1
            free.append(fc);per_traj.append({'traj_id':tid,'teacher_forced':correctness_summary(correct),'free_running':correctness_summary(fc)})
        if len(correctness)!=len(ids): raise ValueError('evaluation lost trajectories')
        delta=abs(metrics['sid_nll']-original['sid_nll'])
        result={'status':'completed' if delta<=.005 and metrics['denominators']==original['denominators'] else 'evaluation_parity_failed',
            'analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_manifest_sha256':digest(run/'run_manifest.json'),'training_revision':manifest['revision'],
            'source_artifact_count':count,'vocab_sizes':sizes,'seed':manifest['seed'],'population':len(ids),'elapsed_seconds':time.monotonic()-started,
            'teacher_forced':correctness_summary(np.concatenate(correctness)),
            'free_running':correctness_summary(np.concatenate(free)),'free_running_failed_trajectories':failed,
            'free_running_extra_events':extra_events,'reevaluated_metrics':metrics,'original_sid_nll':original['sid_nll'],
            'sid_nll_absolute_difference':delta,'per_trajectory':per_traj,
            'interpretation':['teacher-forced later-layer conditional accuracy is error association, not causal propagation',
                'free-running uses ordinal target event alignment; missing events and failed trajectories count wrong',
                'free-running extra events reported separately; tuple accuracy denominator is all target events',
                'same 1024 validation trajectories; frozen saved adapter; no retraining; supplementary diagnostic']}
        atomic_json(output/'metrics.json',result)
        if result['status']!='completed': raise ValueError('restored-model evaluation parity failed')
        del model;torch.cuda.empty_cache()
        print(cell,result['teacher_forced']['tuple_accuracy'],result['free_running']['tuple_accuracy'],flush=True)

if __name__=='__main__':main()
