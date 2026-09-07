"""Frozen-code view probes, sampling diagnostics, and aligned GPS errors."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import adjusted_mutual_info_score

from hierarchical_rqvae.preflight import atomic_json


def restored_view(batch, stride):
    if stride not in (2, 3):
        raise ValueError('diagnostic stride must be 2 or 3')
    result={k:v.clone() for k,v in batch.items()}
    for b,length in enumerate(batch['lengths'].tolist()):
        times=batch['time'][b,:length].double().cpu().numpy()
        if not np.all(np.diff(times)>0):
            raise ValueError('sampling diagnostics require strictly increasing precise timestamps')
        keep=np.unique(np.r_[np.arange(0,length,stride),length-1])
        selected=times[keep]
        for d in range(2):
            original=batch['gps'][b,:length,d].cpu().numpy()
            result['gps'][b,:length,d]=torch.as_tensor(np.interp(times,selected,original[keep]),device=batch['gps'].device)
        # argmin resolves exact distance ties toward the earlier observation.
        nearest=keep[np.abs(times[:,None]-selected[None,:]).argmin(axis=1)]
        index=torch.as_tensor(nearest,device=batch['gps'].device)
        for name in ('grid','point_road'):
            result[name][b,:length]=batch[name][b,index]
    return result


@torch.no_grad()
def collect_codes(model, loader, device, guard=None, sampling=False):
    model.eval().to(device)
    prefixes=[]; layers=[]; codes=[]; target_rows={v:[] for v in ('gps','grid','road')}
    masks=[]; point_weights=[]; ids=[]; perturbations={2:[],3:[]}
    for batch in loader:
        if guard: guard()
        batch={k:v.to(device) for k,v in batch.items()}
        output=model.encode_views(batch)
        prefixes.append(output['prefix_vectors'].cpu())
        layers.append(output['layer_vectors'].cpu())
        codes.append(output['codes'][output['latent_mask']].cpu())
        for b,(length,events) in enumerate(zip(batch['lengths'].tolist(),output['latent_lengths'].tolist())):
            mask=torch.arange(events*8).reshape(events,8)<length
            masks.append(mask)
            point_weights.append(mask.float()/length)
            for name,key in [('gps','gps'),('grid','grid'),('road','point_road')]:
                value=batch[key][b,:length].cpu()
                shape=(events*8,2) if name=='gps' else (events*8,)
                padded=torch.zeros(shape,dtype=value.dtype)
                padded[:length]=value
                target_rows[name].append(padded.reshape(events,8,2) if name=='gps' else padded.reshape(events,8))
            ids.extend([int(batch['traj_id'][b])]*events)
        if sampling:
            for stride in perturbations:
                view=restored_view(batch,stride)
                perturbed=model.encode_views(view)
                if not torch.equal(output['latent_mask'],perturbed['latent_mask']):
                    raise ValueError('restored sampling view changed support')
                perturbations[stride].append(perturbed['codes'][perturbed['latent_mask']].cpu())
    result={'prefixes':torch.cat(prefixes),'layers':torch.cat(layers),'codes':torch.cat(codes),
        'targets':{k:torch.cat(v) for k,v in target_rows.items()},'mask':torch.cat(masks),
        'weights':torch.cat(point_weights),'traj_id':torch.tensor(ids)}
    diagnostic={}
    for stride,rows in perturbations.items():
        if rows:
            a=result['codes']; b=torch.cat(rows)
            diagnostic[str(stride)]={'events':len(a),'layer_agreement':(a==b).float().mean(0).tolist(),
                'tuple_agreement':float((a==b).all(-1).float().mean()),
                'layer_ami':[float(adjusted_mutual_info_score(a[:,i].numpy(),b[:,i].numpy())) for i in range(4)]}
    return result,diagnostic


def probe_view(train, val, representation, depth, view, classes, output, device,
               guard=None, epochs=10, batch_events=256, seed=42):
    # Every comparable view head starts with identical weights and event order.
    torch.manual_seed(seed)
    dimension=train[representation].shape[-1]
    head=nn.Sequential(nn.Linear(dimension+8,64),nn.SiLU(),nn.Linear(64,classes)).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(seed)
    onehot=torch.eye(8,device=device)
    feature_train=train[representation][:,depth].to(device)
    labels_train=train['targets'][view].to(device)
    mask_train=train['mask'].to(device); weights_train=train['weights'].to(device)
    start=time.monotonic(); steps=0
    for epoch in range(epochs):
        order=torch.randperm(len(feature_train),generator=generator)
        head.train()
        for begin in range(0,len(order),batch_events):
            if guard: guard()
            index=order[begin:begin+batch_events].to(device)
            x=feature_train[index,None].expand(-1,8,-1)
            x=torch.cat([x,onehot[None].expand(len(index),-1,-1)],-1)
            mask=mask_train[index]
            logits=head(x[mask].float())
            y=labels_train[index][mask]
            weight=weights_train[index][mask]
            error=(logits-y.float()).square().mean(-1) if view=='gps' else F.cross_entropy(logits,y.long(),reduction='none')
            loss=(error*weight).sum()/weight.sum()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite frozen probe loss')
            optimizer.zero_grad();loss.backward();optimizer.step();steps+=1
        atomic_json(output.with_suffix('.progress.json'),{'epoch':epoch+1,'epochs':epochs,'steps':steps,'status':'running'})
    sums={'loss':0.,'correct':0.,'weight':0.}
    head.eval()
    with torch.no_grad():
        for start_index in range(0,len(val[representation]),batch_events):
            if guard: guard()
            sl=slice(start_index,start_index+batch_events)
            feature=val[representation][sl,depth].to(device)
            x=torch.cat([feature[:,None].expand(-1,8,-1),onehot[None].expand(len(feature),-1,-1)],-1)
            mask=val['mask'][sl].to(device)
            y=val['targets'][view][sl].to(device)[mask]
            weight=val['weights'][sl].to(device)[mask]
            prediction=head(x[mask].float())
            error=(prediction-y.float()).square().mean(-1) if view=='gps' else F.cross_entropy(prediction,y.long(),reduction='none')
            sums['loss']+=float((error*weight).sum());sums['weight']+=float(weight.sum())
            if view!='gps': sums['correct']+=float(((prediction.argmax(-1)==y)*weight).sum())
    result={'representation':representation,'depth':depth+1,'view':view,'loss':sums['loss']/sums['weight'],
        'accuracy':None if view=='gps' else sums['correct']/sums['weight'],
        'epochs':epochs,'batch_events':batch_events,'seed':seed,'steps':steps,'elapsed_seconds':time.monotonic()-start,
        'optimizer':{'name':'AdamW','lr':1e-3,'weight_decay':1e-4},
        'loss_weighting':'valid points weighted by inverse trajectory length; each batch normalized by weight sum'}
    torch.save({'model':head.cpu().state_dict(),'result':result},output)
    atomic_json(output.with_suffix('.progress.json'),{'status':'completed',**result})
    return result


def frozen_probes(train,val,num_grids,num_roads,root,device,guard=None,epochs=10):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    results=[]
    for representation in ('prefixes','layers'):
        for depth in range(4):
            for view,classes in [('gps',2),('grid',num_grids),('road',num_roads)]:
                result=probe_view(train,val,representation,depth,view,classes,
                    root/f'{representation}_{depth+1}_{view}.pt',device,guard=guard,epochs=epochs)
                results.append(result)
                atomic_json(root/'metrics.json',results)
    return results


def progress_gps_errors(real, generated, anchors=100):
    real=np.asarray(real,dtype=np.float64);generated=np.asarray(generated,dtype=np.float64)
    if real.ndim!=2 or generated.ndim!=2 or real.shape[1]!=2 or generated.shape[1]!=2 or min(len(real),len(generated))<1:
        raise ValueError('expected nonempty longitude/latitude trajectories')
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError('nonfinite GPS')
    latitude=np.deg2rad(real[:,1].mean())
    scale=np.asarray([111320*np.cos(latitude),110540.])
    target=np.linspace(0,1,anchors)
    a=np.stack([np.interp(target,np.linspace(0,1,len(real)),real[:,d]) for d in range(2)],-1)
    b=np.stack([np.interp(target,np.linspace(0,1,len(generated)),generated[:,d]) for d in range(2)],-1)
    distances=np.linalg.norm((a-b)*scale,axis=1)
    pair=np.linalg.norm((a[:,None]-b[None])*scale,axis=-1)
    dp=np.full((anchors+1,anchors+1),np.inf);dp[0,0]=0
    for i in range(anchors):
        for j in range(anchors):
            dp[i+1,j+1]=pair[i,j]+min(dp[i,j],dp[i+1,j],dp[i,j+1])
    return {'matched_progress_ade_m':float(distances.mean()),'fde_m':float(distances[-1]),
            'dtw_sum_m':float(dp[-1,-1]),'length_absolute_error':abs(len(real)-len(generated)),
            'anchors':anchors,'time_semantics':'uniform observation progress, not clock time'}


@torch.no_grad()
def decode_level_interventions(backend, model, cache, root):
    """Prefix truncation and matched-support donor swaps, using the same oracle route."""
    import hashlib
    from staged_attribution.decode_stage3 import reconstruct_gps
    from staged_attribution.stage4_metrics import evaluate_population
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    dataset=backend._dataset('test')
    items={int(dataset[i]['traj_id']):dataset[i] for i in range(len(dataset))}
    codes={tid:cache['codes'][cache['traj_id']==tid] for tid in items}
    buckets={}
    for tid,item in items.items():
        buckets.setdefault((len(codes[tid]),len(item['route'])//5),[]).append(tid)
    donor={}
    for group in buckets.values():
        ordered=sorted(group,key=lambda tid:hashlib.sha256(f'42:{tid}'.encode()).hexdigest())
        if len(ordered)>1:
            donor.update({tid:ordered[(i+1)%len(ordered)] for i,tid in enumerate(ordered)})
    conditions={f'prefix_{i}':[] for i in range(1,5)}
    conditions.update({f'swap_{i}':[] for i in range(1,5)})
    conditions['swap_3_4']=[]
    geometry=backend._road_geometries();scale=backend._context().z_score
    model=model.eval().to(backend.device)
    for batch in backend._loader('test'):
        backend.check_budget()
        batch={k:v.to(backend.device) for k,v in batch.items()}
        out=model.encode_views(batch)
        ids=batch['traj_id'].tolist()
        for condition in conditions:
            backend.check_budget()
            if condition.startswith('prefix_'):
                depth=int(condition.split('_')[1]);values=out['prefix_vectors'][:,depth-1]
            else:
                selected_layers=[int(s)-1 for s in condition.split('_')[1:]]
                swapped=[]
                for tid in ids:
                    c=codes[tid].clone()
                    if tid in donor:
                        for layer in selected_layers: c[:,layer]=codes[donor[tid]][:,layer]
                    swapped.append(c)
                c=torch.cat(swapped).to(backend.device)
                values=model.rq_quant.encode(c.T)
            percent,dxy,lengths=model.decode_quantized(values,out['latent_lengths'],out['lens_type'],batch)
            for b,tid in enumerate(ids):
                if condition.startswith('swap_') and tid not in donor: continue
                length=int(lengths[b])
                row={'traj_id':tid,'status':'decoded','road_seq':items[tid]['route'].tolist(),
                     'pred_percent':percent[b,:length].cpu().tolist(),'pred_dxy':dxy[b,:length].cpu().tolist(),
                     'decoded_length':length}
                conditions[condition].append(reconstruct_gps(row,geometry,scale))
    result={'swap_eligible':len(donor),'total_validation':len(items),'donors':donor,
            'truncation_interpretation':'OOD diagnostic: omitted residual vectors are zero, not a trained variable-rate model',
            'swap_interpretation':'same event count and route-length bucket; donor chosen by hash then cyclic shift; association only',
            'conditions':{}}
    for condition,rows in conditions.items():
        ids={int(r['traj_id']) for r in rows}
        if not ids:
            result['conditions'][condition]={'status':'not_applicable','reason':'no matched donors'};continue
        real={tid:np.asarray(dataset.records_by_id[tid]['gps_seq']) for tid in ids}
        roads={tid:items[tid]['point_road'].tolist() for tid in ids}
        metrics=evaluate_population(real,roads,rows,geometry,backend.city,invalid_penalty=1.,
             mbr=dataset.metadata['mbr'],grid=dataset.metadata['grid'])
        errors=[{'traj_id':r['traj_id'],**progress_gps_errors(real[int(r['traj_id'])],r['gps_traj'])}
                 for r in rows if r['status']=='reconstructed']
        result['conditions'][condition]={'metrics':metrics,'gps_successful_only':errors}
        atomic_json(root/f'{condition}_reconstructed.json',rows)
    # Native full-prefix reference on the identical eligible swap population.
    eligible=[r for r in conditions['prefix_4'] if r['traj_id'] in donor]
    if eligible:
        real={tid:np.asarray(dataset.records_by_id[tid]['gps_seq']) for tid in donor}
        roads={tid:items[tid]['point_road'].tolist() for tid in donor}
        result['swap_native_reference']=evaluate_population(real,roads,eligible,geometry,backend.city,invalid_penalty=1.,
            mbr=dataset.metadata['mbr'],grid=dataset.metadata['grid'])
    atomic_json(root/'metrics.json',result)
    return result
