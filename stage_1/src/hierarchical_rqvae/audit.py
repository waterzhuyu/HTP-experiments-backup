"""Pointwise source alignment and cohort overlap audit, without test metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from staged_attribution.data import AlignedHTPDataset
from staged_attribution.prepare_data import normalize_gps, gps_to_grid_ids
from hierarchical_rqvae.preflight import atomic_json, sha256
from hierarchical_rqvae.run import verify_preflight


def audit(config, preflight):
    cfg=json.loads(Path(config).read_text());manifest=verify_preflight(preflight)
    cohort=Path(manifest['cohort']);source=Path(cfg['paths']['data_root'])/'porto/traj.parquet'
    if sha256(source)!=manifest['source_hashes'][str(source)]:
        raise ValueError('raw source hash changed')
    ids=set(manifest['train_ids']+manifest['validation_ids'])
    raw=pd.read_parquet(source,columns=['traj_id','geometry','opath','time','user_id'])
    raw=raw[raw.traj_id.isin(ids)].set_index('traj_id')
    if len(raw)!=len(ids) or raw.index.duplicated().any():
        raise ValueError('raw cohort membership mismatch')
    result={};road_sets={};user_sets={};dates={}
    for split in ('train','test'):
        dataset=AlignedHTPDataset(cohort,cohort/f'processed_{split}_traj.pkl',split)
        failures=[];max_error=0.;non_increasing=[];road_set=set();users=set();day_set=set()
        for index in range(len(dataset)):
            item=dataset[index];tid=item['traj_id'];record=dataset.records_by_id[tid];row=raw.loc[tid]
            gps=np.vstack(row.geometry).astype(np.float64)
            if not np.array_equal(gps,np.asarray(record['gps_seq'],dtype=np.float64)):
                failures.append([tid,'raw_processed_gps'])
            expected=normalize_gps(gps,dataset.metadata['mbr'])
            error=float(np.max(np.abs(item['gps']-expected)));max_error=max(max_error,error)
            if error>1e-6: failures.append([tid,'normalized_gps'])
            grids=gps_to_grid_ids(gps,dataset.metadata['mbr'],dataset.metadata['grid'])
            if not np.array_equal(grids,item['grid']):failures.append([tid,'grid'])
            if not np.array_equal(np.asarray(row.opath,dtype=np.int64),item['point_road']):failures.append([tid,'point_road'])
            times=np.asarray(record['time_seq'],dtype=np.float64)
            if not np.array_equal(np.asarray(row.time,dtype=np.float64),times):failures.append([tid,'time'])
            if not np.all(np.diff(times)>0): non_increasing.append(tid)
            road_set.update(item['point_road'].tolist());users.add(str(row.user_id));day_set.add(int(times[0]//86400))
        result[split]={'population':len(dataset),'max_normalization_error':max_error,'failures':failures,
                       'non_increasing_time_ids':non_increasing,'unique_users':len(users),'unique_roads':len(road_set),'unique_days':len(day_set)}
        road_sets[split]=road_set;user_sets[split]=users;dates[split]=day_set
    result['overlap']={'vehicles':len(user_sets['train']&user_sets['test']),
        'roads':len(road_sets['train']&road_sets['test']),'days':len(dates['train']&dates['test'])}
    result['interpretation']='held-out trips only; validation filename test refers to official-train holdout'
    result['passed']=not any(result[s]['failures'] or result[s]['non_increasing_time_ids'] for s in ('train','test'))
    atomic_json(Path(preflight)/'alignment_audit.json',result)
    print(json.dumps(result))
    if not result['passed']: raise ValueError('pointwise alignment/time audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--preflight',required=True)
    a=p.parse_args();audit(a.config,a.preflight)
