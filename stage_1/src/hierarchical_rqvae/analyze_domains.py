"""Training-only shared histogram supports; paired population bootstrap of three domains."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from hierarchical_rqvae.analyze import read, digest, verify_attempt, holm, interval, CONTRASTS
from hierarchical_rqvae.run import Backend
from staged_attribution.population_bootstrap import (
    build_metric_contributions, population_regrets, save_metric_contributions,
    HISTOGRAM_METRICS, DOMAIN_METRICS,
)
from staged_attribution.stage4_metrics import _segments, _radius, _point_road_distances


def training_edges(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or len(a)==0 or not np.isfinite(a).all(): raise ValueError('invalid training support data')
    if np.max(a)==np.min(a):
        c=float(a[0]); eps=max(1e-10,abs(c)*1e-10)
        interior=np.array([c-eps,c+eps])
    else:
        interior=np.unique(np.quantile(a,np.linspace(0.,1.,21)))
    # Finite limits serialize in strict JSON and retain under/overflow observations.
    limit=np.finfo(np.float64).max
    return np.r_[-limit,interior,limit].tolist()


def centered_p(draws, estimate):
    """Approximate two-sided null-centered bootstrap test, not P(improvement)."""
    return float((1+np.count_nonzero(np.abs(draws-estimate)>=abs(estimate)))/(len(draws)+1))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cells',nargs='+',default=['B0','B1','B2','H1','H2','H3']); args=p.parse_args()
    if args.output.exists(): raise ValueError('refusing overwrite')
    args.output.mkdir(parents=True)
    backend=Backend(args.config,args.root/'preflight','B2','pilot',42,float('inf'))
    train=backend._dataset('train'); val=backend._dataset('test'); geometry=backend._road_geometries()
    samples={k:[] for k in HISTOGRAM_METRICS}
    for item in (train[i] for i in range(len(train))):
        gps=np.asarray(train.records_by_id[int(item['traj_id'])]['gps_seq'],dtype=np.float64)
        segments=_segments(gps)
        samples['travel_distance_jsd'].append(float(segments.sum()))
        samples['segment_distance_jsd'].extend(segments.tolist())
        samples['radius_jsd'].append(float(_radius(gps)))
        samples['point_to_road_jsd'].extend(_point_road_distances(gps,item['point_road'].tolist(),geometry))
    supports={k:training_edges(v) for k,v in samples.items()}
    support_record={'method':'training_only_20_quantile_intervals_plus_under_overflow_constant_epsilon',
        'training_population':len(train),'preflight_sha256':digest(args.root/'preflight/manifest.json'),
        'edges':supports, 'analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    (args.output/'support.json').write_text(json.dumps(support_record,indent=2,allow_nan=False)+'\n')
    print('Training histogram supports frozen.',flush=True)
    real_gps={}; real_roads={}
    for item in (val[i] for i in range(len(val))):
        tid=int(item['traj_id']); real_gps[tid]=np.asarray(val.records_by_id[tid]['gps_seq'],dtype=np.float64)
        real_roads[tid]=item['point_road'].tolist()
    n=len(val); weights=np.random.default_rng(271828).multinomial(n,np.full(n,1/n),size=2000)
    values={}; points={}; manifests={}; ids=None
    for c in args.cells:
        run=args.root/'pilot/porto/42'/c/'attempt_001'
        m,count=verify_attempt(run,next(iter(manifests.values()),None)); manifests[c]=m
        if m['preflight_sha256']!=support_record['preflight_sha256']: raise ValueError('support cohort mismatch')
        if m['config_sha256']!=digest(args.config): raise ValueError('config mismatch')
        generated=[json.loads(x) for x in (run/'decode/g11_reconstructed.jsonl').read_text().splitlines()]
        contributions=build_metric_contributions(real_gps,real_roads,generated,geometry,'porto',
            histogram_edges=supports,mbr=val.metadata['mbr'],grid=val.metadata['grid'])
        if ids is None: ids=contributions.traj_ids
        if contributions.traj_ids!=ids: raise ValueError('unpaired contributions')
        stage4=read(run/'stage4/stage4_metrics.json')['g11']
        expected={int(r['traj_id']):r['valid'] for r in stage4['per_trajectory']}
        if contributions.valid.tolist()!=[expected[i] for i in ids]: raise ValueError('stage4 validity differs')
        save_metric_contributions(contributions,args.output/c)
        values[c]=population_regrets(contributions,weights,chunk_size=32)
        points[c]=population_regrets(contributions)
        print(f'{c}: contributions and 2000 bootstrap draws complete.',flush=True)
    comparisons=[]
    for b,c in CONTRASTS:
        if b not in values or c not in values: continue
        for metric in DOMAIN_METRICS:
            draws=values[c][metric]-values[b][metric]
            estimate=float(points[c][metric][0]-points[b][metric][0])
            comparisons.append({'contrast':f'{c}-{b}','metric':metric,'baseline_regret':float(points[b][metric][0]),
                'candidate_regret':float(points[c][metric][0]),'difference_regret':estimate,'ci95':interval(draws),
                'p_two_sided_centered_bootstrap':centered_p(draws,estimate)})
    # Prespecified complete family: 5 contrasts x 8 metrics. Missing tests p=1 until all cells finish.
    raw=[r['p_two_sided_centered_bootstrap'] for r in comparisons]
    adjusted=holm(raw+[1.]*(40-len(raw)))[:len(raw)]
    for r,q in zip(comparisons,adjusted): r['holm_adjusted_p']=q
    report={'status':'complete' if len(args.cells)==6 else 'provisional', 'command':sys.argv,
        'analysis_revision':support_record['analysis_revision'],'training_revision':next(iter(manifests.values()))['revision'],
        'support_sha256':digest(args.output/'support.json'),'bootstrap_replicates':2000,'bootstrap_seed':271828,
        'population':n,'family_size':40,'comparisons':comparisons,
        'method':'failure_fraction + valid_fraction * aggregate_regret; recomputed per paired trajectory bootstrap',
        'limitations':['secondary common-support specification finalized after primary results inspected',
                       'approximate centered bootstrap p-values; single seed; CI conditional on trained checkpoints',
                       'density/pattern scores converted to regret=1-score; all reported differences lower is better',
                       'histogram quantiles determined solely from frozen training cohort; no validation tuning']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
