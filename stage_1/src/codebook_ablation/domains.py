"""Secondary B2/H1 domain comparison on frozen common train supports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from codebook_ablation.queue import ROOT,PREFLIGHT,SEEDS
from codebook_ablation.validation import SETTINGS
from codebook_ablation.collect import unique_completed,validate_identity
from hierarchical_rqvae.run import Backend
from hierarchical_rqvae.analyze import verify_attempt,interval
from hierarchical_rqvae.analyze_domains import centered_p
from hierarchical_rqvae.preflight import atomic_json,sha256
from staged_attribution.population_bootstrap import build_metric_contributions,population_regrets,save_metric_contributions,DOMAIN_METRICS

BOOTSTRAP_REPLICATES=20000
BOOTSTRAP_SEED=271828

SUPPORT_SHA='efa83492f44f2843018b850ecb68d4a9de56ffa3c7dcde093519924cb08e7b38'

def read(p):return json.loads(Path(p).read_text())

def validate_support(path):
    if sha256(path)!=SUPPORT_SHA:raise ValueError('frozen support changed')
    support=read(path)
    if support['preflight_sha256']!=sha256(PREFLIGHT/'manifest.json'):raise ValueError('support cohort mismatch')
    if support['training_population']!=8192 or support['validation_or_generation_used'] is not False:
        raise ValueError('support not training-only')
    return support

def compare(points,draws):
    rows=[]
    for metric in DOMAIN_METRICS:
        d=draws['H1'][metric]-draws['B2'][metric]
        estimate=float(points['H1'][metric][0]-points['B2'][metric][0])
        rows.append({'metric':metric,'baseline_regret':float(points['B2'][metric][0]),
            'candidate_regret':float(points['H1'][metric][0]),'difference_regret':estimate,
            'ci95':interval(d),'p_two_sided_centered_bootstrap':centered_p(d,estimate)})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--setting',choices=list(SETTINGS),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--support',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('refusing overwrite')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('commit before execution')
    support=validate_support(a.support);sources={};manifests={}
    for cell in ('B2','H1'):
        paths=sorted((ROOT/f'codebook_ablation_20260916_{a.setting}'/'pilot/porto'/str(a.seed)/cell).glob('attempt_*/run_manifest.json'))
        q=unique_completed(paths)
        if q is None:raise ValueError('complete B2/H1 pair required')
        m,count=verify_attempt(q.parent);validate_identity(m,a.setting,a.seed,cell)
        sources[cell]=q.parent;manifests[cell]={'path':str(q.parent),'manifest_sha256':sha256(q),'artifacts_verified':count,'training_revision':m['revision']}
    backend=Backend(Path(f'stage_1/configs/codebook_ablation/{a.setting}.json'),PREFLIGHT,'B2','pilot',a.seed,float('inf'))
    val=backend._dataset('test');geometry=backend._road_geometries();real={};roads={}
    for i in range(len(val)):
        item=val[i];tid=int(item['traj_id']);real[tid]=np.asarray(val.records_by_id[tid]['gps_seq'],dtype=np.float64);roads[tid]=item['point_road'].tolist()
    if len(real)!=1024:raise ValueError('invalid validation population')
    a.output.mkdir(parents=True);weights=np.random.default_rng(BOOTSTRAP_SEED).multinomial(1024,np.full(1024,1/1024),size=BOOTSTRAP_REPLICATES)
    points={};draws={};ids=None
    for cell,run in sources.items():
        generated=[json.loads(l) for l in (run/'decode/g11_reconstructed.jsonl').read_text().splitlines()]
        c=build_metric_contributions(real,roads,generated,geometry,'porto',histogram_edges=support['edges'],mbr=val.metadata['mbr'],grid=val.metadata['grid'])
        if ids is not None and c.traj_ids!=ids:raise ValueError('unpaired contribution rows')
        ids=c.traj_ids
        if set(ids)!=set(real):raise ValueError('contribution population mismatch')
        expected={int(r['traj_id']):r['valid'] for r in read(run/'stage4/stage4_metrics.json')['g11']['per_trajectory']}
        if c.valid.tolist()!=[expected[i] for i in ids]:raise ValueError('validity differs from primary')
        save_metric_contributions(c,a.output/cell)
        points[cell]=population_regrets(c);draws[cell]=population_regrets(c,weights,chunk_size=32)
    report={'status':'pair_completed','setting':a.setting,'seed':a.seed,'population':1024,
        'analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'command':sys.argv,'source_runs':manifests,'support_sha256':SUPPORT_SHA,
        'bootstrap_replicates':BOOTSTRAP_REPLICATES,'bootstrap_seed':BOOTSTRAP_SEED,'comparisons':compare(points,draws),
        'multiplicity':'raw p values only; final family is 5 capacities x 3 seeds x 8 metrics = 120, globally Holm adjusted after all pairs',
        'interpretation':'difference in failure_fraction + valid_fraction * aggregate_regret; lower is better; conditional on these checkpoints',
        'limitations':['secondary analysis','paired-trip confidence does not replace independent training seeds','same training-only bins for all capacities and seeds']}
    atomic_json(a.output/'report.json',report)
    files={str(q.relative_to(a.output)):sha256(q) for q in a.output.rglob('*') if q.is_file()}
    atomic_json(a.output/'artifact_manifest.json',{'status':'completed','files':files})
    print(json.dumps({'setting':a.setting,'seed':a.seed,'status':'pair_completed','output':str(a.output)}))

if __name__=='__main__':main()
