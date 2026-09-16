"""Freeze secondary histogram supports from train only, before pilot results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from hierarchical_rqvae.analyze_domains import training_edges
from hierarchical_rqvae.run import Backend
from hierarchical_rqvae.preflight import atomic_json,sha256
from staged_attribution.population_bootstrap import HISTOGRAM_METRICS
from staged_attribution.stage4_metrics import _segments,_radius,_point_road_distances
from codebook_ablation.queue import PREFLIGHT

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('refusing overwrite')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('commit before execution')
    config=Path('stage_1/configs/codebook_ablation/base.json')
    backend=Backend(config,PREFLIGHT,'B2','pilot',42,float('inf'))
    train=backend._dataset('train');geometry=backend._road_geometries()
    if len(train)!=8192:raise ValueError('incorrect training population')
    values={k:[] for k in HISTOGRAM_METRICS}
    for i in range(len(train)):
        item=train[i];gps=np.asarray(train.records_by_id[int(item['traj_id'])]['gps_seq'],dtype=np.float64)
        segments=_segments(gps)
        values['travel_distance_jsd'].append(float(segments.sum()))
        values['segment_distance_jsd'].extend(segments.tolist())
        values['radius_jsd'].append(float(_radius(gps)))
        values['point_to_road_jsd'].extend(_point_road_distances(gps,item['point_road'].tolist(),geometry))
    result={'status':'frozen','training_population':len(train),
        'preflight_sha256':sha256(PREFLIGHT/'manifest.json'),
        'analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'command':sys.argv,'method':'training-only 20-quantile intervals plus finite underflow/overflow; common to all capacities and seeds',
        'observations_per_metric':{k:len(v) for k,v in values.items()},
        'edges':{k:training_edges(v) for k,v in values.items()},
        'validation_or_generation_used':False}
    atomic_json(a.output,result);print(json.dumps({'status':'frozen','path':str(a.output),'sha256':sha256(a.output)}))

if __name__=='__main__':main()
