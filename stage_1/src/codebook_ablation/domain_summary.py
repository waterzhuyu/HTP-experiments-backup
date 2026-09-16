"""Verify all secondary pair artifacts before the global 120-test correction."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from codebook_ablation.domains import SUPPORT_SHA, DOMAIN_METRICS
from codebook_ablation.queue import SEEDS
from codebook_ablation.validation import SETTINGS
from codebook_ablation.collect import validate_identity
from hierarchical_rqvae.analyze import holm, verify_attempt
from hierarchical_rqvae.preflight import atomic_json, sha256


def summarize(reports):
    expected={(setting,seed) for setting in SETTINGS for seed in SEEDS}
    seen={}; rows=[]; revisions=set()
    for report in reports:
        key=(report['setting'],report['seed'])
        if key not in expected or key in seen:
            raise ValueError('unexpected or duplicate pair')
        if (report['status']!='pair_completed' or report['population']!=1024
                or report['support_sha256']!=SUPPORT_SHA
                or report['bootstrap_replicates']!=2000 or report['bootstrap_seed']!=271828):
            raise ValueError('pair protocol mismatch')
        metrics=[r['metric'] for r in report['comparisons']]
        if len(metrics)!=len(DOMAIN_METRICS) or set(metrics)!=set(DOMAIN_METRICS):
            raise ValueError('missing or duplicate metric')
        for cell in ('B2','H1'):
            revisions.add(report['source_runs'][cell]['training_revision'])
        seen[key]=report
        for row in report['comparisons']:
            p=row['p_two_sided_centered_bootstrap']
            values=[row['baseline_regret'],row['candidate_regret'],row['difference_regret'],p,*row['ci95']]
            if not np.isfinite(values).all() or not 0<=p<=1 or len(row['ci95'])!=2 or row['ci95'][0]>row['ci95'][1]:
                raise ValueError('invalid comparison values')
            if not np.isclose(row['difference_regret'],row['candidate_regret']-row['baseline_regret'],rtol=1e-8,atol=1e-12):
                raise ValueError('effect direction mismatch')
            rows.append({'setting':key[0],'seed':key[1],**row})
    if len(revisions)>1:raise ValueError('mixed training revisions')
    missing=sorted(expected-set(seen))
    result={'status':'incomplete' if missing else 'secondary_analysis_ready',
        'verified_pairs':len(seen),'expected_pairs':len(expected),'missing_pairs':missing,
        'family_size':len(expected)*len(DOMAIN_METRICS),'support_sha256':SUPPORT_SHA,
        'limitations':['Secondary comparisons conditional on individual checkpoints.',
            'Per-seed trajectory intervals do not establish cross-seed equivalence.',
            'This report alone cannot exclude capacity effects; use the primary crossed-seed analysis.']}
    if missing:return result
    rows.sort(key=lambda r:(list(SETTINGS).index(r['setting']),SEEDS.index(r['seed']),DOMAIN_METRICS.index(r['metric'])))
    adjusted=holm([r['p_two_sided_centered_bootstrap'] for r in rows])
    for row,p in zip(rows,adjusted):
        row['holm_p_global_120']=float(p)
    result['comparisons']=rows
    result['descriptive_equal_seed_means']=[{
        'setting':setting,'metric':metric,
        'mean_difference_regret':float(np.mean([r['difference_regret'] for r in rows if r['setting']==setting and r['metric']==metric]))
    } for setting in SETTINGS for metric in DOMAIN_METRICS]
    return result


def load_verified(path):
    path=Path(path);root=path.parent
    receipt=json.loads((root/'artifact_manifest.json').read_text())
    if receipt['status']!='completed' or 'report.json' not in receipt['files']:
        raise ValueError('incomplete artifact receipt')
    for name,digest in receipt['files'].items():
        artifact=(root/name).resolve()
        if not artifact.is_relative_to(root.resolve()) or sha256(artifact)!=digest:
            raise ValueError('artifact hash mismatch')
    report=json.loads(path.read_text())
    for cell in ('B2','H1'):
        source=report['source_runs'][cell];run=Path(source['path'])
        if sha256(run/'run_manifest.json')!=source['manifest_sha256']:
            raise ValueError('source manifest changed')
        manifest,_=verify_attempt(run)
        validate_identity(manifest,report['setting'],report['seed'],cell)
        if manifest['revision']!=source['training_revision']:
            raise ValueError('source revision mismatch')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reports',type=Path,nargs='*',default=[])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('refusing overwrite')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('commit before execution')
    result=summarize([load_verified(p) for p in args.reports])
    result.update(analysis_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=sys.argv,source_reports=[{'path':str(p),'sha256':sha256(p)} for p in args.reports])
    atomic_json(args.output,result)
    print(json.dumps({'status':result['status'],'verified_pairs':result['verified_pairs'],'output':str(args.output)}))


if __name__=='__main__':main()
