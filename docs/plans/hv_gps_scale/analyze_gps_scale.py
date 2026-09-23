"""Read-only full-population comparison of GPS scale, HV, and HV-AllRecon."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path('/workspace/guozuyu/HTP-independent-latent-sid-artifacts')
RUNS = {
    'HVGpsScale': ROOT / 'hv_gps_scale_20260924/pilot/porto/42/HVGpsScale/attempt_001',
    'HV': ROOT / 'independent_latent_sid_20260917/pilot/porto/42/HV/attempt_001',
    'HVAllRecon': ROOT / 'hv_allrecon_20260917/pilot/porto/42/HVAllRecon/attempt_001',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(name, path):
    manifest = read(path / 'run_manifest.json')
    if manifest['status'] != 'completed':
        raise ValueError(f'{name} incomplete')
    count = 0
    for stage in manifest['stages'].values():
        for relative, evidence in stage['artifacts'].items():
            artifact = path / relative
            if artifact.stat().st_size != evidence['bytes'] or sha(artifact) != evidence['sha256']:
                raise ValueError(f'{name}: artifact failed SHA: {relative}')
            count += 1
    g00 = read(path / 'stage4/stage4_metrics.json')['g00'] if name == 'HV' else read(path / 'oracle/g00_metrics.json')
    gps = read(path / 'diagnostics/gps_errors.json')['g00'] if name == 'HV' else read(path / 'oracle/gps_errors.json')
    info = read(path / 'diagnostics/information.json')['validation']
    layers = []
    for k, layer in zip((32, 64, 128, 256), info['layers']):
        perplexity = math.exp(layer['entropy_nats'])
        max_share = max(layer['counts']) / sum(layer['counts'])
        layers.append({'K': k, 'active': layer['active_codes'], 'perplexity': perplexity,
                       'max_share': max_share, 'pass': layer['active_codes'] / k >= .1 and perplexity / k >= .05 and max_share <= .5})
    probes = read(path / 'diagnostics/probes/metrics.json')
    if len(probes) != 24 or any(row['epochs'] != 10 or row['steps'] != 1860 for row in probes):
        raise ValueError(f'{name}: incomplete frozen probes')
    selected = [row for row in probes if row['representation'] == 'prefixes' and (row['view'], row['depth']) in (('grid', 1), ('road', 2), ('gps', 4))]
    result = {'manifest_sha256': sha(path / 'run_manifest.json'), 'revision': manifest['revision'], 'config_sha256': manifest['config_sha256'],
              'verified_artifacts': count, 'g00': g00, 'gps_errors': gps,
              'layers': layers, 'tuple_entropy_nats': info['tuple_entropy_nats'], 'tuple_collision_probability': info['pair_collision_probability'],
              'selected_probes': selected}
    if name == 'HVGpsScale':
        progress = read(path / 'stage1/training_progress.json')
        if progress['status'] != 'completed' or progress['step'] != 1280:
            raise ValueError('tokenizer did not complete 1280 steps')
        result['training_progress'] = progress
        result['initial_audit'] = read(path / 'initial_route_audit.json')
        result['final_audit'] = read(path / 'final_route_audit.json')
    return result


def paired(candidate, baseline, key, rng_indices, ids):
    c = {int(row['traj_id']): row[key] for row in candidate}
    b = {int(row['traj_id']): row[key] for row in baseline}
    if set(c) != set(b) or set(c) != set(ids):
        raise ValueError('paired population mismatch')
    x = np.array([c[i] for i in ids])
    y = np.array([b[i] for i in ids])
    xm, ym = x[rng_indices].mean(1), y[rng_indices].mean(1)
    return {'candidate': float(x.mean()), 'baseline': float(y.mean()), 'difference': float((x-y).mean()),
            'difference_ci95': np.quantile(xm-ym, [.025, .975]).tolist(),
            'relative_improvement_pct': float(100*(1-x.mean()/y.mean())),
            'relative_improvement_ci95_pct': np.quantile(100*(1-xm/ym), [.025, .975]).tolist()}


def main():
    output = ROOT / 'hv_gps_scale_20260924/analysis/completion_comparison.json'
    if output.exists() or subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('output exists or dirty analysis tree')
    sources = {name: source(name, path) for name, path in RUNS.items()}
    ids = sorted(int(row['traj_id']) for row in sources['HVGpsScale']['g00']['per_trajectory'])
    if len(ids) != len(set(ids)) or len(ids) != 1024:
        raise ValueError('wrong validation population')
    rng = np.random.default_rng(271828)
    indices = rng.integers(0, 1024, size=(2000, 1024))
    comparisons = {}
    for baseline in ('HV', 'HVAllRecon'):
        a, b = sources['HVGpsScale'], sources[baseline]
        if a['g00']['failed'] or b['g00']['failed']:
            raise ValueError('G00 failure requires explicit penalty handling for GPS metrics')
        comparisons[f'HVGpsScale-vs-{baseline}'] = {
            'g00': paired(a['g00']['per_trajectory'], b['g00']['per_trajectory'], 'penalized_error', indices, ids),
            'ade': paired(a['gps_errors'], b['gps_errors'], 'matched_progress_ade_m', indices, ids),
            'fde': paired(a['gps_errors'], b['gps_errors'], 'fde_m', indices, ids),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {name: {key: value for key, value in row.items() if key not in ('g00', 'gps_errors')} for name, row in sources.items()}
    result = {'status': 'complete_tokenizer_only', 'analysis_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source': summary, 'comparisons': comparisons,
              'bootstrap': {'unit': 'paired validation trajectory', 'replicates': 2000, 'seed': 271828, 'ci': 'percentile95; single training seed'},
              'penalty10': 'same as penalty1 because all G00 variants have zero failures',
              'gate': {'g00_no_worse_than_10pct': sources['HVGpsScale']['g00']['population_mean_penalized_error'] <= 1.1*sources['HV']['g00']['population_mean_penalized_error'],
                       'validity': sources['HVGpsScale']['g00']['validity_rate'] >= .99,
                       'all_layers_healthy': all(row['pass'] for row in sources['HVGpsScale']['layers']),
                       'llm': 'not run in this tokenizer-only experiment'}}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'comparisons': comparisons, 'gate': result['gate'], 'layers': summary['HVGpsScale']['layers']}, indent=2))


if __name__ == '__main__':
    main()
