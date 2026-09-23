"""Summarize full-population first-wrong SID replacement ADE by train-defined quartiles."""

import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path('/workspace/guozuyu/HTP-independent-latent-sid-artifacts/independent_latent_sid_20260917/analysis')
OLD = ROOT / 'diagnosis_replacement_full_20260923.json'
NEW = ROOT / 'diagnosis_replacement_gps_full_20260924.json'
OUTPUT = ROOT / 'diagnosis_replacement_gps_summary_20260924.json'


def read(path):
    return json.loads(path.read_text())


def summarize(rows):
    if not rows:
        return {'rows': 0}
    errors = [r['improvement'] for r in rows]
    ades = [r['ade_improvement_m'] for r in rows if r['ade_improvement_m'] is not None]
    return {'rows': len(rows), 'valid': len(ades), 'mean_main_improvement': statistics.mean(errors),
            'median_main_improvement': statistics.median(errors), 'main_better_fraction': sum(x > 0 for x in errors) / len(errors),
            'mean_ade_improvement_m': statistics.mean(ades) if ades else None,
            'median_ade_improvement_m': statistics.median(ades) if ades else None,
            'ade_better_fraction': sum(x > 0 for x in ades) / len(ades) if ades else None,
            'mean_original_error': statistics.mean(r['original_error'] for r in rows),
            'mean_original_ade_m': statistics.mean(r['original_ade_m'] for r in rows)}


def quartile(value, edges):
    return sum(value >= edge for edge in edges)


def main():
    if OUTPUT.exists() or subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('output exists or dirty analysis worktree')
    old, new = read(OLD), read(NEW)
    if old['sample_size'] != 1024 or new['sample_size'] != 1024:
        raise ValueError('wrong population')
    result = {'status': 'complete', 'analysis_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_revisions': {'main': old['analysis_revision'], 'gps_addendum': new['analysis_revision']},
              'method': 'full 1024 shared validation trips; first wrong code per layer/tuple; 100-progress-anchor ADE; train-only quartile cutoffs; error-conditioned descriptive estimates, not additive causal effects',
              'cells': {}}
    for cell in ('HV', 'LF'):
        a, b = old['cells'][cell], new['cells'][cell]
        if a['sample_ids'] != b['sample_ids'] or a['skipped'] != b['skipped'] or a['parity_max'] != b['parity_max']:
            raise ValueError('replacement population/parity changed')
        x = {(r['traj_id'], r['intervention']): r for r in a['rows']}
        y = {(r['traj_id'], r['intervention']): r for r in b['rows']}
        if len(x) != len(a['rows']) or set(x) != set(y):
            raise ValueError('replacement keys changed')
        for key in x:
            for field in ('event', 'original_error', 'replaced_error', 'improvement', 'valid'):
                if x[key][field] != y[key][field]:
                    raise ValueError(f'main-score parity failed: {key} {field}')
        if b['ade_parity_max_m'] > 1e-7:
            raise ValueError('ADE archive parity failed')
        groups = defaultdict(list)
        strata = defaultdict(list)
        for row in b['rows']:
            groups[row['intervention']].append(row)
            features = b['trip_features'][str(row['traj_id'])]
            for feature, edges in b['train_quartile_edges'].items():
                strata[(row['intervention'], feature, quartile(features[feature], edges))].append(row)
        result['cells'][cell] = {'population': len(b['sample_ids']), 'skipped': len(b['skipped']),
                                 'ade_parity_max_m': b['ade_parity_max_m'],
                                 'train_quartile_edges': b['train_quartile_edges'],
                                 'interventions': {name: summarize(rows) for name, rows in groups.items()},
                                 'strata': {f'{intervention}/{feature}/q{q+1}': summarize(rows) for (intervention, feature, q), rows in strata.items()}}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({cell: {'interventions': value['interventions'], 'ade_parity_max_m': value['ade_parity_max_m']} for cell, value in result['cells'].items()}, indent=2))


if __name__ == '__main__':
    main()
