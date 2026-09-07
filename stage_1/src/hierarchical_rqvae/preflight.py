"""Freeze an audited pilot cohort and train-only normalization before any training."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def fingerprint(record):
    digest = hashlib.sha256()
    for field in ('gps_seq', 'time_seq'):
        arr = np.asarray(record[field], dtype='<f8')
        if not np.isfinite(arr).all():
            raise ValueError('nonfinite canonical trajectory')
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def choose_groups(records, count, seed, excluded=()):
    groups = {}
    excluded = set(excluded)
    for record in records:
        key = fingerprint(record)
        if key in excluded:
            continue
        groups.setdefault(key, []).append(int(record['traj_id']))
    ordered = sorted(groups, key=lambda key: (hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest(), key))
    if len(ordered) < count:
        raise ValueError('not enough independent trajectory groups')
    # One deterministic representative per identical GPS/time group.
    return [(key, min(groups[key])) for key in ordered[:count]], groups


def export_cohort(root, metadata, source_root, records_by_id, groups):
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'metadata.json', metadata)
    source_ids = np.load(source_root / 'train_traj_ids.npy')
    source_offsets = np.load(source_root / 'train_offsets.npy')
    row_by_id = {int(t): i for i, t in enumerate(source_ids)}
    observed = {}
    for split, ids in groups.items():
        selected_records = [records_by_id[i] for i in ids]
        with (root / f'processed_{split}_traj.pkl').open('wb') as f:
            pickle.dump(selected_records, f, protocol=5)
        offsets = [0]
        for i in ids:
            row = row_by_id[i]
            offsets.append(offsets[-1] + int(source_offsets[row+1] - source_offsets[row]))
        np.save(root / f'{split}_traj_ids.npy', np.asarray(ids, dtype=np.int64))
        np.save(root / f'{split}_offsets.npy', np.asarray(offsets, dtype=np.int64))
        for view in ('gps', 'grid', 'road'):
            source = np.load(source_root / f'train_{view}.npy', mmap_mode='r')
            arrays = [source[source_offsets[row_by_id[i]]:source_offsets[row_by_id[i]+1]] for i in ids]
            values = np.concatenate(arrays)
            np.save(root / f'{split}_{view}.npy', values)
            if split == 'train' and view in ('grid', 'road'):
                observed[view] = len(np.unique(values))
    return observed


def prepare(config_path, output):
    cfg = json.loads(Path(config_path).read_text())
    output = Path(output)
    if (output / 'complete.json').exists():
        raise FileExistsError('immutable preflight already complete; verify instead of overwrite')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('preflight requires committed code')
    source = Path(cfg['paths']['porto_aligned_root'])
    data = Path(cfg['paths']['data_root']) / 'porto'
    metadata = json.loads((source / 'metadata.json').read_text())
    from staged_attribution.data import AlignedHTPDataset, assert_split_disjoint
    split_audit = assert_split_disjoint(source)
    dataset = AlignedHTPDataset(source, data / 'processed_train_traj.pkl', 'train')
    records = list(dataset.records_by_id.values())
    chosen, groups = choose_groups(records, cfg['data']['train_n'] + cfg['data']['validation_n'], cfg['data']['split_seed'])
    val_pairs = chosen[:cfg['data']['validation_n']]
    train_pairs = chosen[cfg['data']['validation_n']:]
    val_ids = [i for _, i in val_pairs]
    train_ids = [i for _, i in train_pairs]
    # Holdout membership is audited, without fitting models or measuring holdout metrics.
    with (data / 'processed_test_traj.pkl').open('rb') as f:
        test_records = pickle.load(f)
    test_pairs, test_groups = choose_groups(test_records, cfg['data']['heldout_test_n'], cfg['data']['split_seed'], groups)
    cohort = output / 'cohort'
    observed = export_cohort(cohort, metadata, source, dataset.records_by_id, {'train':train_ids, 'test':val_ids})
    selected = [dataset.records_by_id[i] for i in train_ids]
    dxy = np.concatenate([np.asarray(r['dxy'], dtype=np.float64) for r in selected])
    pct = np.concatenate([np.asarray(r['percent_dist'], dtype=np.float64) for r in selected])
    statistics = dict(zip(('mean_dx','mean_dy'), dxy.mean(0).tolist()))
    statistics.update(dict(zip(('std_dx','std_dy'), dxy.std(0).tolist())))
    statistics.update(mean_percent_dist=float(pct.mean()), std_percent_dist=float(pct.std()))
    if any(not np.isfinite(v) for v in statistics.values()) or any(statistics[k] <= 0 for k in ('std_dx','std_dy','std_percent_dist')):
        raise ValueError('invalid train-only normalization')
    atomic_json(cohort / 'z_score.json', statistics)
    files = [data / 'processed_train_traj.pkl', data / 'processed_test_traj.pkl', data / 'road_emb_128d.pt', data / 'rn/edge_info.csv', data / 'traj.parquet', source / 'metadata.json']
    files += sorted(source.glob('*.npy'))
    source_hashes = {str(p):sha256(p) for p in files}
    metric_path = Path('stage_1/src/staged_attribution/stage4_metrics.py')
    manifest = {
        'revision':revision, 'config_sha256':sha256(config_path), 'status':'prepared',
        'train_ids':train_ids, 'validation_ids':val_ids, 'sealed_test_ids':[i for _,i in test_pairs],
        'validation_source':'official_train', 'cohort_test_filename_means':'validation_only',
        'train_group_hashes':[k for k,_ in train_pairs], 'validation_group_hashes':[k for k,_ in val_pairs],
        'split_audit':split_audit, 'exact_duplicate_train_groups':sum(len(v)>1 for v in groups.values()),
        'cohort':str(cohort.resolve()), 'observed_vocab':observed,
        'normalization':statistics, 'source_hashes':source_hashes,
        'metric':{'source_sha256':sha256(metric_path), 'invalid_penalty':1., 'sensitivity_penalty':10.,
                  'formula':'mean_trip(mean_4_stats(abs(log1p(generated_km)-log1p(real_km)))); invalid=penalty'},
        'road_embedding_provenance':'inherited Node2Vec; graph-only code inspected; original artifact training manifest unavailable',
        'limitations':['same-city trip holdout; road and time overlap permitted','vehicle-disjoint generalization not claimed','original Node2Vec artifact training provenance not independently certified'],
        'cohort_hashes':{p.name:sha256(p) for p in sorted(cohort.iterdir()) if p.is_file()},
    }
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'complete.json', {'manifest_sha256':sha256(output / 'manifest.json'), 'revision':revision})
    print(json.dumps({'status':'prepared','train':len(train_ids),'validation':len(val_ids),'test_sealed':len(test_pairs),'observed_vocab':observed}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    prepare(args.config, args.output)
