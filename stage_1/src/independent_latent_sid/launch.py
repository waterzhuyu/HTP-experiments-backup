"""Start one bounded pilot only after both new cells pass full smoke evidence checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from hierarchical_rqvae.preflight import atomic_json, sha256


def validate_smoke(root, config_sha):
    evidence = {}
    for cell in ('LF', 'HV'):
        attempts = sorted((root / 'smoke/porto/42' / cell).glob('attempt_*/run_manifest.json'))
        accepted = []
        for path in attempts:
            m = json.loads(path.read_text())
            if m['status'] != 'completed' or m['config_sha256'] != config_sha:
                continue
            if set(m['stages']) != {'stage1', 'tokens', 'stage2_data', 'stage2', 'stage3',
                                     'g_attribution', 'decode', 'stage4', 'diagnostics'}:
                raise ValueError('incomplete smoke stages')
            for stage in m['stages'].values():
                for name, info in stage['artifacts'].items():
                    if sha256(path.parent / name) != info['sha256']:
                        raise ValueError(f'changed smoke artifact: {name}')
            for variant in ('g00', 'g10', 'g01', 'g11'):
                rows = [json.loads(x) for x in (path.parent / 'decode' / f'{variant}_reconstructed.jsonl').read_text().splitlines()]
                if len(rows) != 8 or len({x['traj_id'] for x in rows}) != 8:
                    raise ValueError('smoke population mismatch')
                if variant == 'g00' and any(x['status'] != 'reconstructed' for x in rows):
                    raise ValueError('oracle decoding failed')
            accepted.append({'path': str(path), 'sha256': sha256(path), 'revision': m['revision']})
        if len(accepted) != 1:
            raise ValueError(f'{cell}: expected one verified completed smoke')
        evidence[cell] = accepted[0]
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    config = Path(args.config).resolve()
    plan = json.loads(config.read_text())
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('dirty worktree')
    root = Path(plan['paths']['artifact_root']) / plan['experiment_id']
    evidence = validate_smoke(root, sha256(config))
    receipt = root / 'pilot_launch.json'
    # Exclusive creation also protects concurrent launchers from duplicating runs.
    with receipt.open('x') as handle:
        json.dump({'status': 'launching', 'smoke_evidence': evidence}, handle)
    command = ['uv', 'run', 'python', '-m', 'independent_latent_sid.run', '--config', str(config),
               '--preflight', plan['ablation']['preflight_path'], '--phase', 'pilot',
               '--cells', 'HV', 'LF', 'H1', 'B2', '--seed', '42']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1')
    with (root / 'pilot.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    atomic_json(receipt, {'status': 'launched', 'pid': process.pid, 'command': command,
        'gpu': args.gpu, 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'config_sha256': sha256(config), 'smoke_evidence': evidence, 'log': str(root/'pilot.log')})
    print(receipt.read_text())


if __name__ == '__main__':
    main()
