"""Same train batch, same precision objective gradients for frozen HV/AllRecon."""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from hierarchical_rqvae.preflight import atomic_json, sha256
from independent_latent_sid.backend import Backend as OriginalBackend
from independent_latent_sid.recon_backend import Backend as AllBackend


ROOT = Path('/workspace/guozuyu/HTP-independent-latent-sid-artifacts')
PREFLIGHT = Path('/workspace/guozuyu/HTP-hierarchical-rqvae-artifacts/hierarchical_rqvae_gps_grid_road_v1_20260907/preflight')
SOURCES = {
    'HV': (OriginalBackend, Path('stage_1/configs/independent_latent_sid/pilot.json'),
           ROOT / 'independent_latent_sid_20260917/pilot/porto/42/HV/attempt_001'),
    'HVAllRecon': (AllBackend, Path('stage_1/configs/independent_latent_sid/hv_allrecon.json'),
                   ROOT / 'hv_allrecon_20260917/pilot/porto/42/HVAllRecon/attempt_001'),
}


def pack(grad):
    return None if grad is None else grad.detach().float().flatten()


def norm(grad):
    return None if grad is None else float(grad.norm())


def cosine(a, b):
    if a is None or b is None or a.norm() == 0 or b.norm() == 0:
        return None
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def main():
    output = ROOT / 'independent_latent_sid_20260917/analysis/diagnosis_gradient_pair_20260924.json'
    if output.exists() or subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('output exists or dirty analysis worktree')
    torch.set_num_threads(4)
    rows = {}
    ids = None
    for name, (backend_type, config, run_dir) in SOURCES.items():
        manifest = json.loads((run_dir / 'run_manifest.json').read_text())
        if manifest['status'] != 'completed' or manifest['config_sha256'] != sha256(config) or manifest['preflight_sha256'] != sha256(PREFLIGHT / 'manifest.json'):
            raise ValueError(f'{name} identity mismatch')
        for relative, evidence in manifest['stages']['stage1']['artifacts'].items():
            if sha256(run_dir / relative) != evidence['sha256']:
                raise ValueError(f'{name} stage1 SHA mismatch')
        backend = backend_type(config, PREFLIGHT, name, 'pilot', 42, time.monotonic() + 1800)
        model = backend._load_model(run_dir).to('cuda').train()
        batch = {key: value.to('cuda') for key, value in next(iter(backend._loader('train'))).items()}
        batch_ids = batch['traj_id'].tolist()
        if ids is None:
            ids = batch_ids
        elif ids != batch_ids:
            raise ValueError('different train batch identities')
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # FP32 train mode, the same loss decomposition and weighted view terms.
        result = model(batch)
        terms = {'rec': result['reconstruction_loss'], 'vq': result['quant_loss'],
                 **{view: .1 * result['view_losses'][view] for view in ('grid', 'road', 'gps')}}
        parameters = [model.view_projections[view].weight for view in ('grid', 'road', 'gps')]
        grads = {key: [pack(g) for g in torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)] for key, loss in terms.items()}
        views = {}
        for index, view in enumerate(('grid', 'road', 'gps')):
            views[view] = {'weighted_objective_projection_grad_l2': {key: norm(value[index]) for key, value in grads.items()},
                           'rec_vq_cosine': cosine(grads['rec'][index], grads['vq'][index]),
                           'rec_own_aux_cosine': cosine(grads['rec'][index], grads[view][index]),
                           'vq_own_aux_cosine': cosine(grads['vq'][index], grads[view][index])}
        rows[name] = {'revision': manifest['revision'], 'manifest_sha256': sha256(run_dir / 'run_manifest.json'),
                      'checkpoint_sha256': sha256(run_dir / 'stage1/stage1_model.pt'),
                      'reconstruction_loss': float(result['reconstruction_loss'].detach()),
                      'quant_loss': float(result['quant_loss'].detach()), 'views': views}
    payload = {'status': 'completed', 'analysis_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'command': sys.argv, 'train_traj_ids': ids, 'preflight_sha256': sha256(PREFLIGHT / 'manifest.json'),
               'method': 'same first 128 train trajectories; FP32 train mode; dropout seed42; final frozen checkpoints; actual weighted component losses; projection weights only; no optimizer/clip',
               'limitations': 'single final-checkpoint batch, not training history; projection gradient does not represent all model parameters',
               'cells': rows}
    atomic_json(output, payload)
    print(json.dumps({name: row['views'] for name, row in rows.items()}, indent=2))


if __name__ == '__main__':
    main()
