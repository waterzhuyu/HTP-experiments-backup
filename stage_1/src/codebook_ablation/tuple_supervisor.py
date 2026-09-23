"""Serially finish and verify the remaining codebook tuple audits.

The first job may already be running. Existing attempts are observed, never
restarted. Any missing, failed, or inconsistent result stops the queue.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ANALYSIS = Path('/workspace/guozuyu/HTP-codebook-ablation-artifacts/analysis')
ARTIFACTS = ANALYSIS.parent
WORKTREE = Path('/workspace/guozuyu/HTP-codebook-ablation-tuple-supervisor')
TRAINING_REVISION = '52b6666e6a65eeee7fbe7ecc7cae8b84548f7ea3'
ORIGINAL_ANALYSIS_REVISION = '32493fdfc2240ae141ba8abd668566bfdb1b07e3'
SUPERVISOR_PROGRESS = ANALYSIS / 'tuple_supervisor_001.progress.json'
JOBS = [('quarter', 314159, 'H1')] + [
    (setting, 20260907, cell)
    for setting in ('base', 'front_large', 'front_matched', 'scale4', 'quarter')
    for cell in ('B2', 'H1')
]


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as handle:
        json.dump(data, handle, indent=2)
    temporary.replace(path)


def progress(status: str, completed: int, **extra: object) -> None:
    data = dict(status=status, completed=completed, expected=len(JOBS),
                jobs=[dict(setting=s, seed=n, cell=c) for s, n, c in JOBS],
                updated_at=time.time(), **extra)
    if SUPERVISOR_PROGRESS.exists():
        SUPERVISOR_PROGRESS.unlink()
    save_json(SUPERVISOR_PROGRESS, data)


def audit_paths(setting: str, seed: int, cell: str):
    name = f'tuple_{setting}_{seed}_{cell}_cpu_001'
    output = ANALYSIS / name
    source = ARTIFACTS / f'codebook_ablation_20260916_{setting}' / 'pilot' / 'porto' / str(seed) / cell / 'attempt_001'
    return name, output, source


def verify(output: Path, source: Path, cell: str, revision: str) -> dict:
    metrics_path = output / cell / 'metrics.json'
    progress_path = output / cell / 'progress.json'
    metrics = json.loads(metrics_path.read_text())
    state = json.loads(progress_path.read_text())
    assert metrics['status'] == state['status'] == 'completed'
    assert metrics['population'] == state['population_completed'] == 1024
    assert metrics['training_revision'] == TRAINING_REVISION
    assert metrics['analysis_revision'] == revision
    assert metrics['source_manifest_sha256'] == hashlib.sha256(
        (source / 'run_manifest.json').read_bytes()).hexdigest()
    assert metrics['evaluation_device'] == 'cpu'
    assert metrics['evaluation_dtype'] == 'float32'
    assert metrics['restoration_verified_before_dtype_conversion']
    assert metrics['teacher_forced']['events'] == metrics['free_running']['events'] == 5996
    assert all(metrics['reevaluated_metrics']['denominators'][f'layer_{layer}'] == 5996
               for layer in range(4))
    assert metrics['sid_nll_absolute_difference'] <= .005
    return dict(
        status='completed', analysis_revision=revision,
        training_revision=TRAINING_REVISION,
        source_manifest_sha256=metrics['source_manifest_sha256'],
        population=1024, evaluation_device='cpu', evaluation_dtype='float32',
        elapsed_seconds=metrics['elapsed_seconds'],
        original_sid_nll=metrics['original_sid_nll'],
        sid_nll_absolute_difference=metrics['sid_nll_absolute_difference'],
        files={f'{cell}/{filename}': hashlib.sha256(
            (output / cell / filename).read_bytes()).hexdigest()
            for filename in ('metrics.json', 'progress.json')},
        teacher_tuple_accuracy=metrics['teacher_forced']['tuple_accuracy'],
        free_tuple_accuracy=metrics['free_running']['tuple_accuracy'],
        verified_at=time.time(),
    )


def check_existing_verification(output: Path, source: Path, cell: str, revision: str) -> None:
    old = json.loads((output / 'verification.json').read_text())
    fresh = verify(output, source, cell, revision)
    for key, value in fresh.items():
        if key != 'verified_at':
            assert old[key] == value, (output, key)


def launch(name: str, output: Path, source: Path, cell: str, revision: str) -> dict:
    receipt = ANALYSIS / f'{name}.launch.json'
    log = ANALYSIS / f'{name}.log'
    assert not any(path.exists() for path in (receipt, log, output))
    command = ['timeout', '--signal=TERM', '--kill-after=30s', '3600',
               'uv', 'run', 'python', '-m', 'codebook_ablation.tuple_accuracy',
               '--run', str(source), '--output', str(output), '--device', 'cpu',
               '--cpu-threads', '16']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='16',
                       MKL_NUM_THREADS='16')
    with log.open('x') as handle:
        process = subprocess.Popen(command, cwd=WORKTREE, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    data = dict(pid=process.pid, command=command, analysis_revision=revision,
                started_at=time.time(), device='cpu',
                evaluation_dtype='float32 after verified BF16 restoration',
                cpu_threads=16, timeout_seconds=3600, log=str(log),
                output=str(output), source=str(source),
                scientific_status='supplementary diagnostic awaiting parity; no retraining')
    save_json(receipt, data)
    return data


def main() -> None:
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=WORKTREE, text=True).strip()
    assert revision != ORIGINAL_ANALYSIS_REVISION
    assert not subprocess.check_output(['git', 'status', '--porcelain'],
                                       cwd=WORKTREE, text=True).strip()
    completed = 0
    progress('running', completed)
    try:
        for setting, seed, cell in JOBS:
            name, output, source = audit_paths(setting, seed, cell)
            receipt_path = ANALYSIS / f'{name}.launch.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt['output'] == str(output)
                assert receipt['command'][receipt['command'].index('--run') + 1] == str(source)
                assert receipt['command'][receipt['command'].index('--device') + 1] == 'cpu'
                job_revision = receipt['analysis_revision']
                assert job_revision in (revision, ORIGINAL_ANALYSIS_REVISION)
            else:
                assert output.exists() is False
                receipt = launch(name, output, source, cell, revision)
                job_revision = revision
            progress('running', completed, active=name, pid=receipt['pid'])
            while True:
                state_path = output / cell / 'progress.json'
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                running = subprocess.run(['ps', '-p', str(receipt['pid'])],
                                         capture_output=True).returncode == 0
                if state.get('status') == 'completed' and not running:
                    break
                if not running:
                    raise RuntimeError(f'{name} exited without completed result: {state}')
                if state.get('status') == 'failed' or state.get('error'):
                    raise RuntimeError(f'{name} failed: {state}')
                time.sleep(20)
            if (output / 'verification.json').exists():
                check_existing_verification(output, source, cell, job_revision)
            else:
                save_json(output / 'verification.json',
                          verify(output, source, cell, job_revision))
            completed += 1
            progress('running', completed, just_verified=name)
        progress('completed', completed)
    except Exception as error:
        progress('failed', completed, error=repr(error))
        raise


if __name__ == '__main__':
    main()
