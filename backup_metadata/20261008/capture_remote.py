"""Capture remote experiment Git history and uncommitted source without edits."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path('/workspace/guozuyu/HTP')
OUT = Path(tempfile.mkdtemp(prefix='htp-code-backup-20261008-'))

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-C', str(cwd), *args])

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

raw = git('worktree', 'list', '--porcelain').decode()
worktrees = []
for block in raw.strip().split('\n\n'):
    item = {}
    for line in block.splitlines():
        key, _, value = line.partition(' ')
        item[key] = value if value else True
    worktrees.append(item)
refs = git('for-each-ref', '--format=%(refname) %(objectname)').decode()
(OUT / 'refs.txt').write_text(refs)
(OUT / 'worktrees.txt').write_text(raw)
heads = sorted({w['HEAD'] for w in worktrees})
subprocess.run(['git', '-C', str(ROOT), 'bundle', 'create', str(OUT / 'htp-all.bundle'), '--all', *heads], check=True)
subprocess.run(['git', '-C', str(ROOT), 'bundle', 'verify', str(OUT / 'htp-all.bundle')], check=True, stdout=subprocess.DEVNULL)
assert refs == git('for-each-ref', '--format=%(refname) %(objectname)').decode(), 'Refs changed during capture'
supplement = []
excluded = []
code_ext = {'.py', '.sh', '.bash', '.fish', '.c', '.h', '.cpp', '.cu', '.js', '.ts', '.tsx', '.jsx', '.toml', '.yaml', '.yml', '.ini', '.cfg', '.ipynb'}
names = {'.gitignore', '.gitattributes', 'uv.lock', 'requirements.txt', 'Dockerfile', 'Makefile'}
with tarfile.open(OUT / 'uncommitted-source.tar.gz', 'w:gz') as archive:
    for item in worktrees:
        p = Path(item['worktree'])
        state_before = git('status', '--porcelain=v1', '-z', cwd=p)
        item['name'] = p.name
        item['status'] = git('status', '--short', cwd=p).decode()
        assert item['HEAD'] == git('rev-parse', 'HEAD', cwd=p).decode().strip()
        state_dir = OUT / 'states' / p.name
        state_dir.mkdir(parents=True)
        (state_dir / 'status.txt').write_text(item['status'])
        for args, filename in [(['diff', '--binary', 'HEAD'], 'working-vs-head.patch'), (['diff', '--cached', '--binary'], 'staged.patch')]:
            (state_dir / filename).write_bytes(git(*args, cwd=p))
        paths = set(git('ls-files', '-m', '-z', cwd=p).split(b'\0'))
        untracked = git('ls-files', '--others', '--exclude-standard', '-z', cwd=p).split(b'\0')
        for entry in untracked:
            if not entry:
                continue
            rel = Path(os.fsdecode(entry))
            is_config_json = rel.suffix == '.json' and any(part in {'config', 'configs'} for part in rel.parts)
            if rel.suffix in code_ext or rel.name in names or is_config_json:
                paths.add(entry)
            else:
                excluded.append({'worktree': p.name, 'path': str(rel)})
        for entry in sorted(paths):
            if not entry:
                continue
            rel = Path(os.fsdecode(entry))
            source = p / rel
            if not source.is_file() or source.is_symlink():
                continue
            if source.stat().st_size > 10 * 1024 * 1024:
                excluded.append({'worktree': p.name, 'path': str(rel), 'reason': 'over 10 MiB'})
                continue
            data_before = sha(source)
            archive.add(source, arcname=f'{p.name}/{rel}', recursive=False)
            assert data_before == sha(source), f'File changed during capture: {source}'
            supplement.append({'worktree': p.name, 'path': str(rel), 'size': source.stat().st_size, 'sha256': data_before})
        assert state_before == git('status', '--porcelain=v1', '-z', cwd=p), f'State changed: {p}'
manifest = {
    'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_host': '3090s', 'source_repository': str(ROOT),
    'scope': 'All Git refs and active worktree HEADs; staged/unstaged patches; untracked source and configuration files. Untracked data, checkpoints, logs, reports and ignored files excluded.',
    'worktrees': worktrees, 'refs': [line.split(' ', 1) for line in refs.splitlines()],
    'supplement_files': supplement, 'excluded_untracked_files': excluded,
    'bundle_sha256': sha(OUT / 'htp-all.bundle'),
    'supplement_sha256': sha(OUT / 'uncommitted-source.tar.gz'),
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
files = sorted(f for f in OUT.rglob('*') if f.is_file())
(OUT / 'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(OUT)}\n' for f in files))
payload = OUT / 'transfer.tar.gz'
with tarfile.open(payload, 'w:gz') as archive:
    for f in [*files, OUT / 'SHA256SUMS']:
        archive.add(f, arcname=str(f.relative_to(OUT)), recursive=False)
print(json.dumps({'remote_transfer': str(payload), 'transfer_sha256': sha(payload), 'transfer_size': payload.stat().st_size, 'worktrees': len(worktrees), 'refs': len(manifest['refs']), 'supplement_files': len(supplement)}))
