"""Verify downloaded backup and materialize every recorded experiment version."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def run(*args):
    r = subprocess.run(args, check=True, capture_output=True, text=True)
    return r.stdout.strip()

def extract_safe(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as t:
        for member in t.getmembers():
            target = (destination / member.name).resolve()
            assert target.is_relative_to(destination.resolve()), member.name
            assert member.isfile() or member.isdir(), member.name
        t.extractall(destination)

receipt = json.loads((ROOT / 'capture_receipt.json').read_text())
assert sha(ROOT / 'transfer.tar.gz') == receipt['transfer_sha256']
extract_safe(ROOT / 'transfer.tar.gz', ROOT)
checks = []
for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
    digest, name = line.split('  ', 1)
    assert sha(ROOT / name) == digest, name
    checks.append(name)
manifest = json.loads((ROOT / 'manifest.json').read_text())
mirror = ROOT / 'repository.git'
run('git', 'clone', '--mirror', str(ROOT / 'htp-all.bundle'), str(mirror))
run('git', '-C', str(mirror), 'bundle', 'verify', str(ROOT / 'htp-all.bundle'))
fsck = run('git', '-C', str(mirror), 'fsck', '--full')
refs = run('git', '-C', str(mirror), 'for-each-ref', '--format=%(refname) %(objectname)')
assert refs == (ROOT / 'refs.txt').read_text().strip()
verified_worktrees = []
for w in manifest['worktrees']:
    target = ROOT / 'worktrees' / w['name']
    run('git', '-C', str(mirror), 'cat-file', '-e', w['HEAD'] + '^{commit}')
    run('git', '-C', str(mirror), 'worktree', 'add', '--detach', str(target), w['HEAD'])
    assert run('git', '-C', str(target), 'rev-parse', 'HEAD') == w['HEAD']
    assert run('git', '-C', str(target), 'status', '--porcelain') == ''
    verified_worktrees.append({'source': w['worktree'], 'local': str(target), 'revision': w['HEAD'], 'branch': w.get('branch', 'detached'), 'clean_committed_checkout': True})
extract_safe(ROOT / 'uncommitted-source.tar.gz', ROOT / 'uncommitted-source')
for f in manifest['supplement_files']:
    p = ROOT / 'uncommitted-source' / f['worktree'] / f['path']
    assert sha(p) == f['sha256'], str(p)
result = {'status': 'verified', 'archive_sha256': receipt['transfer_sha256'], 'verified_payload_files': len(checks), 'verified_refs': len(manifest['refs']), 'verified_worktrees': verified_worktrees, 'verified_supplement_files': len(manifest['supplement_files']), 'fsck_stdout': fsck, 'note': 'Worktree source directories contain committed revisions. Uncommitted files are preserved separately under uncommitted-source; patches under states. No environment or training runs executed.'}
(ROOT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'verified', 'refs': len(manifest['refs']), 'worktrees': len(verified_worktrees), 'supplement_files': len(manifest['supplement_files'])}))
