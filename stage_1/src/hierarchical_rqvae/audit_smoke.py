"""Run new diagnostics against immutable completed smoke artifacts, without retraining."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import torch
from hierarchical_rqvae.run import Backend
from hierarchical_rqvae.preflight import atomic_json, sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--preflight',required=True)
    parser.add_argument('--source-root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--cells',nargs='+',required=True)
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('commit diagnostic code before real-data validation')
    torch.set_num_threads(4)
    deadline=time.monotonic()+3600
    for cell in args.cells:
        source=Path(args.source_root)/cell/'attempt_001'
        previous=json.loads((source/'run_manifest.json').read_text())
        if previous['status']!='completed':raise ValueError('source smoke is not complete')
        for stage in previous['stages'].values():
            for relative,artifact in stage['artifacts'].items():
                if sha256(source/relative)!=artifact['sha256']:raise ValueError('source artifact corruption')
        destination=Path(args.output)/cell
        destination.mkdir(parents=True,exist_ok=False)
        for directory in ('stage1','stage2','stage2_data','tokens','decode','stage4'):
            (destination/directory).symlink_to(source/directory,target_is_directory=True)
        backend=Backend(args.config,args.preflight,cell,'smoke',42,deadline)
        backend.artifact_scope=Path(args.output)
        record={'status':'running','source':str(source),'source_revision':previous['revision'],
            'execution_revision':backend.resolved['revision'],'source_manifest_sha256':sha256(source/'run_manifest.json')}
        atomic_json(destination/'audit_manifest.json',record)
        try:
            print(cell+': diagnostics',flush=True)
            paths=backend._diagnostics(destination)
            record['artifacts']={str(p.relative_to(destination)):sha256(p) for p in paths.values()}
            record['status']='completed'
        except Exception as error:
            record['status']='failed';record['error']=repr(error);raise
        finally:
            atomic_json(destination/'audit_manifest.json',record)


if __name__=='__main__': main()
