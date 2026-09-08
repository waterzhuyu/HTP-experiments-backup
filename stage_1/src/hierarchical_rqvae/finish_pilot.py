"""One-shot dependent analysis queue, after the existing pilot process exits."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from hierarchical_rqvae.preflight import atomic_json

CELLS=('B0','B1','B2','H1','H2','H3')


def source_states(root):
    states={}
    for c in CELLS:
        p=root/'pilot/porto/42'/c/'attempt_001/run_manifest.json'
        states[c]=json.loads(p.read_text()) if p.exists() else {'status':'pending'}
    return states


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source-pid',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise ValueError('refusing overwrite of completion queue')
    a.output.mkdir(parents=True)
    with (a.output/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','status','--porcelain'],text=True).strip(): raise ValueError('analysis tree must be committed')
        record={'status':'waiting_for_existing_pilot','pid':os.getpid(),'source_pid':a.source_pid,'analysis_revision':revision,
            'root':str(a.root),'config':str(a.config),'command':sys.argv,'steps':[],'started_at':time.time()}
        try:
            source=psutil.Process(a.source_pid);birth=source.create_time()
            if 'hierarchical_rqvae.run' not in ' '.join(source.cmdline()): raise ValueError('source PID is not requested pilot')
            wait_deadline=time.monotonic()+8*3600
            while source.is_running() and source.create_time()==birth and source.status()!=psutil.STATUS_ZOMBIE:
                states=source_states(a.root)
                record['source_states']={c:{'status':s['status'],'stage':s.get('current_stage')} for c,s in states.items()}
                record['updated_at']=time.time();atomic_json(a.output/'queue_manifest.json',record)
                if time.monotonic()>wait_deadline: raise TimeoutError('pilot dependency still running after 8h; no restart')
                time.sleep(30)
            states=source_states(a.root)
            if any(s['status']!='completed' for s in states.values()): raise ValueError('pilot terminated without all six complete')
            total=sum(s['elapsed_seconds'] for s in states.values())
            # Conservative cumulative wall time on the single GPU, including diagnostic work.
            remaining=36*3600-total
            if remaining<=0: raise ValueError('pilot GPU-hour budget exhausted')
            free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<23*1024: raise ValueError('GPU0 not free for dependent diagnostics')
            if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=revision or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
                raise ValueError('analysis source changed while dependency ran')
            record['status']='running_postprocessing';record['pilot_elapsed_seconds']=total
            corrected=a.output/'corrected_interventions'
            commands=[
                [sys.executable,'-m','hierarchical_rqvae.correct_interventions','--root',str(a.root),'--config',str(a.config),'--output',str(corrected)],
                [sys.executable,'-m','hierarchical_rqvae.analyze','--root',str(a.root),'--interventions-root',str(corrected),'--output',str(a.output/'final_statistics.json')],
                [sys.executable,'-m','hierarchical_rqvae.analyze_domains','--root',str(a.root),'--config',str(a.config),'--output',str(a.output/'domains')],
                [sys.executable,'-m','hierarchical_rqvae.analyze_information','--root',str(a.root),'--output',str(a.output/'information.json')],
            ]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            post_started=time.monotonic()
            for index,command in enumerate(commands):
                timeout=remaining-(time.monotonic()-post_started)
                if timeout<=0: raise TimeoutError('cumulative 36h budget exhausted')
                step={'command':command,'status':'running','started_at':time.time()};record['steps'].append(step)
                atomic_json(a.output/'queue_manifest.json',record)
                with (a.output/f'step_{index+1}.log').open('w') as log:
                    subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
                step['status']='completed';step['elapsed_seconds']=time.time()-step['started_at']
                atomic_json(a.output/'queue_manifest.json',record)
            record['status']='analysis_artifacts_ready_for_final_review'
            record['postprocessing_elapsed_seconds']=time.monotonic()-post_started
            record['total_conservative_gpu_hours']=(total+record['postprocessing_elapsed_seconds'])/3600
        except Exception as error:
            record['status']='failed';record['error']=f'{type(error).__name__}: {error}'
            atomic_json(a.output/'queue_manifest.json',record)
            raise
        atomic_json(a.output/'queue_manifest.json',record)

if __name__=='__main__':main()
