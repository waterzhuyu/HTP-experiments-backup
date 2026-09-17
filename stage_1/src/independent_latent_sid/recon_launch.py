"""Exclusive detached launch for a committed tokenizer-only ablation."""
import argparse,json,os,subprocess
from pathlib import Path
from hierarchical_rqvae.preflight import atomic_json,sha256

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--phase',choices=['smoke','pilot'],required=True)
    parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('dirty worktree')
    config=Path(args.config).resolve();plan=json.loads(config.read_text())
    root=Path(plan['paths']['artifact_root'])/plan['experiment_id'];root.mkdir(parents=True,exist_ok=True)
    receipt=root/f'{args.phase}_launch.json';log=root/f'{args.phase}.log'
    command=['uv','run','python','-m','independent_latent_sid.recon_run','--config',str(config),'--preflight',plan['ablation']['preflight_path'],'--phase',args.phase,'--cells','HVAllRecon','--seed','42']
    with receipt.open('x') as f:json.dump({'status':'launching'},f)
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with log.open('xb') as f:
            process=subprocess.Popen(command,env=env,cwd=Path.cwd(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(receipt,{'status':'launched','pid':process.pid,'command':command,'gpu':args.gpu,'log':str(log),'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'config_sha256':sha256(config)})
    except Exception as e:
        atomic_json(receipt,{'status':'launch_failed','error':repr(e)});raise
    print(receipt.read_text())
if __name__=='__main__':main()
