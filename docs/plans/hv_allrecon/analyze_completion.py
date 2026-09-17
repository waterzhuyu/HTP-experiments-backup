"""Read-only paired comparison for the completed tokenizer-only ablation."""
import json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
root=Path('/workspace/guozuyu/HTP-independent-latent-sid-artifacts')
a=root/'hv_allrecon_20260917/pilot/porto/42/HVAllRecon/attempt_001'
b=root/'independent_latent_sid_20260917/pilot/porto/42/HV/attempt_001'
def read(p):return json.loads(p.read_text())
audit={}
for name,p in [('HVAllRecon',a),('HV',b)]:
 m=read(p/'run_manifest.json');count=0
 assert m['status']=='completed'
 for stage in m['stages'].values():
  for rel,e in stage['artifacts'].items():
   f=p/rel;assert f.stat().st_size==e['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==e['sha256'];count+=1
 audit[name]={'revision':m['revision'],'artifacts_verified':count,'manifest_sha256':hashlib.sha256((p/'run_manifest.json').read_bytes()).hexdigest()}
cm=read(a/'oracle/g00_metrics.json');bm=read(b/'stage4/stage4_metrics.json')['g00']
ids=sorted(x['traj_id'] for x in cm['per_trajectory'])
assert len(ids)==len(set(ids))==1024 and set(ids)=={x['traj_id'] for x in bm['per_trajectory']}
rng=np.random.default_rng(271828);indices=rng.integers(0,1024,size=(2000,1024))
def paired(c,base,key):
 ca={x['traj_id']:x[key] for x in c};ba={x['traj_id']:x[key] for x in base}
 assert set(ca)==set(ba)==set(ids)
 x=np.array([ca[i] for i in ids]);y=np.array([ba[i] for i in ids]);xm=x[indices].mean(1);ym=y[indices].mean(1)
 return {'candidate':float(x.mean()),'baseline':float(y.mean()),'difference':float((x-y).mean()),'difference_ci95':np.quantile(xm-ym,[.025,.975]).tolist(),'relative_improvement_pct':float(100*(1-x.mean()/y.mean())),'relative_improvement_ci95_pct':np.quantile(100*(1-xm/ym),[.025,.975]).tolist()}
out={'status':'completed_tokenizer_only','analysis_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'audit':audit,'bootstrap':{'replicates':2000,'seed':271828,'unit':'paired validation trajectory','ci':'percentile95, descriptive single-seed; no multiplicity adjustment'},'g00':paired(cm['per_trajectory'],bm['per_trajectory'],'penalized_error')}
ce=read(a/'oracle/gps_errors.json');be=read(b/'diagnostics/gps_errors.json')['g00']
out['ade']=paired(ce,be,'matched_progress_ade_m');out['fde']=paired(ce,be,'fde_m')
out['validity']={'candidate':cm['validity_rate'],'baseline':bm['validity_rate'],'candidate_success':cm['successful'],'baseline_success':bm['successful']}
assert cm['failed']==bm['failed']==0
out['penalty10_sensitivity']='identical to penalty1 because neither G00 population contains failure'
for name,p in [('HVAllRecon',a),('HV',b)]:
 info=read(p/'diagnostics/information.json')['validation'];layers=[]
 for k,l in zip([32,64,128,256],info['layers']):
  pp=math.exp(l['entropy_nats']);share=max(l['counts'])/sum(l['counts']);active=l['active_codes']
  layers.append({'K':k,'active':active,'perplexity':pp,'max_share':share,'pass':active/k>=.1 and pp/k>=.05 and share<=.5})
 probes=read(p/'diagnostics/probes/metrics.json');assert len(probes)==24 and all(x['epochs']==10 and x['steps']==1860 for x in probes)
 selected=[x for x in probes if x['representation']=='prefixes' and (x['view'],x['depth']) in [('grid',1),('road',2),('gps',4)]]
 out[name]={'layers':layers,'tuple_entropy_nats':info['tuple_entropy_nats'],'tuple_collision_probability':info['pair_collision_probability'],'selected_probes':selected}
out['gate']={'g00_no_worse_than_10pct':out['g00']['candidate']<=1.1*out['g00']['baseline'],'validity':cm['validity_rate']>=.99 and bm['validity_rate']-cm['validity_rate']<=.01,'all_layers_healthy':all(x['pass'] for x in out['HVAllRecon']['layers']),'llm':'NOT LAUNCHED; health gate failed'}
out['latent_rank']=read(a/'diagnostics/latent_rank.json')
p=root/'hv_allrecon_20260917/analysis/completion_comparison.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
