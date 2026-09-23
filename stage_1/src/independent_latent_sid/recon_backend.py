"""Tokenizer-only reconstruction-route ablation; no LLM training."""
import json
import math
import torch
import numpy as np
from independent_latent_sid.backend import Backend as OriginalBackend
from hierarchical_rqvae.preflight import atomic_json
from hierarchical_rqvae.diagnostics import collect_codes, frozen_probes, progress_gps_errors
from hierarchical_rqvae.analyze_information import count_summary, unigram_cross_entropy
from staged_attribution.pipeline import make_collator, decode_variant_records, reconstruct_gps, _atomic_jsonl
from staged_attribution.stage4_metrics import evaluate_population
from staged_attribution.g_attribution import _unique, _validate_oracle, _ready


class Backend(OriginalBackend):
    def _new_model(self):
        model = super()._new_model()
        model.reconstruction_route = self.plan['cells'][self.cell_id]['reconstruction_route']
        model.reconstruction_scale = self.plan['cells'][self.cell_id]['reconstruction_scale']
        return model

    def _route_audit(self, run_dir, trained=False):
        model = (self._load_model(run_dir) if trained else self._new_model()).to(self.device)
        batch = {k:v.to(self.device) for k,v in next(iter(self._loader('test'))).items()}
        evidence = {}
        original_route, original_scale = model.reconstruction_route, model.reconstruction_scale
        for route, scale, label in (('gps_only', 1.0, 'gps_only_baseline'),
                                    ('gps_only', original_scale, 'gps_only_scaled'),
                                    ('all_views', 1.0, 'all_views_reference')):
            model.reconstruction_route = route
            model.reconstruction_scale = scale
            model.train(); model.zero_grad(set_to_none=True)
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(batch)
            # Reconstruction only: no optimizer step, auxiliary or VQ backward.
            out['reconstruction_loss'].backward()
            def norm(parameters):
                return math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in parameters if p.grad is not None))
            evidence[label] = {'reconstruction_loss':float(out['reconstruction_loss'].detach()),
                'encoder_grad_l2':{v:norm(model.view_encoders[v].parameters()) for v in ('grid','road','gps')},
                'projection_grad_l2':{v:norm(model.view_projections[v].parameters()) for v in ('grid','road','gps')},
                'codebook_grad_l2':norm(model.rq_quant.parameters())}
        model.reconstruction_route, model.reconstruction_scale = original_route, original_scale
        p = run_dir/('final_route_audit.json' if trained else 'initial_route_audit.json')
        atomic_json(p, {'mode':'same-checkpoint, fixed first validation batch, train-mode BF16, no optimizer; pre-clipping gradient norms',
            'traj_ids':batch['traj_id'].tolist(), 'seed':self.seed, 'routes':evidence})
        return {'audit':p}

    def _initial_audit(self, run_dir):
        return self._route_audit(run_dir)

    def _oracle(self, run_dir):
        dataset = self._dataset('test')
        records = [json.loads(s) for s in (run_dir/'tokens/test_tokens.jsonl').read_text().splitlines()]
        oracle = []
        for tid,row in sorted(_unique(records, 'oracle').items()):
            _validate_oracle(row)
            oracle.append(_ready(tid, 'g00', row, {}, 'oracle', 'oracle'))
        collate = make_collator(self._context()); contexts = {}; real_gps = {}; real_roads = {}
        for i in range(len(dataset)):
            item = dataset[i]; tid = int(item['traj_id']); batch = collate([item])
            contexts[tid] = {k:batch[k] for k in ('route','route_lengths','route_geometry','route_geometry_lengths')}
            real_gps[tid] = np.asarray(dataset.records_by_id[tid]['gps_seq'], dtype=np.float64)
            real_roads[tid] = item['point_road'].tolist()
        self.check_budget()
        geometry = self._road_geometries()
        decoded = decode_variant_records(self._load_model(run_dir), oracle, contexts, self.device)
        rows = [reconstruct_gps(r, geometry, self._context().z_score) for r in decoded]
        if len(rows)!=len(dataset) or {int(r['traj_id']) for r in rows}!=set(real_gps):
            raise ValueError('G00 population mismatch')
        output=run_dir/'oracle'; output.mkdir(exist_ok=True)
        _atomic_jsonl(output/'g00_reconstructed.jsonl', rows)
        metrics = evaluate_population(real_gps,real_roads,rows,geometry,self.city,invalid_penalty=1.,mbr=dataset.metadata['mbr'],grid=dataset.metadata['grid'])
        atomic_json(output/'g00_metrics.json', metrics)
        errors = [{'traj_id':r['traj_id'], **progress_gps_errors(real_gps[int(r['traj_id'])],r['gps_traj'])} if r['status']=='reconstructed' else {'traj_id':r['traj_id'],'status':'failed','gps_errors':None} for r in rows]
        atomic_json(output/'gps_errors.json',errors)
        return {p.name:p for p in output.iterdir() if p.is_file()}

    def _tokenizer_diagnostics(self, run_dir):
        output=run_dir/'diagnostics'; output.mkdir(exist_ok=True)
        model=self._load_model(run_dir)
        train,_=collect_codes(model,self._loader('train'),self.device,self.check_budget)
        val,sampling=collect_codes(model,self._loader('test'),self.device,self.check_budget,sampling=True)
        torch.save({'train':train,'validation':val},output/'frozen_features.pt')
        atomic_json(output/'sampling.json',sampling)
        sizes=self.plan['model']['vocab_sizes']
        atomic_json(output/'information.json',{'train':count_summary(train['codes'].numpy(),sizes),
            'validation':count_summary(val['codes'].numpy(),sizes),
            'unigram_nll_per_layer':unigram_cross_entropy(train['codes'].numpy(),val['codes'].numpy(),sizes),
            'LLM':'not executed; these are code frequency reference NLLs, not model NLL'})
        model.eval().to(self.device); vectors={v:[] for v in ('grid','road','gps')}
        with torch.no_grad():
            for batch in self._loader('test'):
                self.check_budget()
                batch={k:v.to(self.device) for k,v in batch.items()}
                z,*_=model.independent_latents(batch)
                for v in vectors:vectors[v].append(z[v].cpu())
        ranks={}
        for v,parts in vectors.items():
            x=torch.cat(parts).double(); singular=torch.linalg.svdvals(x-x.mean(0))
            p=singular/singular.sum().clamp_min(1e-12)
            ranks[v]={'events':len(x),'effective_rank_singular_entropy':float((-(p*p.clamp_min(1e-12).log()).sum()).exp()),'mean_norm':float(x.norm(dim=1).mean())}
        atomic_json(output/'latent_rank.json',ranks)
        del model
        meta=json.loads((self.aligned_root/'metadata.json').read_text())
        frozen_probes(train,val,meta['num_grids'],meta['num_roads'],output/'probes',self.device,self.check_budget,epochs=1 if self.phase=='smoke' else 10)
        self._route_audit(run_dir, trained=True)
        return {**{str(p.relative_to(output)):p for p in output.rglob('*') if p.is_file()},'final_route_audit':run_dir/'final_route_audit.json'}
