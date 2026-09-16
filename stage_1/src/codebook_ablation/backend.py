from __future__ import annotations
import json
import math
from pathlib import Path
import torch
from hierarchical_rqvae.run import Backend as OriginalBackend
from hierarchical_rqvae.model import Float32RQ
from hierarchical_rqvae.preflight import atomic_json
from staged_attribution.token_protocol import current_vocab_sizes

BASE_SIZES = (32, 64, 128, 256)

def resize_codebooks(model, sizes, seed):
    """Preserve all shared parameters, common codewords, layer scales, and RNG."""
    sizes = tuple(sizes)
    if sizes == BASE_SIZES:
        return model
    base = math.lcm(*sizes)
    old = model.rq_quant
    with torch.random.fork_rng(devices=[]):
        new = Float32RQ(base, old.codebooks[0].codebook.embedding_dim, 4,
                       [base // k for k in sizes], beta=old.beta)
        for layer, (before, after) in enumerate(zip(old.codebooks, new.codebooks)):
            torch.manual_seed(seed + 10000 + layer)
            with torch.no_grad():
                after.codebook.weight.uniform_(-1 / BASE_SIZES[layer], 1 / BASE_SIZES[layer])
                n = min(before.n_embed, after.n_embed)
                after.codebook.weight[:n].copy_(before.codebook.weight[:n])
    model.rq_quant = new
    return model

class Backend(OriginalBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tuple(self.plan['model']['vocab_sizes']) != current_vocab_sizes():
            raise ValueError('protocol/model capacity mismatch')
        self.resolved['codebook_initialization'] = 'base_layer_scale_nested_common_codes_v1'
        self.resolved['rq_vocab_sizes'] = list(current_vocab_sizes())

    def _new_model(self):
        # Build the original model first: variable K cannot shift decoder/fusion RNG.
        model = super()._new_model()
        return resize_codebooks(model, self.plan['model']['vocab_sizes'], self.seed)

    def _stage2(self, run_dir):
        paths = super()._stage2(run_dir)
        p = run_dir/'stage2/adapter/rq_protocol.json'
        atomic_json(p, {'vocab_sizes':list(current_vocab_sizes()),
                       'config_sha256':self.resolved['config_sha256']})
        paths['adapter/rq_protocol.json'] = p
        return paths

    def _stage3(self, run_dir):
        p = json.loads((run_dir/'stage2/adapter/rq_protocol.json').read_text())
        if p != {'vocab_sizes':list(current_vocab_sizes()), 'config_sha256':self.resolved['config_sha256']}:
            raise ValueError('adapter protocol identity mismatch')
        return super()._stage3(run_dir)

    def _diagnostics(self, run_dir):
        from hierarchical_rqvae.diagnostics import collect_codes, frozen_probes, progress_gps_errors
        from hierarchical_rqvae.analyze_information import count_summary, unigram_cross_entropy
        output=run_dir/'diagnostics'; output.mkdir(parents=True, exist_ok=True)
        model=self._load_model(run_dir)
        train,_=collect_codes(model,self._loader('train'),self.device,self.check_budget)
        val,sampling=collect_codes(model,self._loader('test'),self.device,self.check_budget,sampling=True)
        torch.save({'train':train,'validation':val},output/'frozen_features.pt')
        atomic_json(output/'sampling.json',sampling)
        metadata=json.loads((self.aligned_root/'metadata.json').read_text())
        frozen_probes(train,val,metadata['num_grids'],metadata['num_roads'],output/'probes',
            self.device,self.check_budget,epochs=1 if self.phase=='smoke' else 10)
        sizes=self.plan['model']['vocab_sizes']
        atomic_json(output/'information.json',{
            'train':count_summary(train['codes'].numpy(),sizes),
            'validation':count_summary(val['codes'].numpy(),sizes),
            'unigram_nll_per_layer':unigram_cross_entropy(train['codes'].numpy(),val['codes'].numpy(),sizes),
            'interventions':'N/A: this ablation uses frozen probes and production singleton GPS decoding; no legacy batch interventions'})
        dataset=self._dataset('test'); errors={}
        for variant in ('g00','g10','g01','g11'):
            rows=[json.loads(line) for line in (run_dir/'decode'/f'{variant}_reconstructed.jsonl').read_text().splitlines()]
            errors[variant]=[{'traj_id':r['traj_id'],**progress_gps_errors(dataset.records_by_id[int(r['traj_id'])]['gps_seq'],r['gps_traj'])}
                if r['status']=='reconstructed' else {'traj_id':r['traj_id'],'status':'failed','gps_errors':None} for r in rows]
        atomic_json(output/'gps_errors.json',errors)
        return {str(p.relative_to(output)):p for p in output.rglob('*') if p.is_file()}
