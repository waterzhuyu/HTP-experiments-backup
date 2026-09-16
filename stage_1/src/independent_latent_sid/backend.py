from codebook_ablation.backend import Backend as BaseBackend
from independent_latent_sid.model import IndependentLatentSID

class Backend(BaseBackend):
    def _new_model(self):
        model = super()._new_model()
        construction = self.plan['cells'][self.cell_id].get('construction')
        if construction is None:
            return model
        import json, torch
        metadata = json.loads((self.aligned_root/'metadata.json').read_text())
        cell = self.plan['cells'][self.cell_id]
        # Reset seed: preserve inherited decoder/adapters/codebooks across all cells.
        from hierarchical_rqvae.run import _set_seed
        _set_seed(self.seed)
        road_features = torch.load(self.data_root/'road_emb_128d.pt', weights_only=True)
        return IndependentLatentSID(self._model_config(), road_features, metadata['num_grids'],
            cell['views'], cell['prefix_depths'], cell['aux_weights'], self.manifest['observed_vocab'],
            head_seed=self.seed, construction=construction)
