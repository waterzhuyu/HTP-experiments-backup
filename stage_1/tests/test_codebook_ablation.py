import copy
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from codebook_ablation.backend import resize_codebooks, BASE_SIZES
from codebook_ablation.validation import SETTINGS, validate_plan
from staged_attribution.token_protocol import codebook_protocol, current_vocab_sizes, special_tokens, serialize_answer, parse_answer, TokenProtocolError

path=Path(__file__).parent/'hierarchical_rqvae/test_hierarchical_model.py'
spec=importlib.util.spec_from_file_location('fixture_cb',path)
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)

@pytest.mark.parametrize('sizes', SETTINGS.values())
def test_protocol_roundtrip_and_bounds(sizes):
    row=[k-1 for k in sizes]
    with codebook_protocol(sizes):
        assert parse_answer(serialize_answer([0,1,0],[row]))['codes']==[row]
        assert len(special_tokens(3))==12+sum(sizes)+3
        for l in range(4):
            bad=row.copy();bad[l]=sizes[l]
            with pytest.raises(TokenProtocolError):serialize_answer([0,1,0],[bad])
    assert current_vocab_sizes()==BASE_SIZES

@pytest.mark.parametrize('sizes', SETTINGS.values())
def test_capacity_preserves_shared_parameters_rng_and_decoding(sizes):
    model=fixtures.model().eval()
    original={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.random.get_rng_state().clone()
    resize_codebooks(model,sizes,42)
    assert torch.equal(rng,torch.random.get_rng_state())
    for name,value in model.state_dict().items():
        if not name.startswith('rq_quant.'):
            assert torch.equal(value,original[name]),name
    assert model.rq_quant.n_embed==list(sizes)
    for i,book in enumerate(model.rq_quant.codebooks):
        n=min(BASE_SIZES[i],sizes[i]);key=f'rq_quant.codebooks.{i}.codebook.weight'
        assert torch.equal(book.codebook.weight[:n],original[key][:n])
        assert book.codebook.weight.abs().max()<=1/BASE_SIZES[i]
    batch=fixtures.fixtures.complete_batch()
    output=model(batch)
    assert torch.isfinite(output['loss'])
    codes=output['codes'][output['latent_mask']]
    with codebook_protocol(sizes):
        assert parse_answer(serialize_answer([0,0,0],codes))['codes']==codes.tolist()
    summed=model.rq_quant.encode(codes.T)
    assert torch.allclose(summed,output['prefix_vectors'][:,-1],atol=1e-7)
    a=model.decode(codes,output['latent_lengths'],output['lens_type'],batch)
    b=model.decode_quantized(summed,output['latent_lengths'],output['lens_type'],batch)
    for x,y in zip(a,b):assert torch.allclose(x,y,atol=1e-6)

def test_baseline_exact_and_context_exception_restore():
    model=fixtures.model().eval();batch=fixtures.fixtures.complete_batch()
    before=model(batch)
    assert resize_codebooks(model,BASE_SIZES,42) is model
    after=model(batch)
    assert torch.equal(before['codes'],after['codes'])
    assert torch.equal(before['loss'],after['loss'])
    with pytest.raises(RuntimeError):
        with codebook_protocol([2]*4):raise RuntimeError('test')
    assert current_vocab_sizes()==BASE_SIZES

def test_only_preregistered_capacity_may_vary():
    for name in SETTINGS:
        p=json.loads(Path(f'stage_1/configs/codebook_ablation/{name}.json').read_text())
        assert validate_plan(p)
        bad=copy.deepcopy(p);bad['model']['latent_dim']=128
        with pytest.raises(ValueError):validate_plan(bad)
        bad=copy.deepcopy(p);bad['tokenizer_training']['epochs']=10
        with pytest.raises(ValueError):validate_plan(bad)
        bad=copy.deepcopy(p);bad['cells']['H1']['prefix_depths']['grid']=2
        with pytest.raises(ValueError):validate_plan(bad)
