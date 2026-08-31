from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from staged_attribution.adapter_storage import (
    ADAPTER_MODEL,
    COMPACT_MANIFEST,
    EXTENSION_MODEL,
    AdapterStorageError,
    compact_peft_adapter,
    restore_and_validate_loaded_compact_adapter,
    sha256_file,
    validate_compact_adapter,
)
from staged_attribution.generate_stage3 import run_hf_generation


INPUT_BASE = "base_model.model.model.embed_tokens.token_adapter.base_layer.weight"
OUTPUT_BASE = "base_model.model.lm_head.token_adapter.base_layer.weight"
INPUT_DELTA = "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta"
OUTPUT_DELTA = "base_model.model.lm_head.token_adapter.trainable_tokens_delta"
LORA = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"


def _adapter(tmp_path, *, equal_bases: bool = True):
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "config.json").write_text(
        '{"model_type":"qwen3","vocab_size":8}\n'
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base_model)}) + "\n"
    )
    base = torch.arange(48, dtype=torch.float32).reshape(12, 4).to(torch.bfloat16)
    output = base.clone()
    if not equal_bases:
        output[0, 0] += 1
    tensors = {
        INPUT_BASE: base,
        OUTPUT_BASE: output,
        INPUT_DELTA: torch.arange(16, dtype=torch.float32).reshape(4, 4),
        OUTPUT_DELTA: torch.arange(16, dtype=torch.float32).reshape(4, 4),
        LORA: torch.ones((2, 4), dtype=torch.float32),
    }
    save_file(tensors, adapter / ADAPTER_MODEL, metadata={"format": "pt"})
    return adapter, base_model


def test_compact_adapter_removes_only_reconstructible_duplicate_bases(tmp_path) -> None:
    adapter, base_model = _adapter(tmp_path)
    original_bytes = (adapter / ADAPTER_MODEL).stat().st_size
    with safe_open(adapter / ADAPTER_MODEL, framework="pt", device="cpu") as handle:
        expected_extension = handle.get_tensor(INPUT_BASE)[8:].clone()
    payload = compact_peft_adapter(
        adapter,
        base_model_path=base_model,
        training_seed=42,
    )

    with safe_open(adapter / ADAPTER_MODEL, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    assert INPUT_BASE not in keys
    assert OUTPUT_BASE not in keys
    assert keys == {INPUT_DELTA, OUTPUT_DELTA, LORA}
    assert payload["training_seed"] == 42
    assert payload["base_vocab_size"] == 8
    assert payload["base_embedding_extension_shape"] == [4, 4]
    assert (adapter / EXTENSION_MODEL).is_file()
    with safe_open(adapter / EXTENSION_MODEL, framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor("new_rows"), expected_extension)
    assert payload["original_bytes"] == original_bytes
    assert payload["compact_bytes"] == (
        (adapter / ADAPTER_MODEL).stat().st_size
        + (adapter / EXTENSION_MODEL).stat().st_size
    )
    assert payload["saved_bytes"] > 0
    assert payload["removed_tensors"][0]["sha256"] == payload["removed_tensors"][1]["sha256"]
    assert validate_compact_adapter(adapter) == payload
    assert json.loads((adapter / COMPACT_MANIFEST).read_text()) == payload


def test_compaction_rejects_nonidentical_base_embeddings_without_mutation(tmp_path) -> None:
    adapter, base_model = _adapter(tmp_path, equal_bases=False)
    original_sha = sha256_file(adapter / ADAPTER_MODEL)
    with pytest.raises(AdapterStorageError, match="differ at row chunk"):
        compact_peft_adapter(adapter, base_model_path=base_model, training_seed=42)
    assert sha256_file(adapter / ADAPTER_MODEL) == original_sha
    assert not (adapter / COMPACT_MANIFEST).exists()


def test_compact_validation_rejects_model_tampering(tmp_path) -> None:
    adapter, base_model = _adapter(tmp_path)
    compact_peft_adapter(adapter, base_model_path=base_model, training_seed=42)
    model_file = adapter / ADAPTER_MODEL
    data = bytearray(model_file.read_bytes())
    data[-1] ^= 1
    model_file.write_bytes(data)
    with pytest.raises(AdapterStorageError, match="model SHA256 mismatch"):
        validate_compact_adapter(adapter)


def test_compact_validation_rejects_extension_tampering(tmp_path) -> None:
    adapter, base_model = _adapter(tmp_path)
    compact_peft_adapter(adapter, base_model_path=base_model, training_seed=42)
    extension_file = adapter / EXTENSION_MODEL
    data = bytearray(extension_file.read_bytes())
    data[-1] ^= 1
    extension_file.write_bytes(data)
    with pytest.raises(AdapterStorageError, match="extension SHA256 mismatch"):
        validate_compact_adapter(adapter)


def test_legacy_adapter_restore_is_noop(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    class LegacyModel:
        def named_parameters(self):
            raise AssertionError("legacy adapter must not inspect model parameters")

    restore_and_validate_loaded_compact_adapter(LegacyModel(), adapter)


def test_loaded_compact_adapter_verifies_reconstructed_tensor_bytes(tmp_path) -> None:
    adapter, base_model = _adapter(tmp_path)
    payload = compact_peft_adapter(adapter, base_model_path=base_model, training_seed=42)
    restored = torch.arange(48, dtype=torch.float32).reshape(12, 4).to(torch.bfloat16)
    randomly_initialized = restored.clone()
    randomly_initialized[8:] = -1

    class LoadedModel:
        def named_parameters(self, remove_duplicate=True):
            assert remove_duplicate is False
            for item in payload["removed_tensors"]:
                yield item["name"], torch.nn.Parameter(
                    randomly_initialized.clone(), requires_grad=False
                )

    model = LoadedModel()
    restore_and_validate_loaded_compact_adapter(model, adapter)
    restored[0, 0] += 1

    class WrongBaseModel:
        def named_parameters(self, remove_duplicate=True):
            assert remove_duplicate is False
            for item in payload["removed_tensors"]:
                yield item["name"], torch.nn.Parameter(restored.clone(), requires_grad=False)

    with pytest.raises(AdapterStorageError, match="tensor SHA256 mismatch"):
        restore_and_validate_loaded_compact_adapter(WrongBaseModel(), adapter)


def test_generation_uses_generation_seed_before_and_after_model_loading(
    tmp_path, monkeypatch
) -> None:
    seed_calls: list[int] = []
    load_seed: list[int] = []

    class FakeTokenizer:
        eos_token = "<eos>"
        eos_token_id = 0
        unk_token_id = -1
        pad_token = "<eos>"
        pad_token_id = 0
        padding_side = "right"

        @classmethod
        def from_pretrained(cls, _path):
            return cls()

        def convert_tokens_to_ids(self, values):
            return list(range(1, len(values) + 1))

        def __call__(self, prompts, **_kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]] * len(prompts), dtype=torch.long),
                "attention_mask": torch.ones((len(prompts), 2), dtype=torch.long),
            }

        def batch_decode(self, values, **_kwargs):
            answer = (
                "<t_0>, <|t_begin|><|p_begin|><a_0><b_0><c_0><d_0>"
                "<|p_end|><|t_end|>"
            )
            return [answer] * len(values)

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _path, **_kwargs):
            load_seed.append(seed_calls[-1])
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, input_ids, attention_mask, **_kwargs):
            suffix = torch.ones((len(input_ids), 1), dtype=input_ids.dtype)
            return torch.cat([input_ids, suffix], dim=1)

    def fake_set_seed(value):
        seed_calls.append(int(value))

    monkeypatch.setattr(
        "staged_attribution.generate_stage3.restore_and_validate_loaded_compact_adapter",
        lambda _model, _path: None,
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(AutoPeftModelForCausalLM=FakeModel))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer, set_seed=fake_set_seed),
    )
    frame = pd.DataFrame([{"traj_id": 1, "desc": "route"}])
    result = run_hf_generation(
        frame,
        tmp_path / "adapter",
        tmp_path / "generation.jsonl",
        num_roads=2,
        seed=42,
        device="cpu",
    )
    assert load_seed == [42]
    assert seed_calls == [42, 42]
    assert result["audit"] == {"population": 1, "completed": 1, "failed": 0}
