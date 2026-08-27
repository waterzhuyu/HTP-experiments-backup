from __future__ import annotations

import pandas as pd
import pytest
import torch

from staged_attribution.train_stage2 import (
    AssistantOnlyCollator,
    Stage2DataError,
    assert_equivalent_token_budget,
    build_stage2_plan,
    build_stage2_training_arguments,
    format_training_text,
    tokenize_training_record,
)


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) + 1 for character in text]


def test_assistant_only_mask_starts_at_answer_and_masks_padding() -> None:
    tokenizer = CharacterTokenizer()
    first = tokenize_training_record(tokenizer, {"traj_id": 1, "desc": "trip one", "answer": "<t_0>, answer"})
    second = tokenize_training_record(tokenizer, {"traj_id": 2, "desc": "x", "answer": "<t_1>, z"})

    batch = AssistantOnlyCollator(tokenizer)([first, second])

    assert batch["input_ids"].shape[0] == 2
    assert torch.all(batch["labels"][0, : first["label_start"]] == -100)
    assert torch.equal(
        batch["labels"][0, first["label_start"] : len(first["input_ids"])],
        batch["input_ids"][0, first["label_start"] : len(first["input_ids"])],
    )
    assert torch.all(batch["labels"][1, len(second["input_ids"]) :] == -100)
    assert torch.all(batch["attention_mask"][1, len(second["input_ids"]) :] == 0)


def test_training_format_preserves_original_system_and_think_block() -> None:
    text, answer_start = format_training_text("ROAD PROMPT", "SEMANTIC ID")

    assert text.startswith("<|im_start|>system\nYou are a professional spatial-temporal expert")
    assert "<|im_start|>user\nROAD PROMPT<|im_end|>" in text
    assert text[answer_start:] == "SEMANTIC ID<|im_end|>"
    assert text[:answer_start].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_equivalent_budget_requires_same_population_and_lengths() -> None:
    reference = [
        {"traj_id": 1, "prompt_tokens": 10, "answer_tokens": 9},
        {"traj_id": 2, "prompt_tokens": 12, "answer_tokens": 13},
    ]
    candidate = list(reversed(reference))

    assert_equivalent_token_budget(reference, candidate)

    with pytest.raises(Stage2DataError, match="population"):
        assert_equivalent_token_budget(reference, candidate[:1])
    changed = [dict(reference[0]), dict(reference[1], answer_tokens=14)]
    with pytest.raises(Stage2DataError, match="token budget"):
        assert_equivalent_token_budget(reference, changed)


def test_dry_plan_does_not_require_transformers_or_a_model() -> None:
    frame = pd.DataFrame(
        {
            "traj_id": [3, 4, 5],
            "desc": ["a", "bb", "ccc"],
            "answer": ["x", "yy", "zzz"],
        }
    )

    plan = build_stage2_plan(frame, epochs=2, max_steps=7, batch_size=2, gradient_accumulation=4)

    assert plan == {
        "population": 3,
        "epochs": 2,
        "max_steps": 7,
        "batch_size": 2,
        "gradient_accumulation": 4,
        "optimizer_steps_per_epoch": 1,
        "planned_optimizer_steps": 7,
    }


def test_training_arguments_match_locked_transformers_api(tmp_path) -> None:
    arguments = build_stage2_training_arguments(
        output_dir=tmp_path / "trainer",
        seed=42,
        epochs=1,
        max_steps=2,
        batch_size=4,
        gradient_accumulation=2,
        learning_rate=5e-4,
    )

    assert arguments.output_dir == str(tmp_path / "trainer")
    assert arguments.max_steps == 2
    assert arguments.per_device_train_batch_size == 4
    assert arguments.gradient_accumulation_steps == 2


def test_collator_rejects_missing_answer_span() -> None:
    tokenizer = CharacterTokenizer()
    with pytest.raises(Stage2DataError, match="label_start"):
        AssistantOnlyCollator(tokenizer)(
            [{"input_ids": [1, 2], "attention_mask": [1, 1], "label_start": 2, "traj_id": 9}]
        )
