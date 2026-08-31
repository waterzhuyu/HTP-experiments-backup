from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .adapter_storage import compact_peft_adapter
from .evaluate_stage2 import grouped_teacher_forced_accuracy
from .token_protocol import PREFIXES, VOCAB_SIZES, special_tokens


SYSTEM_MESSAGE = (
    "You are a professional spatial-temporal expert who needs to generate the possible "
    "sampling type and travel pattern for users based on their trip information."
)
THINK_BLOCK = "<think>\n\n</think>\n\n"


class Stage2DataError(ValueError):
    """Raised when Stage2 supervision or token budgets are not comparable."""


def format_training_text(description: str, answer: str) -> tuple[str, int]:
    prefix = (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{description}<|im_end|>\n"
        f"<|im_start|>assistant\n{THINK_BLOCK}"
    )
    text = prefix + answer + "<|im_end|>"
    return text, len(prefix)


def tokenize_training_record(tokenizer: Any, record: Mapping[str, object]) -> dict[str, object]:
    for field in ("traj_id", "desc", "answer"):
        if field not in record:
            raise Stage2DataError(f"training record missing {field}")
    text, character_start = format_training_text(str(record["desc"]), str(record["answer"]))
    prefix_text = text[:character_start]
    answer_text = text[character_start:]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if not answer_ids:
        raise Stage2DataError("answer tokenization is empty")
    input_ids = list(prefix_ids) + list(answer_ids)
    return {
        "traj_id": int(record["traj_id"]),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "label_start": len(prefix_ids),
        "prompt_tokens": len(prefix_ids),
        "answer_tokens": len(answer_ids),
    }


class AssistantOnlyCollator:
    def __init__(self, tokenizer: Any) -> None:
        if tokenizer.pad_token_id is None:
            raise Stage2DataError("tokenizer requires a pad_token_id")
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, object]]) -> dict[str, torch.Tensor]:
        if not features:
            raise Stage2DataError("cannot collate an empty Stage2 batch")
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for feature in features:
            ids = [int(value) for value in feature["input_ids"]]
            mask = [int(value) for value in feature["attention_mask"]]
            start = int(feature["label_start"])
            if len(ids) != len(mask):
                raise Stage2DataError("input_ids and attention_mask lengths differ")
            if start < 0 or start >= len(ids):
                raise Stage2DataError("label_start must point to a non-empty answer span")
            padding = maximum - len(ids)
            input_rows.append(ids + [self.pad_token_id] * padding)
            mask_rows.append(mask + [0] * padding)
            label_rows.append([-100] * start + ids[start:] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }


def _budget_map(records: Sequence[Mapping[str, object]]) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for record in records:
        traj_id = int(record["traj_id"])
        if traj_id in result:
            raise Stage2DataError("duplicate traj_id in token budget")
        result[traj_id] = (int(record["prompt_tokens"]), int(record["answer_tokens"]))
    return result


def assert_equivalent_token_budget(
    reference: Sequence[Mapping[str, object]], candidate: Sequence[Mapping[str, object]]
) -> None:
    reference_map = _budget_map(reference)
    candidate_map = _budget_map(candidate)
    if set(reference_map) != set(candidate_map):
        raise Stage2DataError("Stage2 population differs across cells")
    changed = [
        traj_id for traj_id in reference_map if reference_map[traj_id] != candidate_map[traj_id]
    ]
    if changed:
        raise Stage2DataError(f"Stage2 token budget differs for traj_id values {changed[:10]}")


def build_stage2_plan(
    frame: pd.DataFrame,
    epochs: int,
    max_steps: int,
    batch_size: int,
    gradient_accumulation: int,
) -> dict[str, int]:
    if frame["traj_id"].duplicated().any():
        raise Stage2DataError("Stage2 frame contains duplicate traj_id")
    if epochs <= 0 or batch_size <= 0 or gradient_accumulation <= 0:
        raise Stage2DataError("epochs, batch_size and gradient_accumulation must be positive")
    if max_steps == 0 or max_steps < -1:
        raise Stage2DataError("max_steps must be -1 or positive")
    optimizer_steps = math.ceil(math.ceil(len(frame) / batch_size) / gradient_accumulation)
    planned = max_steps if max_steps > 0 else epochs * optimizer_steps
    return {
        "population": len(frame),
        "epochs": epochs,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "optimizer_steps_per_epoch": optimizer_steps,
        "planned_optimizer_steps": planned,
    }


def build_stage2_training_arguments(
    output_dir: str | Path,
    seed: int,
    epochs: int,
    max_steps: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
) -> Any:
    """Construct arguments against the locked Transformers API."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        save_strategy="no",
        logging_steps=1 if max_steps > 0 else 10,
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )


class _TokenizedDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        return self.rows[index]


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def evaluate_teacher_forced_model(
    model: torch.nn.Module,
    dataset: Dataset,
    collator: AssistantOnlyCollator,
    token_groups: dict[str, set[int]],
    device: torch.device,
    batch_size: int = 4,
) -> dict[str, object]:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = {name: 0 for name in token_groups}
    denominators = {name: 0 for name in token_groups}
    first_errors: Counter[str] = Counter()
    exact = 0
    population = 0
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        output = model(**batch)
        shifted_logits = output.logits[:, :-1]
        shifted_labels = batch["labels"][:, 1:]
        tokens = int((shifted_labels != -100).sum())
        total_loss += float(
            F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        )
        total_tokens += tokens
        grouped = grouped_teacher_forced_accuracy(
            shifted_logits.argmax(dim=-1).cpu(), shifted_labels.cpu(), token_groups
        )
        population += int(grouped["population"])
        exact += round(float(grouped["sequence_exact_accuracy"]) * int(grouped["population"]))
        first_errors.update(grouped["first_error_histogram"])
        for name in token_groups:
            count = int(grouped["denominators"][name])
            denominators[name] += count
            if count:
                correct[name] += round(float(grouped["accuracy"][name]) * count)
    if total_tokens == 0:
        raise Stage2DataError("teacher-forced evaluation produced no supervised tokens")
    nll = total_loss / total_tokens
    return {
        "population": population,
        "tokens": total_tokens,
        "nll": nll,
        "perplexity": math.exp(nll),
        "accuracy": {
            name: correct[name] / denominators[name] if denominators[name] else float("nan")
            for name in token_groups
        },
        "denominators": denominators,
        "sequence_exact_accuracy": exact / population,
        "first_error_histogram": dict(first_errors),
    }


def run_hf_stage2(
    frame: pd.DataFrame,
    model_path: str | Path,
    output_dir: str | Path,
    num_roads: int,
    seed: int,
    epochs: int,
    max_steps: int,
    batch_size: int = 16,
    gradient_accumulation: int = 1,
    learning_rate: float = 5e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    eval_frame: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Train the frozen Stage2 recipe; heavyweight libraries are imported only here."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed

    set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = build_stage2_plan(frame, epochs, max_steps, batch_size, gradient_accumulation)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.pad_token = tokenizer.eos_token
    protocol_tokens = special_tokens(num_roads)
    added = tokenizer.add_special_tokens({"additional_special_tokens": protocol_tokens})
    token_ids = tokenizer.convert_tokens_to_ids(protocol_tokens)
    valid_ids = [value for value in token_ids if value != tokenizer.unk_token_id]
    if len(valid_ids) != len(protocol_tokens) or len(set(valid_ids)) != len(valid_ids):
        raise Stage2DataError("not every protocol token received a unique tokenizer ID")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        trainable_token_indices={"embed_tokens": valid_ids},
    )
    model = get_peft_model(model, lora)
    records = frame.sort_values("traj_id").to_dict("records")
    tokenized = [tokenize_training_record(tokenizer, record) for record in records]
    collator = AssistantOnlyCollator(tokenizer)
    arguments = build_stage2_training_arguments(
        output / "trainer",
        seed,
        epochs,
        max_steps,
        batch_size,
        gradient_accumulation,
        learning_rate,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=_TokenizedDataset(tokenized),
        data_collator=collator,
    )
    train_result = trainer.train()
    adapter_path = output / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    adapter_storage = compact_peft_adapter(
        adapter_path,
        base_model_path=model_path,
        training_seed=seed,
    )
    trainer.save_state()
    train_metrics = {key: float(value) for key, value in train_result.metrics.items()}
    teacher_forced = None
    if eval_frame is not None:
        eval_records = eval_frame.sort_values("traj_id").to_dict("records")
        eval_tokenized = [tokenize_training_record(tokenizer, record) for record in eval_records]
        token_groups = {
            "sampling": set(tokenizer.convert_tokens_to_ids([f"<t_{value}>" for value in range(8)])),
        }
        for prefix, size in zip(PREFIXES, VOCAB_SIZES):
            token_groups[f"layer_{PREFIXES.index(prefix)}"] = set(
                tokenizer.convert_tokens_to_ids([f"<{prefix}_{value}>" for value in range(size)])
            )
        teacher_forced = evaluate_teacher_forced_model(
            model,
            _TokenizedDataset(eval_tokenized),
            collator,
            token_groups,
            next(model.parameters()).device,
        )
    metrics = {"train": train_metrics, "teacher_forced": teacher_forced}
    token_budget = {
        "population": len(tokenized),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in tokenized),
        "answer_tokens": sum(int(row["answer_tokens"]) for row in tokenized),
        "added_special_tokens": int(added),
        "protocol_token_count": len(protocol_tokens),
    }
    _atomic_json(output / "stage2_plan.json", plan)
    _atomic_json(output / "stage2_metrics.json", metrics)
    _atomic_json(output / "token_budget.json", token_budget)
    return {
        "plan": plan,
        "metrics": metrics,
        "token_budget": token_budget,
        "adapter": str(adapter_path),
        "adapter_storage": adapter_storage,
        "trainer_state": str(output / "trainer" / "trainer_state.json"),
    }
