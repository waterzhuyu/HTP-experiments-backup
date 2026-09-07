from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evaluation_progress(
    *,
    status: str,
    started_at: str,
    batch_size: int,
    batches_total: int,
    batches_completed: int,
    population_total: int,
    population_completed: int,
    supervised_tokens: int,
    elapsed_seconds: float,
    error: Exception | None = None,
) -> dict[str, object]:
    rate = batches_completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = max(batches_total - batches_completed, 0)
    eta_seconds = remaining / rate if rate > 0 and remaining else 0.0 if not remaining else None
    now = _utc_now()
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "updated_at": now,
        "finished_at": now if status in {"completed", "failed"} else None,
        "batch_size": batch_size,
        "batches_total": batches_total,
        "batches_completed": batches_completed,
        "population_total": population_total,
        "population_completed": population_completed,
        "supervised_tokens": supervised_tokens,
        "elapsed_seconds": elapsed_seconds,
        "batches_per_second": rate,
        "eta_seconds": eta_seconds,
        "error": None,
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    return payload


class _TrainingProgressRecorder:
    """Persist optimizer-step progress without participating in training decisions."""

    def __init__(
        self,
        path: str | Path,
        steps_total: int,
        progress_interval: int = 100,
    ) -> None:
        if steps_total <= 0:
            raise Stage2DataError("training progress steps_total must be positive")
        if progress_interval <= 0:
            raise Stage2DataError("training progress_interval must be positive")
        self.path = Path(path)
        self.steps_total = int(steps_total)
        self.progress_interval = int(progress_interval)
        self.started_at = _utc_now()
        self.started = time.monotonic()
        self.steps_completed = 0
        self.epoch = 0.0
        self.last_metrics: dict[str, float] = {}
        self._write("running")

    @staticmethod
    def _finite_metrics(metrics: Mapping[str, object] | None) -> dict[str, float]:
        if metrics is None:
            return {}
        result: dict[str, float] = {}
        for name in ("loss", "grad_norm", "learning_rate"):
            value = metrics.get(name)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                result[name] = numeric
        return result

    def _payload(
        self,
        status: str,
        error: Exception | None = None,
    ) -> dict[str, object]:
        elapsed = max(time.monotonic() - self.started, 0.0)
        rate = self.steps_completed / elapsed if elapsed > 0 else 0.0
        remaining = max(self.steps_total - self.steps_completed, 0)
        if remaining == 0:
            eta_seconds: float | None = 0.0
        elif rate > 0:
            eta_seconds = remaining / rate
        else:
            eta_seconds = None
        now = _utc_now()
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": status,
            "started_at": self.started_at,
            "updated_at": now,
            "finished_at": now if status in {"completed", "failed"} else None,
            "steps_total": self.steps_total,
            "steps_completed": self.steps_completed,
            "epoch": self.epoch,
            "elapsed_seconds": elapsed,
            "steps_per_second": rate,
            "eta_seconds": eta_seconds,
            "last_metrics": dict(self.last_metrics),
            "error": None,
        }
        if error is not None:
            payload["error"] = {"type": type(error).__name__, "message": str(error)}
        return payload

    def _write(self, status: str, error: Exception | None = None) -> None:
        _atomic_json(self.path, self._payload(status, error))

    def update(
        self,
        steps_completed: int,
        epoch: float | None = None,
        metrics: Mapping[str, object] | None = None,
        force: bool = False,
    ) -> bool:
        step = int(steps_completed)
        if step < self.steps_completed:
            raise Stage2DataError("training progress cannot move backwards")
        self.steps_completed = min(step, self.steps_total)
        if epoch is not None:
            numeric_epoch = float(epoch)
            if math.isfinite(numeric_epoch):
                self.epoch = numeric_epoch
        self.last_metrics.update(self._finite_metrics(metrics))
        should_write = (
            force
            or self.steps_completed == self.steps_total
            or self.steps_completed % self.progress_interval == 0
        )
        if should_write:
            self._write("running")
        return should_write

    def complete(
        self,
        steps_completed: int,
        epoch: float | None = None,
        metrics: Mapping[str, object] | None = None,
    ) -> None:
        self.update(steps_completed, epoch, metrics, force=False)
        if self.steps_completed != self.steps_total:
            raise Stage2DataError(
                "training returned before the planned optimizer-step count was reached"
            )
        self._write("completed")

    def fail(self, error: Exception) -> None:
        self._write("failed", error)


def _training_progress_callback(recorder: _TrainingProgressRecorder) -> object:
    """Build the locked Transformers callback while keeping imports lazy."""
    from transformers import TrainerCallback

    class TrainingProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            recorder.update(state.global_step, state.epoch, force=True)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            recorder.update(state.global_step, state.epoch, logs)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            recorder.update(state.global_step, state.epoch)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            recorder.complete(state.global_step, state.epoch)
            return control

    return TrainingProgressCallback()


@torch.no_grad()
def evaluate_teacher_forced_model(
    model: torch.nn.Module,
    dataset: Dataset,
    collator: AssistantOnlyCollator,
    token_groups: dict[str, set[int]],
    device: torch.device,
    batch_size: int = 4,
    progress_path: str | Path | None = None,
    progress_interval: int = 100,
    runtime_guard=None,
) -> dict[str, object]:
    if batch_size <= 0:
        raise Stage2DataError("teacher-forced evaluation batch_size must be positive")
    if progress_interval <= 0:
        raise Stage2DataError("teacher-forced progress_interval must be positive")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    progress = Path(progress_path) if progress_path is not None else None
    started_at = _utc_now()
    started = time.monotonic()
    batches_total = len(loader)
    batches_completed = 0
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = {name: 0 for name in token_groups}
    denominators = {name: 0 for name in token_groups}
    group_loss_sums = {name: 0.0 for name in token_groups}
    first_errors: Counter[str] = Counter()
    exact = 0
    population = 0
    if progress is not None:
        _atomic_json(
            progress,
            _evaluation_progress(
                status="running",
                started_at=started_at,
                batch_size=batch_size,
                batches_total=batches_total,
                batches_completed=0,
                population_total=len(dataset),
                population_completed=0,
                supervised_tokens=0,
                elapsed_seconds=0.0,
            ),
        )
    try:
        for batch in loader:
            if runtime_guard is not None:
                runtime_guard()
            batch = {name: value.to(device) for name, value in batch.items()}
            output = model(**batch)
            shifted_logits = output.logits[:, :-1]
            shifted_labels = batch["labels"][:, 1:]
            tokens = int((shifted_labels != -100).sum())
            token_losses = F.cross_entropy(
                shifted_logits.float().reshape(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1), ignore_index=-100, reduction="none",
            ).reshape_as(shifted_labels)
            total_loss += float(token_losses.sum())
            for group_name, ids in token_groups.items():
                selected = torch.isin(shifted_labels, torch.tensor(sorted(ids), device=device))
                group_loss_sums[group_name] += float(token_losses[selected].sum())
            total_tokens += tokens
            grouped = grouped_teacher_forced_accuracy(
                shifted_logits.argmax(dim=-1).cpu(), shifted_labels.cpu(), token_groups
            )
            population += int(grouped["population"])
            exact += round(
                float(grouped["sequence_exact_accuracy"]) * int(grouped["population"])
            )
            first_errors.update(grouped["first_error_histogram"])
            for name in token_groups:
                count = int(grouped["denominators"][name])
                denominators[name] += count
                if count:
                    correct[name] += round(float(grouped["accuracy"][name]) * count)
            batches_completed += 1
            if progress is not None and (
                batches_completed % progress_interval == 0 or batches_completed == batches_total
            ):
                _atomic_json(
                    progress,
                    _evaluation_progress(
                        status="running",
                        started_at=started_at,
                        batch_size=batch_size,
                        batches_total=batches_total,
                        batches_completed=batches_completed,
                        population_total=len(dataset),
                        population_completed=population,
                        supervised_tokens=total_tokens,
                        elapsed_seconds=time.monotonic() - started,
                    ),
                )
        if total_tokens == 0:
            raise Stage2DataError("teacher-forced evaluation produced no supervised tokens")
        nll = total_loss / total_tokens
        result = {
            "population": population,
            "tokens": total_tokens,
            "nll": nll,
            "perplexity": math.exp(nll),
            "accuracy": {
                name: correct[name] / denominators[name]
                if denominators[name]
                else float("nan")
                for name in token_groups
            },
            "denominators": denominators,
            "group_nll": {name: group_loss_sums[name] / denominators[name]
                          if denominators[name] else None for name in token_groups},
            "sid_nll": (sum(v for name, v in group_loss_sums.items() if name.startswith("layer_"))
                        / max(1, sum(v for name, v in denominators.items() if name.startswith("layer_")))),
            "sequence_exact_accuracy": exact / population,
            "first_error_histogram": dict(first_errors),
        }
        if progress is not None:
            _atomic_json(
                progress,
                _evaluation_progress(
                    status="completed",
                    started_at=started_at,
                    batch_size=batch_size,
                    batches_total=batches_total,
                    batches_completed=batches_completed,
                    population_total=len(dataset),
                    population_completed=population,
                    supervised_tokens=total_tokens,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
        return result
    except Exception as error:
        if progress is not None:
            _atomic_json(
                progress,
                _evaluation_progress(
                    status="failed",
                    started_at=started_at,
                    batch_size=batch_size,
                    batches_total=batches_total,
                    batches_completed=batches_completed,
                    population_total=len(dataset),
                    population_completed=population,
                    supervised_tokens=total_tokens,
                    elapsed_seconds=time.monotonic() - started,
                    error=error,
                ),
            )
        raise


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
    runtime_guard=None,
    extra_callbacks=None,
) -> dict[str, object]:
    """Train the frozen Stage2 recipe; heavyweight libraries are imported only here."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        set_seed,
    )

    set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = build_stage2_plan(frame, epochs, max_steps, batch_size, gradient_accumulation)
    _atomic_json(output / "stage2_plan.json", plan)
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
    training_progress = _TrainingProgressRecorder(
        output / "training_progress.json",
        int(plan["planned_optimizer_steps"]),
    )

    try:
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=_TokenizedDataset(tokenized),
            data_collator=collator,
            callbacks=[_training_progress_callback(training_progress)] + list(extra_callbacks or []),
        )
        train_result = trainer.train()
        training_progress.complete(
            trainer.state.global_step,
            trainer.state.epoch,
            train_result.metrics,
        )
    except Exception as error:
        training_progress.fail(error)
        raise
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
            progress_path=output / "evaluation_progress.json",
            runtime_guard=runtime_guard,
        )
    metrics = {"train": train_metrics, "teacher_forced": teacher_forced}
    token_budget = {
        "population": len(tokenized),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in tokenized),
        "answer_tokens": sum(int(row["answer_tokens"]) for row in tokenized),
        "added_special_tokens": int(added),
        "protocol_token_count": len(protocol_tokens),
    }
    _atomic_json(output / "stage2_metrics.json", metrics)
    _atomic_json(output / "token_budget.json", token_budget)
    return {
        "plan": plan,
        "metrics": metrics,
        "token_budget": token_budget,
        "adapter": str(adapter_path),
        "adapter_storage": adapter_storage,
        "training_progress": str(output / "training_progress.json"),
        "evaluation_progress": (
            str(output / "evaluation_progress.json") if eval_frame is not None else None
        ),
        "trainer_state": str(output / "trainer" / "trainer_state.json"),
    }
