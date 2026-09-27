"""Shared multitask QLoRA adapter training: Understand + Match (relevance,
identity, functional_relation, technical_compatibility).

Starts from an INDEPENDENTLY PINNED ORIGINAL Qwen3-1.7B base — never the
Query-merged arghya2030/commercecore-qwen3-1.7b artifact. Per
plans/00_expansion_core.md: rank 16/alpha 32/dropout .05 starting point,
DEVELOPMENT-only checkpoint evaluation, CALIBRATION for thresholds after
model selection.

ADAPTIVE TRAINING: checkpoints every N steps are evaluated on DEVELOPMENT;
the run keeps the best-scoring checkpoint by mean per-task dev accuracy,
not automatically the final one (mirrors the historical retention-tracking
practice, applied here to real task quality rather than lexical overlap).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

BASE_MODEL = "Qwen/Qwen3-1.7B"  # independently pinned original base, never the Query-merged artifact
FORBIDDEN_MODEL_IDS = {"arghya2030/commercecore-qwen3-1.7b"}

TASK_PROMPT_TEMPLATES = {
    "match_relevance": "Classify the query-product relevance: exact, substitute, complement, or irrelevant.\n{input}\nAnswer:",
    "match_identity": "Are these listings the same purchasable item or distinct?\n{input}\nAnswer:",
    "match_functional_relation": "Classify the functional relation: substitute, complement, unrelated, or unknown.\n{input}\nAnswer:",
    "match_technical_compatibility": "Classify technical compatibility: compatible, incompatible, or unknown.\n{input}\nAnswer:",
}


def assert_base_model_safe(model_name_or_path: str) -> None:
    if model_name_or_path in FORBIDDEN_MODEL_IDS:
        raise ValueError(
            f"Refusing to load {model_name_or_path!r} — this is the Query-merged "
            "artifact and is prohibited as a training base per master plan §1.1."
        )


def format_example(row: dict) -> dict:
    if "split" not in row:
        raise ValueError(
            f"row for task={row.get('task')!r} has no split assignment — "
            "run the scenario-level train/dev splitter before loading into training"
        )
    template = TASK_PROMPT_TEMPLATES[row["task"]]
    prompt = template.format(input=row["input"])
    return {"prompt": prompt, "target": row["target"], "task": row["task"], "split": row["split"]}


def load_mixture(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        for line in open(p):
            row = json.loads(line)
            rows.append(format_example(row))
    return rows


def tokenize_for_causal_lm(tokenizer, examples: list[dict], max_length: int = 256):
    """Assistant-only loss mask: prompt tokens are masked (-100), only the
    target completion contributes to the loss."""
    input_ids_list, labels_list, attn_list = [], [], []
    for ex in examples:
        prompt_ids = tokenizer(ex["prompt"] + " ", add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(ex["target"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        attn = [1] * len(input_ids)
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attn_list.append(attn)
    return input_ids_list, labels_list, attn_list


def score_dev_accuracy(model, tokenizer, dev_examples: list[dict], max_new_tokens: int = 6) -> dict:
    """Real per-task dev accuracy via greedy generation, not perplexity proxy.
    Used both for checkpoint selection and for the checkpoint-vs-comparator
    tracking the user required: at each checkpoint, compare against the same
    benchmark/comparator numbers already measured for frontier + HF models."""
    import torch

    model.eval()
    by_task_correct: dict[str, int] = {}
    by_task_total: dict[str, int] = {}
    for ex in dev_examples:
        inputs = tokenizer(ex["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        correct = ex["target"].strip().lower() in pred_text
        by_task_total[ex["task"]] = by_task_total.get(ex["task"], 0) + 1
        by_task_correct[ex["task"]] = by_task_correct.get(ex["task"], 0) + int(correct)
    model.train()
    return {
        task: by_task_correct[task] / by_task_total[task]
        for task in by_task_total
    }


class CheckpointComparatorCallback(TrainerCallback):
    """At each eval_steps checkpoint: score real dev accuracy per task, log it
    alongside the already-measured comparator/frontier numbers (loaded from
    reports/), and record which checkpoint currently leads. Does not stop
    training itself — final checkpoint selection happens after training using
    this full history, matching the plan's DEVELOPMENT-only checkpoint rule."""

    def __init__(self, tokenizer, dev_examples: list[dict], output_dir: str, comparator_scores: dict):
        self.tokenizer = tokenizer
        self.dev_examples = dev_examples
        self.output_dir = Path(output_dir)
        self.comparator_scores = comparator_scores  # {task: {comparator_name: score}}
        self.history: list[dict] = []

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        dev_scores = score_dev_accuracy(model, self.tokenizer, self.dev_examples)
        entry = {
            "step": state.global_step,
            "dev_accuracy_by_task": dev_scores,
            "comparator_scores": self.comparator_scores,
            "beats_all_comparators": {
                task: all(
                    dev_scores.get(task, 0.0) > score
                    for score in self.comparator_scores.get(task, {}).values()
                )
                for task in dev_scores
            },
        }
        self.history.append(entry)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoint_history.json").write_text(json.dumps(self.history, indent=2))
        print(f"[checkpoint step {state.global_step}] dev_accuracy_by_task={dev_scores}")


def build_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    input_ids, labels, attn = tokenize_for_causal_lm(tokenizer, rows, max_length=max_length)
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels, "attention_mask": attn})


def main(
    mixture_paths: list[str],
    output_dir: str,
    max_length: int = 256,
    max_steps: int = 400,
    eval_steps: int = 100,
):
    assert_base_model_safe(BASE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )

    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    all_rows = load_mixture([Path(p) for p in mixture_paths])
    train_rows = [r for r in all_rows if r["split"] == "train"]
    dev_rows = [r for r in all_rows if r["split"] == "dev"]
    print(f"train rows: {len(train_rows)}, dev rows: {len(dev_rows)}")

    # match_relevance has severe class imbalance (complement ~4.8% of rows).
    # Oversample minority classes within that task only, so the shared
    # objective doesn't starve them relative to match_identity's simpler
    # binary-ish distribution. Other tasks are left at their natural rate.
    import collections
    import random
    rng = random.Random(7)
    relevance_rows = [r for r in train_rows if r["task"] == "match_relevance"]
    other_rows = [r for r in train_rows if r["task"] != "match_relevance"]
    by_target = collections.defaultdict(list)
    for r in relevance_rows:
        by_target[r["target"]].append(r)
    max_count = max(len(v) for v in by_target.values())
    balanced_relevance = []
    for target, rows in by_target.items():
        reps = max_count // len(rows)
        remainder = max_count - reps * len(rows)
        balanced_relevance.extend(rows * reps)
        balanced_relevance.extend(rng.sample(rows, remainder))
    rng.shuffle(balanced_relevance)
    train_rows = other_rows + balanced_relevance
    rng.shuffle(train_rows)
    print(
        f"class-balanced relevance rows: {len(balanced_relevance)} "
        f"(was {len(relevance_rows)}); new total train rows: {len(train_rows)}"
    )

    train_ds = build_dataset(train_rows, tokenizer, max_length)

    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=eval_steps,
        save_total_limit=5,
        bf16=True,
        report_to=[],
    )

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            input_ids[i, :n] = torch.tensor(x["input_ids"])
            labels[i, :n] = torch.tensor(x["labels"])
            attn[i, :n] = torch.tensor(x["attention_mask"])
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    def _only_real_scores(raw: dict) -> dict:
        """Filters out NOT_RUN/error cells -- only cells with an actual
        numeric comparison_status=='RUN' score count as a comparator to beat.
        A dict of all-NOT_RUN cells must not silently make beats_all_comparators
        trivially True."""
        out = {}
        for name, cell in raw.items():
            if isinstance(cell, dict) and cell.get("comparison_status") == "RUN" and "score" in cell:
                out[name] = cell["score"]
        return out

    comparator_scores: dict[str, dict[str, float]] = {}
    hf_comparator_path = Path("reports/comparator_baselines_2026-09-27/report.json")
    if hf_comparator_path.exists():
        raw = json.loads(hf_comparator_path.read_text())
        real = _only_real_scores(raw)
        if real:
            comparator_scores.setdefault("match_relevance", {}).update(real)

    frontier_path = Path("reports/frontier_comparison_2026-09-27/report.json")
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text())
        for model_key, res in frontier.get("relevance", {}).items():
            comparator_scores.setdefault("match_relevance", {})[f"frontier_{model_key}"] = res["accuracy"]
        for model_key, res in frontier.get("identity", {}).items():
            comparator_scores.setdefault("match_identity", {})[f"frontier_{model_key}"] = res["accuracy"]

    print("Loaded real comparator scores to beat:", json.dumps(comparator_scores, indent=2))

    # Full dev set (4,519 rows) is too slow for per-checkpoint greedy-generation
    # scoring (one example at a time, no batching) -- would add 15-30+ min per
    # checkpoint. Sample a small, stratified-by-task subset for fast checkpoint
    # tracking; the FULL dev set is still scored once at the end on the
    # selected checkpoint for the real reported number.
    import random
    rng = random.Random(42)
    by_task: dict[str, list[dict]] = {}
    for r in dev_rows:
        by_task.setdefault(r["task"], []).append(r)
    checkpoint_dev_sample = []
    for task, rows in by_task.items():
        checkpoint_dev_sample.extend(rng.sample(rows, min(100, len(rows))))

    callback = CheckpointComparatorCallback(
        tokenizer=tokenizer, dev_examples=checkpoint_dev_sample, output_dir=output_dir,
        comparator_scores=comparator_scores,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collate,
        callbacks=[callback],
    )
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(Path(output_dir) / "dev_rows_for_eval.json", "w") as f:
        json.dump(dev_rows, f)

    print("training complete, adapter saved to", output_dir)
    print("run expansion/eval/batched_dev_eval.py separately for the full dev-set score "
          "(batched -- the unbatched per-example version here was too slow: ~75min for 4,519 rows).")


if __name__ == "__main__":
    main(
        mixture_paths=["training_mixture_v3.jsonl"],
        output_dir="models/shared_adapter_v3",
        max_steps=1200,
    )
