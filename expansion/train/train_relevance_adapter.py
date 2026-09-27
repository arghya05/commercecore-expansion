"""Dedicated, non-shared relevance adapter.

Rationale (see README.md v1/v2/v3 history): v1's shared multitask adapter
plateaued at 0.50-0.54 relevance accuracy regardless of data volume (v3: 5x
data, 2x rank, 3x steps) or class balance (v2: balanced oversampling made it
worse). The plateau pattern -- fast rise then hard flatten, unmoved by more
data/capacity/steps -- suggests capacity competition from the other three
subtasks in the shared adapter, not a data or class-balance problem.

This trains relevance ALONE: same base, own adapter, no competing objectives
from match_identity/match_functional_relation/match_technical_compatibility.
Those three subtasks are NOT part of this project's decision to abandon --
v1's shared adapter already handles them well (1.0/1.0/0.914 on identity+
minority tasks) and remains the model of record for them. This experiment
tests one specific, previously untested hypothesis for relevance alone.
"""
from __future__ import annotations

import json
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
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from expansion.train.train_shared_adapter import (
    BASE_MODEL,
    FORBIDDEN_MODEL_IDS,
    assert_base_model_safe,
    score_dev_accuracy,
    CheckpointComparatorCallback,
)

RELEVANCE_PROMPT_TEMPLATE = (
    "Classify the query-product relevance: exact, substitute, complement, or irrelevant.\n{input}\nAnswer:"
)


def load_relevance_rows(path: Path) -> list[dict]:
    rows = []
    for line in open(path):
        row = json.loads(line)
        rows.append({
            "prompt": RELEVANCE_PROMPT_TEMPLATE.format(input=row["input"]),
            "target": row["target"],
            "task": "match_relevance",
            "split": row["split"],
        })
    return rows


def tokenize_for_causal_lm(tokenizer, examples: list[dict], max_length: int = 256):
    input_ids_list, labels_list, attn_list = [], [], []
    for ex in examples:
        prompt_ids = tokenizer(ex["prompt"] + " ", add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(ex["target"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attn_list.append([1] * len(input_ids))
    return input_ids_list, labels_list, attn_list


def build_dataset(rows, tokenizer, max_length):
    input_ids, labels, attn = tokenize_for_causal_lm(tokenizer, rows, max_length=max_length)
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels, "attention_mask": attn})


def main(
    mixture_path: str = "relevance_rows_v3.jsonl",
    output_dir: str = "models/relevance_adapter_v1",
    max_length: int = 256,
    max_steps: int = 1200,
    lora_rank: int = 16,
    save_steps: int | None = None,
):
    assert_base_model_safe(BASE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    all_rows = load_relevance_rows(Path(mixture_path))
    train_rows = [r for r in all_rows if r["split"] == "train"]
    dev_rows = [r for r in all_rows if r["split"] == "dev"]
    print(f"relevance-only: train rows: {len(train_rows)}, dev rows: {len(dev_rows)}")

    # The first attempt at this run collapsed to always predicting the
    # majority class ("exact", ~43% of natural-distribution data) within the
    # first 150 steps and never recovered -- confirmed by direct inspection
    # of real predictions, not inferred from the aggregate accuracy alone.
    # Root cause: natural class imbalance plus an unconstrained single-task
    # objective and no class weighting let the model find that degenerate,
    # low-loss shortcut immediately. Fix: per-class-weighted loss (not
    # duplicate-row oversampling -- that was v2's already-rejected approach,
    # which made the SHARED adapter's relevance worse via different means).
    import collections
    class_counts = collections.Counter(r["target"] for r in train_rows)
    print("train class distribution:", dict(class_counts))
    total = sum(class_counts.values())
    class_weight = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    print("computed class weights:", class_weight)

    train_ds = build_dataset(train_rows, tokenizer, max_length)

    # Build a vocab-sized per-token weight tensor: only the token(s) that
    # begin each class label get a non-1.0 weight. Every other token
    # (prompt tokens are already masked to -100 and excluded; this only
    # affects supervised completion tokens) stays at weight 1.0.
    # NOTE: len(tokenizer) (151,669) does NOT match the model's actual
    # output vocab / lm_head size (151,936, includes reserved/padding
    # tokens) -- caught by a real crash on the first attempt. Must read
    # vocab_size from the model's own config, not the tokenizer.
    vocab_size = model.config.vocab_size
    token_weight_by_id = torch.ones(vocab_size, dtype=torch.float32)
    for cls, weight in class_weight.items():
        first_token_id = tokenizer(cls, add_special_tokens=False)["input_ids"][0]
        token_weight_by_id[first_token_id] = weight
    print("token weights applied at first-token ids:", {
        cls: (tokenizer(cls, add_special_tokens=False)["input_ids"][0], w)
        for cls, w in class_weight.items()
    })

    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,  # lowered from 1e-4 -- the higher LR let the model collapse to majority-class within 150 steps
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=save_steps or max(max_steps // 8, 1),
        save_total_limit=None,  # keep every checkpoint -- do not prune; v3 lost its best checkpoint (step 300) to this limit and it could not be recovered
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

    class ClassWeightedTrainer(Trainer):
        """Per-TOKEN class weighting via a fixed vocab-sized weight tensor
        passed to F.cross_entropy's native `weight` argument -- the standard,
        fused PyTorch path, not a manual per-example reimplementation (which
        two prior attempts in this run showed was ~50-100x slower for reasons
        not fully isolated; using the built-in weighted-CE path sidesteps
        that investigation entirely rather than chasing it further)."""

        def __init__(self, *args, token_weight_by_id: torch.Tensor, **kwargs):
            super().__init__(*args, **kwargs)
            self.token_weight_by_id = token_weight_by_id

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            weight = self.token_weight_by_id.to(device=shift_logits.device, dtype=shift_logits.dtype)
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                weight=weight,
                ignore_index=-100,
            )
            return (loss, outputs) if return_outputs else loss

    comparator_scores = {}
    frontier_path = Path("reports/frontier_comparison_2026-09-27/report.json")
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text())
        for model_key, res in frontier.get("relevance", {}).items():
            comparator_scores.setdefault("match_relevance", {})[f"frontier_{model_key}"] = res["accuracy"]
    ecellm_path = Path("reports/ecellm_result.json")
    if ecellm_path.exists():
        ecellm = json.loads(ecellm_path.read_text())
        comparator_scores.setdefault("match_relevance", {})["ecellm_s"] = ecellm["relevance_accuracy"]
    print("Loaded real comparator scores to beat:", json.dumps(comparator_scores, indent=2))

    import random
    rng = random.Random(42)
    checkpoint_dev_sample = rng.sample(dev_rows, min(150, len(dev_rows)))

    callback = CheckpointComparatorCallback(
        tokenizer=tokenizer, dev_examples=checkpoint_dev_sample, output_dir=output_dir,
        comparator_scores=comparator_scores,
    )

    trainer = ClassWeightedTrainer(
        model=model, args=args, train_dataset=train_ds,
        data_collator=collate, callbacks=[callback],
        token_weight_by_id=token_weight_by_id,
    )
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(Path(output_dir) / "dev_rows_for_eval.json", "w") as f:
        json.dump(dev_rows, f)

    print("training complete, adapter saved to", output_dir)
    print("run expansion/eval/batched_dev_eval.py for the full dev-set score.")


if __name__ == "__main__":
    main()
