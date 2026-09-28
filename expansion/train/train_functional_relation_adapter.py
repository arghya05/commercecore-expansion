"""Dedicated, non-shared functional_relation adapter.

Rationale: v1's shared adapter reported 1.0 accuracy on functional_relation,
but that number was measured against a dev split bug -- all 15 original dev
rows were labeled "complement" (a stratification bug in the original
synthetic-data split assignment, not something either training run caused).
After fixing the split to include all three classes, v1's REAL accuracy on
a corrected 13-example dev set is 0.692 (9/13), while every tested frontier
model (Claude Haiku/Sonnet, GPT-5-mini, GPT-5) scores 0.846 (11/13) on the
same corrected set -- a real, previously-hidden 15.4-point gap.

This trains functional_relation as its own dedicated adapter (same
approach that worked for the Understand brand/color adapter), on an
expanded, independently re-audited dataset: the original 65 audited rows
plus 122 new audit-supported rows from 48 new, genuinely distinct item-pair
scenarios (not just more paraphrases of the original 15) -- 187 rows total,
stratified train/dev/locked_test by label.

Does NOT touch technical_compatibility -- that task already ties every
frontier model at 1.0 on its own corrected dev set and needs no further
work. v1's shared adapter remains the model of record for
relevance/identity/technical_compatibility; this adapter only replaces
functional_relation.
"""
from __future__ import annotations

import collections
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
    assert_base_model_safe,
    CheckpointComparatorCallback,
)

PROMPT_TEMPLATE = (
    "Classify the functional relation: substitute, complement, or unrelated.\n{input}\nAnswer:"
)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in open(path):
        row = json.loads(line)
        rows.append({
            "prompt": PROMPT_TEMPLATE.format(input=row["generated_text"]),
            "target": row["label"],
            "task": "match_functional_relation",
            "split": row["split"],
        })
    return rows


def tokenize_for_causal_lm(tokenizer, examples: list[dict], max_length: int = 320):
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


class ClassWeightedTrainer(Trainer):
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


def main(
    data_path: str = "data/synthetic_match/functional_relation_v3_final.jsonl",
    output_dir: str = "models/functional_relation_adapter_v1",
    max_length: int = 320,
    max_steps: int = 600,
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

    all_rows = load_rows(Path(data_path))
    train_rows = [r for r in all_rows if r["split"] == "train"]
    dev_rows = [r for r in all_rows if r["split"] == "dev"]
    print(f"functional_relation: train rows: {len(train_rows)}, dev rows: {len(dev_rows)}")

    class_counts = collections.Counter(r["target"] for r in train_rows)
    print("train class distribution:", dict(class_counts))
    total = sum(class_counts.values())
    class_weight = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    print("computed class weights:", class_weight)

    train_ds = build_dataset(train_rows, tokenizer, max_length)

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
        learning_rate=2e-5,  # lower than relevance's 3e-5 -- this dataset is much smaller (131 train rows), higher LR risks overfitting/collapse faster
        warmup_steps=max(int(max_steps * 0.03), 1),
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=save_steps or max(max_steps // 12, 1),
        save_total_limit=None,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
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

    comparator_scores = {}
    frontier_path = Path("reports/minority_frontier_comparison_2026-09-28/report.json")
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text())
        for model_key, res in frontier.get("functional_relation", {}).items():
            comparator_scores.setdefault("match_functional_relation", {})[f"frontier_{model_key}"] = res["accuracy"]
    print("Loaded real comparator scores to beat:", json.dumps(comparator_scores, indent=2))

    callback = CheckpointComparatorCallback(
        tokenizer=tokenizer, dev_examples=dev_rows, output_dir=output_dir,
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


if __name__ == "__main__":
    main()
