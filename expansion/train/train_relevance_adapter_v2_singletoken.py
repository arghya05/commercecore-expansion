"""Dedicated relevance adapter, attempt 3: single-token label reformulation.

Rationale (see README.md relevance history -- 2 prior attempts both
concluded FAILED via majority-class collapse, confirmed by direct
prediction inspection):
  - Attempt 1 (high LR, no class weighting): collapsed to always predicting
    "exact" (majority class) within 150 steps.
  - Attempt 2 (class-weighted cross-entropy via a vocab-sized per-token
    weight tensor, lower LR): did NOT collapse to majority class, but
    NEVER predicted "substitute" or "complement" even once across 20
    inspected real predictions. Those are also the only two labels whose
    surface form is 2+ subword tokens under the Qwen3 tokenizer
    ("exact"=1 token, "substitute"=2, "complement"=2, "irrelevant"=2) --
    though token count alone doesn't fully explain it, since
    "irrelevant" (2 tokens) DID get predicted sometimes.

This attempt removes the token-length variable entirely: each of the 4
classes is mapped to a single distinct token (letters A/B/C/D) for
training and generation, then mapped back to the real label names only
for scoring against the fixed frontier benchmark (Claude Sonnet 5:
0.583 accuracy on the same real dev protocol). If this still fails to
move relevance's accuracy, token length is ruled out as the cause.
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

# Single-token label mapping -- fixed, arbitrary, never re-derived from data.
LABEL_TO_TOKEN = {"exact": "A", "substitute": "B", "complement": "C", "irrelevant": "D"}
TOKEN_TO_LABEL = {v: k for k, v in LABEL_TO_TOKEN.items()}

PROMPT_TEMPLATE = (
    "Classify the query-product relevance. Respond with exactly one letter:\n"
    "A = exact match, B = substitute, C = complement, D = irrelevant.\n"
    "{input}\nAnswer:"
)


def load_relevance_rows(path: Path) -> list[dict]:
    rows = []
    for line in open(path):
        row = json.loads(line)
        rows.append({
            "prompt": PROMPT_TEMPLATE.format(input=row["input"]),
            "target": LABEL_TO_TOKEN[row["target"]],
            "real_label": row["target"],
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
    mixture_path: str = "data/relevance_rows_v3.jsonl",
    output_dir: str = "models/relevance_adapter_v2_singletoken",
    max_length: int = 256,
    max_steps: int = 1200,
    lora_rank: int = 16,
    save_steps: int | None = None,
):
    assert_base_model_safe(BASE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify each label token is really a single token under this tokenizer
    # before training on the assumption that it is.
    for label, tok_str in LABEL_TO_TOKEN.items():
        ids = tokenizer(tok_str, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"label token {tok_str!r} for {label!r} is not single-token: {ids}"
    print("verified all label tokens are single-token:", LABEL_TO_TOKEN)

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
    print(f"relevance single-token: train rows: {len(train_rows)}, dev rows: {len(dev_rows)}")

    class_counts = collections.Counter(r["real_label"] for r in train_rows)
    print("train class distribution:", dict(class_counts))
    total = sum(class_counts.values())
    class_weight = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    print("computed class weights:", class_weight)

    train_ds = build_dataset(train_rows, tokenizer, max_length)

    vocab_size = model.config.vocab_size
    token_weight_by_id = torch.ones(vocab_size, dtype=torch.float32)
    for cls, weight in class_weight.items():
        tok_str = LABEL_TO_TOKEN[cls]
        token_id = tokenizer(tok_str, add_special_tokens=False)["input_ids"][0]
        token_weight_by_id[token_id] = weight
    print("token weights applied at ids:", {
        cls: (tokenizer(LABEL_TO_TOKEN[cls], add_special_tokens=False)["input_ids"][0], w)
        for cls, w in class_weight.items()
    })

    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        warmup_steps=max(int(max_steps * 0.03), 1),
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=save_steps or max(max_steps // 8, 1),
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
    frontier_path = Path("reports/frontier_comparison_2026-09-27/report.json")
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text())
        for model_key, res in frontier.get("relevance", {}).items():
            comparator_scores.setdefault("match_relevance", {})[f"frontier_{model_key}"] = res["accuracy"]
    print("Loaded real comparator scores to beat:", json.dumps(comparator_scores, indent=2))

    import random
    rng = random.Random(42)
    checkpoint_dev_sample = rng.sample(dev_rows, min(150, len(dev_rows)))
    # CheckpointComparatorCallback compares dev PASS RATE (fraction correct)
    # against the frontier's pass rate -- this is valid regardless of label
    # representation (single letter vs. real word) since both are just
    # accuracy fractions over the same underlying examples.

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
    with open(Path(output_dir) / "label_token_map.json", "w") as f:
        json.dump({"label_to_token": LABEL_TO_TOKEN, "token_to_label": TOKEN_TO_LABEL}, f, indent=2)

    print("training complete, adapter saved to", output_dir)


if __name__ == "__main__":
    main()
