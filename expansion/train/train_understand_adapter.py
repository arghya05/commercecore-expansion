"""Dedicated Understand (brand/color extraction) adapter.

First trained model attempted for this task -- prior work only tested
off-the-shelf rules and GLiNER2/2.5 (all beaten by every frontier model,
see reports/understand_frontier_comparison_2026-09-28/report.json). This
tests whether fine-tuning closes that gap, using real ABO listings (9,396
train / 300 dev, all text-grounded: brand/color values verified to occur
in the source text, not copied from structured metadata). The original
200-example baseline/frontier eval set is held out untouched as the
final locked test -- never used here for training or dev selection.
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
    assert_base_model_safe,
    score_dev_accuracy,
    CheckpointComparatorCallback,
)

PROMPT_TEMPLATE = (
    'Extract the brand and color from this product listing. '
    'Respond with only a JSON object like {{"brand": "...", "color": "..."}}.\n'
    "Listing: {text}\nAnswer:"
)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in open(path):
        row = json.loads(line)
        target = json.dumps({"brand": row["brand"], "color": row["color"]})
        rows.append({
            "prompt": PROMPT_TEMPLATE.format(text=row["text"]),
            "target": target,
            "brand": row["brand"],
            "color": row["color"],
        })
    return rows


def tokenize_for_causal_lm(tokenizer, examples: list[dict], max_length: int):
    input_ids_list, labels_list = [], []
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
    return input_ids_list, labels_list


def build_dataset(rows, tokenizer, max_length):
    input_ids, labels = tokenize_for_causal_lm(tokenizer, rows, max_length)
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels})


def score_field_accuracy(model, tokenizer, dev_examples: list[dict], max_new_tokens: int = 40) -> dict:
    model.eval()
    brand_correct = color_correct = parse_ok = 0
    for ex in dev_examples:
        inputs = tokenizer(ex["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            obj = json.loads(raw)
            parse_ok += 1
            pred_brand = (obj.get("brand") or "").strip().lower()
            pred_color = (obj.get("color") or "").strip().lower()
            gold_brand = ex["brand"].strip().lower()
            gold_color = ex["color"].strip().lower()
            if pred_brand and (pred_brand in gold_brand or gold_brand in pred_brand):
                brand_correct += 1
            if pred_color and (pred_color in gold_color or gold_color in pred_color):
                color_correct += 1
        except Exception:
            pass
    model.train()
    n = len(dev_examples)
    return {
        "match_understand_brand": brand_correct / n,
        "match_understand_color": color_correct / n,
        "parse_rate": parse_ok / n,
    }


def main(
    train_path: str = "understand_train.jsonl",
    output_dir: str = "models/understand_adapter_v1",
    max_length: int = 384,
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
    model.enable_input_require_grads()  # required for gradient checkpointing + PEFT to work together
    model.print_trainable_parameters()

    train_rows = load_rows(Path(train_path))
    print(f"understand: train rows: {len(train_rows)}")
    train_ds = build_dataset(train_rows, tokenizer, max_length)

    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=4,  # lowered from 8 -- OOM'd at step 13 on a 24GB card with max_length=384
        gradient_accumulation_steps=4,  # doubled to keep the same effective batch size (32)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=save_steps or max(max_steps // 8, 1),
        save_total_limit=None,
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
            attn[i, :n] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    dev_path = Path(train_path).parent / "understand_dev.jsonl"
    dev_rows = load_rows(dev_path) if dev_path.exists() else []  # must go through load_rows() for the "prompt" field
    import random
    rng = random.Random(42)
    checkpoint_dev_sample = rng.sample(dev_rows, min(100, len(dev_rows))) if dev_rows else []

    from transformers import TrainerCallback

    class UnderstandCallback(TrainerCallback):
        def __init__(self):
            self.history = []

        def on_save(self, args, state, control, model=None, **kwargs):
            if model is None or not checkpoint_dev_sample:
                return
            scores = score_field_accuracy(model, tokenizer, checkpoint_dev_sample)
            entry = {"step": state.global_step, **scores}
            self.history.append(entry)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "checkpoint_history.json").write_text(json.dumps(self.history, indent=2))
            print(f"[checkpoint step {state.global_step}] {scores}", flush=True)

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
        data_collator=collate, callbacks=[UnderstandCallback()],
    )
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("training complete, adapter saved to", output_dir)


if __name__ == "__main__":
    main()
