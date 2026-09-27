"""eCeLLM-S comparator, GPU version.

Independently verified via HfApi.model_info: 2,779,683,840 parameters (2.78B),
F16, PhiForCausalLM (Phi-2 base) -- NOT the 7.24B Mistral-7B figure an earlier
pass in this project incorrectly reported without checking the Hub API
directly. At 2.78B this is comfortably GPU-feasible; the earlier CPU attempt
was abandoned based on the wrong (10x too large) parameter count.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "NingLab/eCeLLM-S"

RELEVANCE_PROMPT = """Classify the relationship between this shopper query and product into exactly one of: exact, substitute, complement, irrelevant.

Query: {query}
Product: {product_title}

Answer with exactly one word: exact, substitute, complement, or irrelevant."""

IDENTITY_PROMPT = """Are these two product listings describing the SAME purchasable item/variant, or DISTINCT products?

Listing A: {title_a}
Listing B: {title_b}

Answer with exactly one word: same, distinct, or unknown."""


def normalize(raw: str, labels: list[str]) -> str:
    raw = raw.strip().lower()
    for label in labels:
        if label in raw:
            return label
    return "invalid_output"


def main():
    print("loading eCeLLM-S...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    relevance_examples = json.loads(Path("data/frontier_eval_relevance.json").read_text())
    identity_examples = json.loads(Path("data/frontier_eval_identity.json").read_text())

    def generate(prompt: str, max_new_tokens: int = 8) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    rel_preds = []
    for i, ex in enumerate(relevance_examples):
        prompt = RELEVANCE_PROMPT.format(query=ex["query"], product_title=ex["product_title"])
        raw = generate(prompt)
        rel_preds.append(normalize(raw, ["exact", "substitute", "complement", "irrelevant"]))
        print(f"relevance {i+1}/{len(relevance_examples)}", flush=True)
    rel_correct = sum(1 for p, ex in zip(rel_preds, relevance_examples) if p == ex["gold"])
    rel_acc = rel_correct / len(relevance_examples)

    id_preds = []
    for i, ex in enumerate(identity_examples):
        prompt = IDENTITY_PROMPT.format(title_a=ex["title_a"], title_b=ex["title_b"])
        raw = generate(prompt)
        id_preds.append(normalize(raw, ["same", "distinct", "unknown"]))
        print(f"identity {i+1}/{len(identity_examples)}", flush=True)
    id_correct = sum(1 for p, ex in zip(id_preds, identity_examples) if p == ex["gold"])
    id_acc = id_correct / len(identity_examples)

    result = {
        "model": MODEL_ID,
        "comparison_status": "RUN",
        "relevance_accuracy": rel_acc,
        "identity_accuracy": id_acc,
        "n_relevance": len(relevance_examples),
        "n_identity": len(identity_examples),
    }
    print(json.dumps(result, indent=2))
    Path("ecellm_gpu_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
