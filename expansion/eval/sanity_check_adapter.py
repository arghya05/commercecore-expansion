"""Real local sanity check: load the trained adapter and run a handful of
real held-out examples through it, checking output is sensible, not garbage.
Not a substitute for the full dev-set score -- a smoke test before trusting it.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_PATH = "models/shared_adapter_v1"
BASE_MODEL = "Qwen/Qwen3-1.7B"

TASK_PROMPT_TEMPLATES = {
    "match_relevance": "Classify the query-product relevance: exact, substitute, complement, or irrelevant.\n{input}\nAnswer:",
    "match_identity": "Are these listings the same purchasable item or distinct?\n{input}\nAnswer:",
}


def load_adapter():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 6) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    model, tokenizer = load_adapter()

    relevance_examples = json.loads(Path("data/frontier_eval_relevance.json").read_text())[:8]
    identity_examples = json.loads(Path("data/frontier_eval_identity.json").read_text())[:8]

    print("=== Relevance sanity check ===")
    correct = 0
    for ex in relevance_examples:
        prompt = TASK_PROMPT_TEMPLATES["match_relevance"].format(
            input=f"Query: {ex['query']}\nProduct: {ex['product_title']}"
        )
        pred = generate(model, tokenizer, prompt)
        is_correct = ex["gold"] in pred.lower()
        correct += is_correct
        print(f"gold={ex['gold']:12s} pred={pred!r:30s} {'OK' if is_correct else 'WRONG'}")
    print(f"relevance sample accuracy: {correct}/{len(relevance_examples)}")

    print("\n=== Identity sanity check ===")
    correct = 0
    for ex in identity_examples:
        prompt = TASK_PROMPT_TEMPLATES["match_identity"].format(
            input=f"Listing A: {ex['title_a']}\nListing B: {ex['title_b']}"
        )
        pred = generate(model, tokenizer, prompt)
        is_correct = ex["gold"] in pred.lower()
        correct += is_correct
        print(f"gold={ex['gold']:10s} pred={pred!r:30s} {'OK' if is_correct else 'WRONG'}")
    print(f"identity sample accuracy: {correct}/{len(identity_examples)}")


if __name__ == "__main__":
    main()
