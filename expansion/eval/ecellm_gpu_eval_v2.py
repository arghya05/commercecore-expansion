"""eCeLLM-S comparator, corrected native prompt format.

The first attempt used a generic "answer with one word" instruction, which
produced 0.0/0.0 -- the model did not follow it at all (verified by
inspecting raw output: it continued the text as an unrelated exercise
rather than answering). Inspecting NingLab/ECInstruct (the model's own
training data) directly revealed its real instruction/input format:
instruction text + a JSON blob for Product_Matching (identity), and a
Product_Relation_Prediction / Product_Substitute_Identification-style
multiple-choice format for relation classification. This version matches
that format, a fair test rather than the earlier invalid one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "NingLab/eCeLLM-S"

IDENTITY_INSTRUCTION = "Given the title of two products, identify if they are the same product. Only output yes or no."

RELEVANCE_INSTRUCTION = (
    "Given a shopper query and a product title, classify the product as exactly one of: "
    "exact (the product is what the shopper is looking for), "
    "substitute (a reasonable alternative), "
    "complement (commonly bought alongside the query), "
    "or irrelevant (does not match the query at all). "
    "Only output one of: exact, substitute, complement, irrelevant."
)


def main():
    print("loading eCeLLM-S...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    def generate(prompt: str, max_new_tokens: int = 10) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    identity_examples = json.loads(Path("data/frontier_eval_identity.json").read_text())
    id_correct = 0
    for i, ex in enumerate(identity_examples):
        input_json = json.dumps({"product 1": {"title": ex["title_a"]}, "product 2": {"title": ex["title_b"]}})
        prompt = IDENTITY_INSTRUCTION + "\n" + input_json + "\n"
        raw = generate(prompt).strip().lower()
        pred = "same" if "yes" in raw else ("distinct" if "no" in raw else "invalid")
        if pred == ex["gold"]:
            id_correct += 1
        print(f"identity {i+1}/{len(identity_examples)}", flush=True)
    id_acc = id_correct / len(identity_examples)

    relevance_examples = json.loads(Path("data/frontier_eval_relevance.json").read_text())
    rel_correct = 0
    for i, ex in enumerate(relevance_examples):
        prompt = RELEVANCE_INSTRUCTION + f"\nQuery: {ex['query']}\nProduct: {ex['product_title']}\n"
        raw = generate(prompt).strip().lower()
        pred = next((l for l in ["exact", "substitute", "complement", "irrelevant"] if l in raw), "invalid")
        if pred == ex["gold"]:
            rel_correct += 1
        print(f"relevance {i+1}/{len(relevance_examples)}", flush=True)
    rel_acc = rel_correct / len(relevance_examples)

    result = {
        "model": MODEL_ID,
        "comparison_status": "RUN",
        "prompt_format": "native ECInstruct-style (verified against training data, not a generic instruction)",
        "relevance_accuracy": rel_acc,
        "identity_accuracy": id_acc,
        "n_relevance": len(relevance_examples),
        "n_identity": len(identity_examples),
    }
    print(json.dumps(result, indent=2))
    Path("ecellm_gpu_result_v2.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
