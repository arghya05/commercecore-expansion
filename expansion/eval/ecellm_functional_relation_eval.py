"""eCeLLM-S comparator for match_functional_relation, on the exact same
28-row locked_test set used for the dedicated adapter's honest final
check. Same lesson as ecellm_gpu_eval_v2.py: a generic "answer with one
word" instruction gets ignored by this model -- use its own
multiple-choice instruction style (verified against NingLab/ECInstruct's
Product_Relation_Prediction/Product_Substitute_Identification tasks),
adapted to this project's own substitute/complement/unrelated labels
rather than eCeLLM's own label set (viewed-together/bought-together/
similar), since those are a different taxonomy and cannot be fairly
cross-mapped without inventing a mapping.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "NingLab/eCeLLM-S"

INSTRUCTION = (
    "You will see two product listings. Choose exactly one option that best describes "
    "their relationship:\n"
    "A: substitute (a reasonable alternative to one another for the same purpose)\n"
    "B: complement (commonly bought and used together)\n"
    "C: unrelated (no meaningful functional relationship)\n"
    "Only output the letter A, B, or C."
)

LETTER_TO_LABEL = {"a": "substitute", "b": "complement", "c": "unrelated"}


def main():
    print("loading eCeLLM-S...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    def generate(prompt: str, max_new_tokens: int = 6) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    rows = [json.loads(l) for l in open("data/synthetic_match/functional_relation_v3_final.jsonl")]
    locked = [r for r in rows if r["split"] == "locked_test"]
    print(f"locked_test n={len(locked)}", flush=True)

    correct = 0
    invalid = 0
    predictions = []
    for i, r in enumerate(locked):
        input_json = json.dumps({"Product 1:": r["generated_text"]})
        prompt = INSTRUCTION + "\n" + input_json + "\n"
        raw = generate(prompt).strip().lower()
        letter = next((c for c in raw if c in "abc"), None)
        pred = LETTER_TO_LABEL.get(letter, "invalid")
        if pred == "invalid":
            invalid += 1
        if pred == r["label"]:
            correct += 1
        predictions.append({"gold": r["label"], "pred": pred, "raw": raw})
        print(f"functional_relation {i+1}/{len(locked)}: gold={r['label']} pred={pred}", flush=True)

    accuracy = correct / len(locked)
    result = {
        "model": MODEL_ID,
        "task": "match_functional_relation",
        "comparison_status": "RUN",
        "prompt_format": "own instruction-style, project's own labels (no cross-taxonomy label mapping)",
        "accuracy": accuracy,
        "invalid_rate": invalid / len(locked),
        "n": len(locked),
        "predictions": predictions,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2))
    Path("ecellm_functional_relation_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
