"""Final locked-set evaluation for the dedicated Understand adapter, using
the exact same 200-example set and scoring method as all prior
baseline/frontier measurements (see understand_frontier_comparison.py's
score_field), so results are directly comparable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_PATH = "models/understand_adapter_v1"
BASE_MODEL = "Qwen/Qwen3-1.7B"

PROMPT_TEMPLATE = (
    'Extract the brand and color from this product listing. '
    'Respond with only a JSON object like {{"brand": "...", "color": "..."}}.\n'
    "Listing: {text}\nAnswer:"
)


def score_field(preds: list, gold: list) -> dict:
    tp = fp = fn = 0
    for p, g in zip(preds, gold):
        g_norm = (g or "").strip().lower()
        p_norm = (p or "").strip().lower() if p else ""
        if not p_norm:
            fn += 1
            continue
        if p_norm and (p_norm in g_norm or g_norm in p_norm):
            tp += 1
        else:
            fp += 1
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    examples = json.loads(Path("understand_eval_grounded.json").read_text())
    print(f"scoring {len(examples)} locked examples", flush=True)

    brand_preds, color_preds = [], []
    parse_errors = 0
    for i, ex in enumerate(examples):
        prompt = PROMPT_TEMPLATE.format(text=ex["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            obj = json.loads(raw)
            brand_preds.append(obj.get("brand"))
            color_preds.append(obj.get("color"))
        except Exception:
            brand_preds.append(None)
            color_preds.append(None)
            parse_errors += 1
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(examples)}", flush=True)

    result = {
        "n": len(examples),
        "brand": score_field(brand_preds, [e["brand"] for e in examples]),
        "color": score_field(color_preds, [e["color"] for e in examples]),
        "parse_error_rate": parse_errors / len(examples),
    }
    print(json.dumps(result, indent=2))
    Path("understand_locked_eval_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
