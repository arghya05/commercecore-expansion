"""Batched final dev-set evaluation. Replaces the unbatched per-example
generate() loop in train_shared_adapter.py's final eval, which was
prohibitively slow (~1.2s/example x 4,519 rows ~ 75+ minutes). Batches
same-length-padded prompts together for real GPU throughput.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/shared_adapter_v1"
BASE_MODEL = "Qwen/Qwen3-1.7B"
BATCH_SIZE = 32


def batched_generate(model, tokenizer, prompts: list[str], max_new_tokens: int = 6) -> list[str]:
    tokenizer.padding_side = "left"
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j in range(len(batch)):
            gen = out[j][inputs["input_ids"].shape[1]:]
            outputs.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
        print(f"  batch {i // BATCH_SIZE + 1}/{(len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE}", flush=True)
    return outputs


def main():
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    dev_rows = json.loads((Path(ADAPTER_PATH) / "dev_rows_for_eval.json").read_text())
    print(f"scoring {len(dev_rows)} dev rows", flush=True)

    prompts = [r["prompt"] for r in dev_rows]
    t1 = time.time()
    preds = batched_generate(model, tokenizer, prompts)
    print(f"generation took {time.time()-t1:.1f}s", flush=True)

    by_task_correct: dict[str, int] = {}
    by_task_total: dict[str, int] = {}
    for r, pred in zip(dev_rows, preds):
        correct = r["target"].strip().lower() in pred.lower()
        by_task_total[r["task"]] = by_task_total.get(r["task"], 0) + 1
        by_task_correct[r["task"]] = by_task_correct.get(r["task"], 0) + int(correct)

    scores = {task: by_task_correct[task] / by_task_total[task] for task in by_task_total}
    scores["_n_by_task"] = by_task_total
    print(json.dumps(scores, indent=2))
    Path(ADAPTER_PATH, "final_full_dev_scores_batched.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
