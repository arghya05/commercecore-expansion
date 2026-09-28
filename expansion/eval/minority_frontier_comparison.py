"""Frontier-model comparison for match_functional_relation and
match_technical_compatibility -- these two tasks were never benchmarked
against frontier models (frontier_comparison.py only covers relevance and
identity). This closes that gap, mirroring the same protocol: same
prompt, same currently-callable snapshots, same normalize-first-match
scoring.
"""
from __future__ import annotations

import json
from pathlib import Path

from expansion.eval.frontier_comparison import FRONTIER_MODELS, call_model

FUNCTIONAL_PROMPT = """Classify the functional relationship between these two product listings into exactly one of: substitute, complement, unrelated.

substitute: a reasonable alternative to one another for the same purpose
complement: commonly bought and used together (one supports or is used alongside the other)
unrelated: no meaningful functional relationship

Listing A: {item_a}
Listing B: {item_b}

Answer with exactly one word: substitute, complement, or unrelated."""

COMPAT_PROMPT = """Based on this product listing/support text, is the described item COMPATIBLE or INCOMPATIBLE with the referenced device/model?

Text: {text}

Answer with exactly one word: compatible or incompatible."""


def normalize_functional(raw: str | None) -> str:
    if raw is None:
        return "call_error"
    raw = raw.strip().lower().strip(".")
    for label in ["substitute", "complement", "unrelated"]:
        if label in raw:
            return label
    return "invalid_output"


def normalize_compat(raw: str | None) -> str:
    if raw is None:
        return "call_error"
    raw = raw.strip().lower().strip(".")
    # check incompatible first -- "compatible" is a substring of "incompatible"
    if "incompatible" in raw:
        return "incompatible"
    if "compatible" in raw:
        return "compatible"
    return "invalid_output"


def run_functional_eval(examples: list[dict], models: list[str]) -> dict:
    results = {}
    for model_key in models:
        cfg = FRONTIER_MODELS[model_key]
        preds = []
        for i, ex in enumerate(examples):
            lines = ex["generated_text"].split("\n")
            item_a = lines[0] if lines else ex["generated_text"]
            item_b = lines[1] if len(lines) > 1 else ""
            prompt = FUNCTIONAL_PROMPT.format(item_a=item_a, item_b=item_b)
            raw = call_model(cfg["provider"], cfg["model_id"], prompt)
            preds.append(normalize_functional(raw))
            print(f"  [{model_key}] functional_relation {i+1}/{len(examples)}", flush=True)
        correct = sum(1 for p, ex in zip(preds, examples) if p == ex["label"])
        invalid = sum(1 for p in preds if p in ("invalid_output", "call_error"))
        results[model_key] = {
            "model_id": cfg["model_id"], "n": len(examples),
            "accuracy": correct / len(examples), "invalid_or_error_rate": invalid / len(examples),
            "predictions": preds, "gold": [ex["label"] for ex in examples],
        }
    return results


def run_compat_eval(examples: list[dict], models: list[str]) -> dict:
    results = {}
    for model_key in models:
        cfg = FRONTIER_MODELS[model_key]
        preds = []
        for i, ex in enumerate(examples):
            prompt = COMPAT_PROMPT.format(text=ex["generated_text"])
            raw = call_model(cfg["provider"], cfg["model_id"], prompt)
            preds.append(normalize_compat(raw))
            print(f"  [{model_key}] technical_compatibility {i+1}/{len(examples)}", flush=True)
        correct = sum(1 for p, ex in zip(preds, examples) if p == ex["label"])
        invalid = sum(1 for p in preds if p in ("invalid_output", "call_error"))
        results[model_key] = {
            "model_id": cfg["model_id"], "n": len(examples),
            "accuracy": correct / len(examples), "invalid_or_error_rate": invalid / len(examples),
            "predictions": preds, "gold": [ex["label"] for ex in examples],
        }
    return results


def main():
    fr_rows = [json.loads(l) for l in open("data/synthetic_match/functional_relation.jsonl")]
    fr_dev = [r for r in fr_rows if r["split"] == "dev"]
    compat_rows = [json.loads(l) for l in open("data/synthetic_match/technical_compatibility.jsonl")]
    compat_dev = [r for r in compat_rows if r["split"] == "dev"]

    print(f"functional_relation dev: {len(fr_dev)}, technical_compatibility dev: {len(compat_dev)}")

    models = list(FRONTIER_MODELS.keys())
    functional_results = run_functional_eval(fr_dev, models)
    compat_results = run_compat_eval(compat_dev, models)

    out = {
        "functional_relation": functional_results,
        "technical_compatibility": compat_results,
        "n_functional": len(fr_dev),
        "n_compat": len(compat_dev),
    }
    out_dir = Path("reports/minority_frontier_comparison_2026-09-28")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {m: v["accuracy"] for m, v in out[k].items()} for k in ["functional_relation", "technical_compatibility"]}, indent=2))


if __name__ == "__main__":
    main()
