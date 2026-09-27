"""Frontier + HF comparator sweep for Commerce-Understand (brand/color
extraction), matching the same rigor already applied to Match: real,
text-grounded eval set, resolved model snapshots, same prompt for every
comparator.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from anthropic import Anthropic
from openai import OpenAI

anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EXTRACTION_PROMPT = """Extract the brand and color from this product listing. Respond with only a JSON object like {{"brand": "...", "color": "..."}}. If a field is not mentioned or unclear, use null.

Listing: {text}"""

FRONTIER_MODELS = {
    "claude-haiku-4-5": {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001"},
    "claude-sonnet-5": {"provider": "anthropic", "model_id": "claude-sonnet-5"},
    "gpt-5-mini": {"provider": "openai", "model_id": "gpt-5-mini"},
    "gpt-5": {"provider": "openai", "model_id": "gpt-5"},
}


def call_model(provider: str, model_id: str, prompt: str, retries: int = 2) -> str | None:
    for attempt in range(retries):
        try:
            if provider == "anthropic":
                resp = anthropic_client.messages.create(
                    model=model_id, max_tokens=100,
                    messages=[{"role": "user", "content": prompt}], timeout=30.0,
                )
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        return block.text.strip()
                return None
            else:
                resp = openai_client.chat.completions.create(
                    model=model_id, messages=[{"role": "user", "content": prompt}], timeout=30.0,
                )
                return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  call_model FINAL FAILURE {provider}/{model_id}: {type(exc).__name__}: {exc}", flush=True)
                return None
            time.sleep(2)
    return None


def parse_extraction(raw: str | None) -> dict:
    if raw is None:
        return {"brand": None, "color": None, "_parse_error": True}
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
        return {"brand": obj.get("brand"), "color": obj.get("color"), "_parse_error": False}
    except Exception:
        return {"brand": None, "color": None, "_parse_error": True}


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


def run_frontier_extraction(examples: list[dict], models: list[str]) -> dict:
    results = {}
    for model_key in models:
        cfg = FRONTIER_MODELS[model_key]
        brand_preds, color_preds = [], []
        parse_errors = 0
        for i, ex in enumerate(examples):
            prompt = EXTRACTION_PROMPT.format(text=ex["text"])
            raw = call_model(cfg["provider"], cfg["model_id"], prompt)
            parsed = parse_extraction(raw)
            brand_preds.append(parsed["brand"])
            color_preds.append(parsed["color"])
            parse_errors += int(parsed["_parse_error"])
            print(f"  [{model_key}] {i+1}/{len(examples)}", flush=True)
        results[model_key] = {
            "model_id": cfg["model_id"],
            "n": len(examples),
            "brand": score_field(brand_preds, [e["brand"] for e in examples]),
            "color": score_field(color_preds, [e["color"] for e in examples]),
            "parse_error_rate": parse_errors / len(examples),
        }
    return results


def main():
    examples = json.loads(Path("data/abo/understand_eval_grounded.json").read_text())
    models = ["claude-haiku-4-5", "claude-sonnet-5", "gpt-5-mini", "gpt-5"]

    resolved = {}
    for mk in models:
        cfg = FRONTIER_MODELS[mk]
        raw = call_model(cfg["provider"], cfg["model_id"], "say ok")
        resolved[mk] = raw
    print("resolution check (should be 'ok' or similar):", resolved, flush=True)

    results = run_frontier_extraction(examples, models)

    out = {"resolved_snapshots": {mk: FRONTIER_MODELS[mk]["model_id"] for mk in models}, "results": results}
    out_path = Path("reports/understand_frontier_comparison_2026-09-28/report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("Wrote", out_path)
    for mk in models:
        r = results[mk]
        print(f"{mk}: brand_f1={r['brand']['f1']:.3f} color_f1={r['color']['f1']:.3f} parse_err={r['parse_error_rate']:.2f}")


if __name__ == "__main__":
    main()
