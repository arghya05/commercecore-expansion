"""Frontier-model comparison harness: Claude and OpenAI on the same matched
task/protocol as the trained model, per plans/00_expansion_core.md F03 —
"A model beats a baseline on a benchmark, not the benchmark itself."

Uses currently callable model snapshots, resolved and recorded at experiment
registration time (not a historical alias) per the plan's evaluation rules.
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

RELEVANCE_PROMPT = """Classify the relationship between this shopper query and product into exactly one of: exact, substitute, complement, irrelevant.

exact: the product is what the shopper is looking for
substitute: a reasonable alternative to what was searched for
complement: a product commonly bought alongside what was searched for
irrelevant: the product does not match the query's intent at all

Query: {query}
Product: {product_title}

Answer with exactly one word: exact, substitute, complement, or irrelevant."""

IDENTITY_PROMPT = """Are these two product listings describing the SAME purchasable item/variant, or DISTINCT products?

Listing A: {title_a}
Listing B: {title_b}

Answer with exactly one word: same, distinct, or unknown."""


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
                    model=model_id,
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30.0,
                )
                # Extended-thinking models (e.g. claude-sonnet-5) can return a
                # ThinkingBlock as content[0] with no .text attribute; find
                # the first actual TextBlock instead of assuming index 0.
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        return block.text.strip().lower()
                return None
            else:
                resp = openai_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30.0,
                )
                return resp.choices[0].message.content.strip().lower()
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  call_model FINAL FAILURE {provider}/{model_id}: {type(exc).__name__}: {exc}", flush=True)
                return None
            time.sleep(2)
    return None


def normalize_relevance(raw: str | None) -> str:
    if raw is None:
        return "call_error"
    raw = raw.strip().lower().strip(".")
    for label in ["exact", "substitute", "complement", "irrelevant"]:
        if label in raw:
            return label
    return "invalid_output"


def normalize_identity(raw: str | None) -> str:
    if raw is None:
        return "call_error"
    raw = raw.strip().lower().strip(".")
    for label in ["same", "distinct", "unknown"]:
        if label in raw:
            return label
    return "invalid_output"


def run_relevance_frontier_eval(examples: list[dict], models: list[str]) -> dict:
    """examples: list of {"query":..., "product_title":..., "gold": "exact"|...}"""
    results = {}
    for model_key in models:
        cfg = FRONTIER_MODELS[model_key]
        preds = []
        for i, ex in enumerate(examples):
            prompt = RELEVANCE_PROMPT.format(query=ex["query"], product_title=ex["product_title"])
            raw = call_model(cfg["provider"], cfg["model_id"], prompt)
            preds.append(normalize_relevance(raw))
            print(f"  [{model_key}] relevance {i+1}/{len(examples)}", flush=True)
        correct = sum(1 for p, ex in zip(preds, examples) if p == ex["gold"])
        invalid = sum(1 for p in preds if p in ("invalid_output", "call_error"))
        results[model_key] = {
            "model_id": cfg["model_id"],
            "n": len(examples),
            "accuracy": correct / len(examples),
            "invalid_or_error_rate": invalid / len(examples),
            "predictions": preds,
        }
    return results


def run_identity_frontier_eval(examples: list[dict], models: list[str]) -> dict:
    """examples: list of {"title_a":..., "title_b":..., "gold": "same"|"distinct"}"""
    results = {}
    for model_key in models:
        cfg = FRONTIER_MODELS[model_key]
        preds = []
        for i, ex in enumerate(examples):
            prompt = IDENTITY_PROMPT.format(title_a=ex["title_a"], title_b=ex["title_b"])
            raw = call_model(cfg["provider"], cfg["model_id"], prompt)
            preds.append(normalize_identity(raw))
            print(f"  [{model_key}] identity {i+1}/{len(examples)}", flush=True)
        correct = sum(1 for p, ex in zip(preds, examples) if p == ex["gold"])
        invalid = sum(1 for p in preds if p in ("invalid_output", "call_error"))
        results[model_key] = {
            "model_id": cfg["model_id"],
            "n": len(examples),
            "accuracy": correct / len(examples),
            "invalid_or_error_rate": invalid / len(examples),
            "predictions": preds,
        }
    return results
