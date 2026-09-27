"""HF01: NingLab/eCeLLM-S natural-language prompting on the SAME relevance +
extraction eval sets used elsewhere in this pass.

Per RELATED_MODEL_COMPARISON_MATRIX.md HF01: "Compare each on compatible
extraction, relevance, matching, ranking and QA contracts." eCeLLM-S is a
2.78B-parameter (register-verified via hub_safetensors_parameter_count) Phi-2
derivative -- the smallest of the eCeLLM family. eCeLLM-M (7.24B) and
eCeLLM-L (13.0B) are NOT attempted in this pass: fp32 CPU inference at those
sizes (29GB and 52GB respectively) is infeasible on 8GB-RAM hardware, and is
recorded as NOT_RUN with that reason, not silently skipped.

eCeLLM-S at fp32 is ~11.1GB -- also likely infeasible on this 8GB-RAM
machine. We attempt fp16 (~5.6GB) first, matching the model's native "F16
tensor type" per its card, on a SMALL subset (n=20) given expected slow CPU
generation at this size. If loading OOMs or generation is impractically
slow, record NOT_RUN with the real error/timing, never a fabricated score.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from understand_baselines import score_field

RELEVANCE_PROMPT = """Classify the relationship between this shopper query and product into exactly one of: exact, substitute, complement, irrelevant.

exact: the product is what the shopper is looking for
substitute: a reasonable alternative to what was searched for
complement: a product commonly bought alongside what was searched for
irrelevant: the product does not match the query's intent at all

Query: {query}
Product: {product_title}

Answer with exactly one word: exact, substitute, complement, or irrelevant."""

EXTRACTION_PROMPT = """Extract the brand and color from this product description. Respond with a JSON object like {{"brand": "...", "color": "..."}}. If a field is not mentioned, use null.

Product description: {text}

JSON:"""


def normalize_relevance(raw: str) -> str:
    raw = raw.strip().lower().strip(".")
    for label in ["exact", "substitute", "complement", "irrelevant"]:
        if label in raw:
            return label
    return "invalid_output"


def run_ecellm_s(relevance_examples: list[dict], extraction_examples: list[dict],
                  n_relevance: int = 60, n_extraction: int = 30) -> dict:
    t0 = time.time()
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        repo = "NingLab/eCeLLM-S"
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        load_t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, dtype=torch.float16, device_map="cpu",
        ).eval()
        load_time = time.time() - load_t0

        def generate(prompt: str, max_new_tokens: int = 20) -> str:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return text

        # Relevance subset
        rel_subset = relevance_examples[:n_relevance]
        rel_preds = []
        gen_t0 = time.time()
        for ex in rel_subset:
            prompt = RELEVANCE_PROMPT.format(query=ex["query"], product_title=ex["product_title"])
            raw = generate(prompt, max_new_tokens=8)
            rel_preds.append(normalize_relevance(raw))
        rel_gen_time = time.time() - gen_t0
        rel_correct = sum(1 for p, ex in zip(rel_preds, rel_subset) if p == ex["gold"])

        # Extraction subset
        ext_subset = extraction_examples[:n_extraction]
        brand_preds, color_preds = [], []
        gen_t0 = time.time()
        for ex in ext_subset:
            prompt = EXTRACTION_PROMPT.format(text=ex["text"])
            raw = generate(prompt, max_new_tokens=64)
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                parsed = json.loads(raw[start:end])
                brand_preds.append(parsed.get("brand"))
                color_preds.append(parsed.get("color"))
            except Exception:
                brand_preds.append(None)
                color_preds.append(None)
        ext_gen_time = time.time() - gen_t0

        return {
            "comparison_status": "RUN",
            "method": "transformers.AutoModelForCausalLM, dtype=float16 (matches card's native F16 "
                      "tensor type), greedy decoding, natural-language prompts (no documented official "
                      "template found on model card at preflight time -- prompts analogous to the "
                      "frontier_comparison.py harness for direct comparability)",
            "load_time_sec": load_time,
            "metrics": {
                "relevance_accuracy": rel_correct / len(rel_subset) if rel_subset else None,
                "relevance_n": len(rel_subset),
                "relevance_gen_time_sec": rel_gen_time,
                "extraction_brand": score_field(brand_preds, [e["brand"] for e in ext_subset]),
                "extraction_color": score_field(color_preds, [e["color"] for e in ext_subset]),
                "extraction_n": len(ext_subset),
                "extraction_gen_time_sec": ext_gen_time,
            },
            "n": len(rel_subset) + len(ext_subset),
            "wall_time_sec": time.time() - t0,
        }
    except Exception as exc:
        return {"comparison_status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}",
                "wall_time_sec": time.time() - t0}


if __name__ == "__main__":
    relevance_examples = json.loads(Path("data/frontier_eval_relevance.json").read_text())
    extraction_examples = json.loads(Path("data/abo/understand_eval_grounded.json").read_text())

    print("Running HF01 eCeLLM-S...")
    result = run_ecellm_s(relevance_examples, extraction_examples, n_relevance=60, n_extraction=30)
    print(json.dumps(result, indent=2)[:3000])

    out = {"ecellm_s": result}
    out_path = Path("reports/ecellm_2026-09-27/report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("Wrote", out_path)
