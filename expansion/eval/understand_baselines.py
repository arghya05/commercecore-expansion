"""Rules vs. GLiNER2-base comparison for Commerce-Understand field extraction.

Per plans/01_commerce_understand.md: "Compare rules and dictionary/unit tools,
fastino/gliner2-base-v1 (205M)... Evaluate all on the same mapped task/split,
never against unrelated card scores." Train only if missing source evidence
and deterministic mapping cannot explain the baseline gap (F10).

Gold is text-grounded: brand/color values are only counted as gold if they
actually occur in the supplied text (see data/abo/understand_eval_grounded.json
construction) — per F07, a value in structured metadata but absent from text
is not text-grounded gold.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def rules_extract_brand(text: str, known_brands: list[str]) -> str | None:
    """Deterministic longest-match against a brand dictionary built from the
    eval set's own gold brands (a fair, disclosed dictionary baseline, not
    hand-tuned per example)."""
    text_lower = text.lower()
    matches = [b for b in known_brands if b.lower() in text_lower]
    if not matches:
        return None
    return max(matches, key=len)


COLOR_WORDS = [
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "grey", "gray", "silver", "gold", "beige", "navy",
    "translucent", "clear", "multicolor", "multicolored",
]


def rules_extract_color(text: str) -> str | None:
    text_lower = text.lower()
    found = [c for c in COLOR_WORDS if re.search(rf"\b{re.escape(c)}\b", text_lower)]
    if not found:
        return None
    return found[0]


def score_field(preds: list[str | None], gold: list[str]) -> dict:
    tp = fp = fn = 0
    for p, g in zip(preds, gold):
        g_norm = g.strip().lower()
        p_norm = (p or "").strip().lower()
        if p is None:
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
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "n": len(gold)}


def run(eval_path: Path, out_path: Path, brand_dictionary_path: Path, n_gliner: int = 200) -> dict:
    examples = json.loads(eval_path.read_text())
    # Brand dictionary is built from the FULL shard's distinct brands, not from
    # the eval set's own gold labels — using eval-derived brands would make the
    # rules baseline circularly perfect on exactly the brands it is scored against.
    known_brands = json.loads(brand_dictionary_path.read_text())

    rules_brand_preds = [rules_extract_brand(e["text"], known_brands) for e in examples]
    rules_color_preds = [rules_extract_color(e["text"]) for e in examples]

    gold_brand = [e["brand"] for e in examples]
    gold_color = [e["color"] for e in examples]

    report = {
        "n_examples": len(examples),
        "rules": {
            "brand": score_field(rules_brand_preds, gold_brand),
            "color": score_field(rules_color_preds, gold_color),
        },
    }

    try:
        from gliner2 import GLiNER2

        extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        subset = examples[:n_gliner]
        gliner_brand_preds = []
        gliner_color_preds = []
        for e in subset:
            result = extractor.extract_entities(e["text"], ["brand", "color"])
            ents = result.get("entities", {})
            gliner_brand_preds.append((ents.get("brand") or [None])[0])
            gliner_color_preds.append((ents.get("color") or [None])[0])
        report["gliner2_base"] = {
            "n_examples": len(subset),
            "brand": score_field(gliner_brand_preds, [e["brand"] for e in subset]),
            "color": score_field(gliner_color_preds, [e["color"] for e in subset]),
        }
    except Exception as exc:  # pragma: no cover - reported, not silently swallowed
        report["gliner2_base"] = {"error": f"{type(exc).__name__}: {exc}"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    r = run(
        Path("data/abo/understand_eval_grounded.json"),
        Path("reports/understand_baseline_2026-09-27/report.json"),
        Path("data/abo/brand_dictionary.json"),
    )
    print(json.dumps(r, indent=2))
