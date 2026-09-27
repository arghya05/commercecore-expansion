"""Structured-scenario-first synthetic data generator for Match predicates
with no real-data path: functional_relation and technical_compatibility.

Per plans/00_expansion_core.md data pipeline invariant and F06: the label is
fixed deterministically in code BEFORE any LLM call. The LLM's only job is
paraphrasing the fixed scenario into natural product text. The exact
serialized record that is validated is the exact record that is saved —
this script writes the validated object directly, never a separate draft.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

FUNCTIONAL_SCENARIOS = [
    # (item_a, item_b, use_context, label, directional)
    ("wireless mouse", "USB-A dongle receiver", "connecting a wireless mouse to a laptop", "complement", True),
    ("4K HDMI cable", "8K HDMI cable", "connecting a laptop to a 4K monitor", "substitute", False),
    ("running shoes size 9", "running shoes size 10", "buying shoes when size 9 is out of stock", "substitute", False),
    ("13-inch laptop", "laptop sleeve 13-inch", "protecting a 13-inch laptop while traveling", "complement", True),
    ("espresso machine", "milk frother", "making a latte at home", "complement", True),
    ("almond milk", "oat milk", "a dairy-free milk alternative for coffee", "substitute", False),
    ("printer", "printer paper", "printing documents at home", "complement", True),
    ("tennis racket", "tennis balls", "playing a match of tennis", "complement", True),
    ("iPhone 15", "Samsung Galaxy S24", "buying a flagship smartphone", "substitute", False),
    ("bicycle", "bicycle helmet", "riding a bicycle safely", "complement", True),
    ("bicycle", "car", "commuting to work", "substitute", False),
    ("standing desk", "desk lamp", "furnishing a home office", "complement", True),
    ("yoga mat", "car tires", "unrelated household purchases", "unrelated", False),
    ("standing desk", "cat litter box", "two items a customer happens to browse in one session, with no functional connection to each other", "unrelated", False),
    ("wireless earbuds", "garden hose", "two items a customer happens to browse in one session, with no functional connection to each other", "unrelated", False),
]

COMPATIBILITY_SCENARIOS = [
    # (item_a, item_b, revision, region, label)
    ("iPhone 15 case", "iPhone 15", "iPhone 15 (2023)", "US", "compatible"),
    ("iPhone 14 case", "iPhone 15", "iPhone 15 (2023)", "US", "incompatible"),
    ("PS5 DualSense controller", "PlayStation 5", "PS5 (2020 model)", "US", "compatible"),
    ("PS4 DualShock controller", "PlayStation 5", "PS5 (2020 model)", "US", "incompatible"),
    ("EU plug adapter", "US 120V hairdryer", "US region appliance", "EU", "incompatible"),
    ("US plug adapter", "US 120V hairdryer", "US region appliance", "US", "compatible"),
    ("Nikon F-mount lens", "Nikon Z-mount camera body", "Z-mount system", "Global", "incompatible"),
    ("Nikon Z-mount lens", "Nikon Z-mount camera body", "Z-mount system", "Global", "compatible"),
    ("microSD card 128GB", "Nintendo Switch", "Switch (all revisions)", "Global", "compatible"),
    ("SATA SSD 2.5-inch", "laptop with M.2 NVMe slot only", "M.2-only laptop", "Global", "incompatible"),
]


@dataclass(frozen=True)
class FunctionalRelationScenario:
    item_a: str
    item_b: str
    use_context: str
    label: str
    directional: bool


@dataclass(frozen=True)
class CompatibilityScenario:
    item_a: str
    item_b: str
    revision: str
    region: str
    label: str


def build_functional_scenarios(n_paraphrases: int, seed: int) -> list[FunctionalRelationScenario]:
    rng = random.Random(seed)
    base = [FunctionalRelationScenario(*s) for s in FUNCTIONAL_SCENARIOS]
    out = []
    for _ in range(n_paraphrases):
        out.append(rng.choice(base))
    return out


def build_compatibility_scenarios(n_paraphrases: int, seed: int) -> list[CompatibilityScenario]:
    rng = random.Random(seed)
    base = [CompatibilityScenario(*s) for s in COMPATIBILITY_SCENARIOS]
    out = []
    for _ in range(n_paraphrases):
        out.append(rng.choice(base))
    return out


def paraphrase_functional(scenario: FunctionalRelationScenario, model: str = "claude-haiku-4-5-20251001") -> str:
    """The LLM only paraphrases; the label is already fixed above and is
    NEVER derived from or re-checked against this generated text."""
    prompt = (
        f"Write one short, natural product-listing-style sentence pair for an ecommerce "
        f"context. Item A: \"{scenario.item_a}\". Item B: \"{scenario.item_b}\". "
        f"Use context: {scenario.use_context}. "
        f"Do not state whether they are complements, substitutes, or unrelated — "
        f"just describe the two items naturally as if from a product page or search result. "
        f"Output only the two short lines, nothing else."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def paraphrase_compatibility(scenario: CompatibilityScenario, model: str = "claude-haiku-4-5-20251001") -> str:
    if scenario.label == "compatible":
        instruction = (
            f"Write a short, natural ecommerce product listing or Q&A snippet that describes "
            f"\"{scenario.item_a}\" being used with, fitted to, or working with "
            f"\"{scenario.item_b}\" (specifically {scenario.revision}, region {scenario.region}). "
            f"The text must clearly convey that the item works with / fits / is designed for "
            f"that specific product — e.g. a customer question confirming fit, or copy stating "
            f"'designed for', 'fits', or 'works with' that exact model. Do not use the words "
            f"'compatible' or 'incompatible' directly."
        )
    else:
        instruction = (
            f"Write a short, natural ecommerce product listing, review, or Q&A snippet where "
            f"\"{scenario.item_a}\" does NOT work with / does not fit "
            f"\"{scenario.item_b}\" (specifically {scenario.revision}, region {scenario.region}). "
            f"The text must clearly convey a fit or functional MISMATCH — e.g. a customer "
            f"complaint that it didn't fit, a warning that it requires a different model/region, "
            f"or copy stating it does NOT work with that specific product. Do not simply list "
            f"the two items side by side without stating the mismatch. Do not use the words "
            f"'compatible' or 'incompatible' directly."
        )
    prompt = (
        instruction
        + " Output only the short snippet (2-4 sentences), nothing else."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def validate_functional_record(record: dict) -> tuple[bool, list[str]]:
    errors = []
    if record["label"] not in {"substitute", "complement", "unrelated", "unknown"}:
        errors.append(f"invalid label {record['label']!r}")
    if not record.get("generated_text") or len(record["generated_text"]) < 10:
        errors.append("generated_text missing or too short")
    if record["scenario"]["item_a"].lower() not in record["generated_text"].lower() and \
       record["scenario"]["item_b"].lower().split()[0] not in record["generated_text"].lower():
        errors.append("generated_text does not clearly reference either scenario item (naturalness/coverage gate)")
    return (len(errors) == 0, errors)


def validate_compatibility_record(record: dict) -> tuple[bool, list[str]]:
    """Structural checks only. Keyword matching cannot reliably judge whether
    text conveys a semantic fit/mismatch signal — natural phrasing varies too
    much (see generator revision history for the false-rejection incident this
    caused). The actual semantic judge is the independent audit in
    audit_synthetic_match.py, run by a DIFFERENT model than the generator."""
    errors = []
    if record["label"] not in {"compatible", "incompatible", "unknown"}:
        errors.append(f"invalid label {record['label']!r}")
    text = record.get("generated_text") or ""
    if len(text) < 10:
        errors.append("generated_text missing or too short")
    return (len(errors) == 0, errors)


def run(n_functional: int = 60, n_compat: int = 40, seed: int = 42) -> dict:
    functional_scenarios = build_functional_scenarios(n_functional, seed)
    compat_scenarios = build_compatibility_scenarios(n_compat, seed + 1)

    functional_records = []
    rejected_functional = []
    for s in functional_scenarios:
        text = paraphrase_functional(s)
        record = {
            "task": "match_functional_relation",
            "scenario": asdict(s),
            "generated_text": text,
            "label": s.label,  # fixed BEFORE the LLM call, never re-derived from generated_text
            "label_origin": "structured_scenario_fixed_pre_generation",
        }
        ok, errors = validate_functional_record(record)
        if ok:
            functional_records.append(record)
        else:
            rejected_functional.append({"record": record, "errors": errors})

    compat_records = []
    rejected_compat = []
    for s in compat_scenarios:
        text = paraphrase_compatibility(s)
        record = {
            "task": "match_technical_compatibility",
            "scenario": asdict(s),
            "generated_text": text,
            "label": s.label,
            "label_origin": "structured_scenario_fixed_pre_generation",
        }
        ok, errors = validate_compatibility_record(record)
        if ok:
            compat_records.append(record)
        else:
            rejected_compat.append({"record": record, "errors": errors})

    out_dir = Path("data/synthetic_match")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "functional_relation.jsonl", "w") as f:
        for r in functional_records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "technical_compatibility.jsonl", "w") as f:
        for r in compat_records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "rejected.json", "w") as f:
        json.dump({"functional": rejected_functional, "compat": rejected_compat}, f, indent=2)

    summary = {
        "functional_generated": len(functional_scenarios),
        "functional_passed_gate": len(functional_records),
        "functional_rejected": len(rejected_functional),
        "compat_generated": len(compat_scenarios),
        "compat_passed_gate": len(compat_records),
        "compat_rejected": len(rejected_compat),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
