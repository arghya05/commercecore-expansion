"""Independent semantic audit of generated synthetic Match records.

Uses a DIFFERENT model family (GPT, not Claude) than the generator (Claude
Haiku) to avoid a model auditing its own output. Per plans/00_expansion_core.md:
"Use independently authored semantic fixtures" and per F06/F18: a naturalness
gate alone is not sufficient; a stratified semantic audit with error
categories is required before any generated data is trusted for training.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

AUDIT_PROMPT = """You are auditing a training example for a product-relationship classifier. Given the generated text and the assigned label, judge whether the label is actually supported by the text, as a human annotator would.

Generated text: {text}
Assigned label: {label}
Task: {task}
{extra}

Answer with a JSON object only: {{"supported": true/false, "reason": "<one sentence>"}}"""


def audit_functional_record(record: dict) -> dict:
    extra = f"Scenario: item_a={record['scenario']['item_a']!r}, item_b={record['scenario']['item_b']!r}, use_context={record['scenario']['use_context']!r}"
    prompt = AUDIT_PROMPT.format(
        text=record["generated_text"],
        label=record["label"],
        task="functional_relation (substitute / complement / unrelated)",
        extra=extra,
    )
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    verdict = json.loads(resp.choices[0].message.content)
    return {**record, "audit": verdict}


def audit_compat_record(record: dict) -> dict:
    extra = f"Scenario: item_a={record['scenario']['item_a']!r}, item_b={record['scenario']['item_b']!r}, revision={record['scenario']['revision']!r}, region={record['scenario']['region']!r}"
    prompt = AUDIT_PROMPT.format(
        text=record["generated_text"],
        label=record["label"],
        task="technical_compatibility (compatible / incompatible)",
        extra=extra,
    )
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    verdict = json.loads(resp.choices[0].message.content)
    return {**record, "audit": verdict}


def run() -> dict:
    functional = [json.loads(l) for l in open("data/synthetic_match/functional_relation.jsonl")]
    compat = [json.loads(l) for l in open("data/synthetic_match/technical_compatibility.jsonl")]

    audited_functional = []
    for i, r in enumerate(functional):
        audited_functional.append(audit_functional_record(r))
        print(f"functional {i+1}/{len(functional)}", flush=True)

    audited_compat = []
    for i, r in enumerate(compat):
        audited_compat.append(audit_compat_record(r))
        print(f"compat {i+1}/{len(compat)}", flush=True)

    n_func_supported = sum(1 for r in audited_functional if r["audit"]["supported"])
    n_compat_supported = sum(1 for r in audited_compat if r["audit"]["supported"])

    out_dir = Path("data/synthetic_match")
    with open(out_dir / "functional_relation_audited.jsonl", "w") as f:
        for r in audited_functional:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "technical_compatibility_audited.jsonl", "w") as f:
        for r in audited_compat:
            f.write(json.dumps(r) + "\n")

    summary = {
        "functional_n": len(audited_functional),
        "functional_supported": n_func_supported,
        "functional_supported_rate": n_func_supported / len(audited_functional) if audited_functional else None,
        "compat_n": len(audited_compat),
        "compat_supported": n_compat_supported,
        "compat_supported_rate": n_compat_supported / len(audited_compat) if audited_compat else None,
        "unsupported_examples": [
            {"text": r["generated_text"], "label": r["label"], "reason": r["audit"]["reason"]}
            for r in (audited_functional + audited_compat)
            if not r["audit"]["supported"]
        ][:20],
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
