"""Independent audit of the v2 functional_relation expansion only, same
protocol as audit_synthetic_match.py (GPT-5-mini judges Claude-Haiku-
generated text, different model family than the generator)."""
from __future__ import annotations

import json
from pathlib import Path

from expansion.data.audit_synthetic_match import audit_functional_record


def main():
    records = [json.loads(l) for l in open("data/synthetic_match/functional_relation_v2_raw.jsonl")]
    audited = []
    for i, r in enumerate(records):
        audited.append(audit_functional_record(r))
        print(f"functional_v2 {i+1}/{len(records)}", flush=True)

    n_supported = sum(1 for r in audited if r["audit"]["supported"])
    out_dir = Path("data/synthetic_match")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "functional_relation_v2_audited.jsonl").write_text(
        "\n".join(json.dumps(r) for r in audited) + "\n"
    )

    summary = {
        "n": len(audited),
        "supported": n_supported,
        "supported_rate": n_supported / len(audited) if audited else None,
        "unsupported": [
            {"text": r["generated_text"], "label": r["label"], "reason": r["audit"]["reason"]}
            for r in audited if not r["audit"]["supported"]
        ],
    }
    (out_dir / "functional_relation_v2_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
