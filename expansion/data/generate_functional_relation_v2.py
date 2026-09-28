"""Additive expansion of match_functional_relation synthetic data only.
Does NOT touch technical_compatibility.jsonl (that task already ties every
frontier model at 1.0 on the corrected, stratified dev set -- no reason to
regenerate it). Motivated by a real, previously-hidden gap: v1's adapter
scored 0.692 on functional_relation vs. 0.846 for every tested frontier
model, once the dev split was corrected to include all three classes
(the original split had all 15 dev rows labeled "complement").

Uses the expanded 63-scenario bank in generate_synthetic_match.py (up from
the original 15) -- new, distinct item pairs, not just more paraphrases of
the same handful of scenarios, since scenario diversity (not row count
alone) is the likely bottleneck at n=65 training rows.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from expansion.data.generate_synthetic_match import (
    FUNCTIONAL_SCENARIOS,
    FunctionalRelationScenario,
    paraphrase_functional,
    validate_functional_record,
)


def main(paraphrases_per_scenario: int = 2):
    scenarios = [FunctionalRelationScenario(*s) for s in FUNCTIONAL_SCENARIOS]

    records = []
    rejected = []
    for s in scenarios:
        for _ in range(paraphrases_per_scenario):
            text = paraphrase_functional(s)
            record = {
                "task": "match_functional_relation",
                "scenario": asdict(s),
                "generated_text": text,
                "label": s.label,
                "label_origin": "structured_scenario_fixed_pre_generation",
            }
            ok, errors = validate_functional_record(record)
            if ok:
                records.append(record)
            else:
                rejected.append({"record": record, "errors": errors})

    out_dir = Path("data/synthetic_match")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "functional_relation_v2_raw.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    (out_dir / "functional_relation_v2_rejected.json").write_text(json.dumps(rejected, indent=2))

    print(json.dumps({
        "scenarios": len(scenarios),
        "paraphrases_per_scenario": paraphrases_per_scenario,
        "generated": len(scenarios) * paraphrases_per_scenario,
        "passed_gate": len(records),
        "rejected": len(rejected),
    }, indent=2))


if __name__ == "__main__":
    main()
