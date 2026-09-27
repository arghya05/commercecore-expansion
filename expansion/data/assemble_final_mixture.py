"""Assembles the final training mixture from all quality-gated sources.

Only pulls from sources whose registry ingestion_gate == PASS. Records the
exact row counts, task/split distribution, and source provenance for the
experiment report, per 00_expansion_core.md's "Required experiment output."
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expansion.data.registry import SourceRegistry


def main():
    reg = SourceRegistry(Path("expansion/manifests/sources/registry.json"))

    required_sources = ["esci", "wdc_products_80pair", "synthetic_match_functional_compat"]
    for s in required_sources:
        reg.require_gate(s, gate="PASS")

    rows = []
    with open("data/training_mixture_v1.jsonl") as f:
        rows.extend(json.loads(l) for l in f)
    with open("data/synthetic_match/functional_relation.jsonl") as f:
        rows.extend(json.loads(l) for l in f)
    with open("data/synthetic_match/technical_compatibility.jsonl") as f:
        rows.extend(json.loads(l) for l in f)

    # excluded-task guard, defense in depth
    from expansion.data.validate_record import EXCLUDED_TASK_IDS
    for r in rows:
        if r["task"].lower() in EXCLUDED_TASK_IDS:
            raise ValueError(f"excluded task leaked into final mixture: {r['task']}")

    dist = Counter((r["task"], r["split"]) for r in rows)

    out_path = Path("data/training_mixture_final.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "total_rows": len(rows),
        "distribution": {f"{k[0]}/{k[1]}": v for k, v in dist.items()},
        "sources": required_sources,
        "output_path": str(out_path),
    }
    Path("data/training_mixture_final_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
