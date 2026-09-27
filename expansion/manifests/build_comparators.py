"""Build expansion/manifests/comparators.json from the governing plan's
38-model register (capability_expansion/related_model_comparators.json) plus
its external comparator and provider-role entries, per R02
(RELATED_MODEL_COMPARISON_MATRIX.md). This is the full seed inventory -- every
row is preserved, none are dropped, regardless of whether this pass attempts
to run it.
"""
from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path("/Users/arghyamukherjee/Downloads/training/capability_expansion/related_model_comparators.json")
OUT = Path("expansion/manifests/comparators.json")


def main():
    src = json.loads(SOURCE.read_text())

    manifest = {
        "schema_version": 1,
        "built_from": str(SOURCE),
        "built_date": "2026-09-27",
        "purpose": "Full seed inventory of required related-model comparators (38 HF repos + 1 external "
                   "+ 4 frontier provider roles) per RELATED_MODEL_COMPARISON_MATRIX.md R02. This file "
                   "does not itself carry run results -- see benchmark_cells.json for the per-(model, "
                   "task) attempted/NOT_RUN disposition produced in this pass.",
        "models": src["models"],
        "external_comparators": src["external_comparators"],
        "provider_roles": src["provider_roles"],
        "counts": {
            "hf_models": len(src["models"]),
            "external_comparators": len(src["external_comparators"]),
            "provider_roles": len(src["provider_roles"]),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {OUT} with {manifest['counts']}")


if __name__ == "__main__":
    main()
