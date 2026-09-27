"""Driver script: run frontier_comparison.py harness against the held-out
frontier_eval_relevance.json / frontier_eval_identity.json sets and record
resolved model snapshots (not aliases) per R02 requirement.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from anthropic import Anthropic
from openai import OpenAI

import frontier_comparison as fc

anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def resolve_snapshot(model_key: str, cfg: dict) -> str:
    try:
        if cfg["provider"] == "anthropic":
            r = anthropic_client.messages.create(
                model=cfg["model_id"], max_tokens=5,
                messages=[{"role": "user", "content": "say ok"}],
            )
            return r.model
        else:
            r = openai_client.chat.completions.create(
                model=cfg["model_id"], messages=[{"role": "user", "content": "say ok"}],
            )
            return r.model
    except Exception as exc:
        return f"RESOLUTION_FAILED: {type(exc).__name__}: {exc}"


def main():
    models = ["claude-haiku-4-5", "claude-sonnet-5", "gpt-5-mini", "gpt-5"]

    resolved = {mk: resolve_snapshot(mk, fc.FRONTIER_MODELS[mk]) for mk in models}
    print("Resolved snapshots:", json.dumps(resolved, indent=2))

    relevance_examples = json.loads(Path("data/frontier_eval_relevance.json").read_text())
    identity_examples = json.loads(Path("data/frontier_eval_identity.json").read_text())

    print(f"Running relevance eval: {len(relevance_examples)} examples x {len(models)} models")
    relevance_results = fc.run_relevance_frontier_eval(relevance_examples, models)

    print(f"Running identity eval: {len(identity_examples)} examples x {len(models)} models")
    identity_results = fc.run_identity_frontier_eval(identity_examples, models)

    out = {
        "resolved_snapshots": resolved,
        "relevance": relevance_results,
        "identity": identity_results,
    }
    out_path = Path("reports/frontier_comparison_2026-09-27/report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("Wrote", out_path)
    for mk in models:
        rel = relevance_results[mk]
        idn = identity_results[mk]
        print(f"{mk} ({rel['model_id']}): relevance_acc={rel['accuracy']:.3f} identity_acc={idn['accuracy']:.3f}")


if __name__ == "__main__":
    main()
