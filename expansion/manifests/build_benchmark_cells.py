"""Build expansion/manifests/benchmark_cells.json: one row per (model, task)
cell across the full 38-model + external + provider-role register, recording
comparison_status (RUN/NOT_RUN), metrics, n, exact revision, and -- for every
NOT_RUN cell -- a specific technical reason. Per R02: no model from the
register is silently dropped, even if this pass did not attempt it.

Run AFTER expansion/eval/{extraction_comparators,comparator_baselines,
run_frontier}.py have produced their reports/*/report.json files.
"""
from __future__ import annotations

import json
from pathlib import Path

COMPARATORS_PATH = Path("expansion/manifests/comparators.json")
OUT_PATH = Path("expansion/manifests/benchmark_cells.json")

EXTRACTION_REPORT = Path("reports/extraction_comparators_2026-09-27/report.json")
COMPARATOR_REPORT = Path("reports/comparator_baselines_2026-09-27/report.json")
FRONTIER_REPORT = Path("reports/frontier_comparison_2026-09-27/report.json")
UNDERSTAND_BASELINE_REPORT = Path("reports/understand_baseline_2026-09-27/report.json")
MATCH_BASELINE_REPORT = Path("reports/match_baseline_2026-09-27/report.json")

RUN_DATE = "2026-09-27"


def load_json(p: Path) -> dict:
    if p.exists():
        return json.loads(p.read_text())
    return {}


def cell(repository, revision, group, comparison_role, task, metric_name, metrics, n,
         comparison_status, reason=None, method=None, wall_time_sec=None):
    row = {
        "repository": repository,
        "revision": revision,
        "group": group,
        "comparison_role": comparison_role,
        "task": task,
        "comparison_status": comparison_status,
        "run_date": RUN_DATE if comparison_status == "RUN" else None,
        "n_examples": n,
        "metrics": metrics if comparison_status == "RUN" else None,
        "method": method,
        "wall_time_sec": wall_time_sec,
        "reason": reason,
    }
    return row


def main():
    comparators = load_json(COMPARATORS_PATH)
    extraction = load_json(EXTRACTION_REPORT)
    comparator = load_json(COMPARATOR_REPORT)
    frontier = load_json(FRONTIER_REPORT)
    understand_baseline = load_json(UNDERSTAND_BASELINE_REPORT)
    match_baseline = load_json(MATCH_BASELINE_REPORT)

    models_by_repo = {m["repository"]: m for m in comparators["models"]}
    rows = []

    def add_attempted(repo, task, metric_key_map, source_dict, method_key="method"):
        """source_dict is the specific per-model report dict (e.g.
        extraction["hf08_gliner2_5_base_v1"])."""
        m = models_by_repo[repo]
        status = source_dict.get("comparison_status", "NOT_RUN")
        if status == "RUN":
            metrics = {k: source_dict[v] for k, v in metric_key_map.items() if v in source_dict}
            rows.append(cell(
                repo, m["observed_revision"], m["group"], m["comparison_role"], task,
                None, metrics, source_dict.get("n_examples") or source_dict.get("n"),
                "RUN", method=source_dict.get(method_key), wall_time_sec=source_dict.get("wall_time_sec"),
            ))
        else:
            rows.append(cell(
                repo, m["observed_revision"], m["group"], m["comparison_role"], task,
                None, None, None, "NOT_RUN", reason=source_dict.get("reason", "attempted but result missing"),
            ))

    # ---- HF08: gliner2.5-base-v1 (Understand: brand+color extraction) ----
    if "hf08_gliner2_5_base_v1" in extraction:
        src = extraction["hf08_gliner2_5_base_v1"]
        if src.get("comparison_status") == "RUN":
            rows.append(cell(
                "fastino/gliner2.5-base-v1", models_by_repo["fastino/gliner2.5-base-v1"]["observed_revision"],
                "HF08", "STRUCTURED_EXTRACTION", "Understand",
                None, {"brand_f1": src["brand"]["f1"], "brand_precision": src["brand"]["precision"],
                       "brand_recall": src["brand"]["recall"], "color_f1": src["color"]["f1"],
                       "color_precision": src["color"]["precision"], "color_recall": src["color"]["recall"]},
                src["n_examples"], "RUN", method=src.get("loader"), wall_time_sec=src.get("wall_time_sec"),
            ))
        else:
            rows.append(cell(
                "fastino/gliner2.5-base-v1", models_by_repo["fastino/gliner2.5-base-v1"]["observed_revision"],
                "HF08", "STRUCTURED_EXTRACTION", "Understand", None, None, None, "NOT_RUN",
                reason=src.get("reason"),
            ))
    else:
        rows.append(cell(
            "fastino/gliner2.5-base-v1", models_by_repo["fastino/gliner2.5-base-v1"]["observed_revision"],
            "HF08", "STRUCTURED_EXTRACTION", "Understand", None, None, None, "NOT_RUN",
            reason="not attempted in this pass",
        ))

    # ---- HF08: gliner2-base-v1 -- ALREADY RUN in an earlier session (understand_baselines.py) ----
    if understand_baseline.get("gliner2_base") and "error" not in understand_baseline["gliner2_base"]:
        src = understand_baseline["gliner2_base"]
        rows.append(cell(
            "fastino/gliner2-base-v1", models_by_repo["fastino/gliner2-base-v1"]["observed_revision"],
            "HF08", "STRUCTURED_EXTRACTION", "Understand",
            None, {"brand_f1": src["brand"]["f1"], "brand_precision": src["brand"]["precision"],
                   "brand_recall": src["brand"]["recall"], "color_f1": src["color"]["f1"],
                   "color_precision": src["color"]["precision"], "color_recall": src["color"]["recall"]},
            src["n_examples"], "RUN",
            method="gliner2.GLiNER2.from_pretrained (legacy span loader; correct native loader for this "
                   "checkpoint, unlike gliner2.5-base-v1 which requires AutoExtractor) -- run in prior "
                   "session, reused here",
        ))
    else:
        rows.append(cell(
            "fastino/gliner2-base-v1", models_by_repo["fastino/gliner2-base-v1"]["observed_revision"],
            "HF08", "STRUCTURED_EXTRACTION", "Understand", None, None, None, "NOT_RUN",
            reason="prior-session report missing or errored",
        ))

    # ---- HF09: NuExtract-2.0-2B ----
    if "hf09_nuextract_2_0_2b" in extraction:
        src = extraction["hf09_nuextract_2_0_2b"]
        if src.get("comparison_status") == "RUN":
            rows.append(cell(
                "numind/NuExtract-2.0-2B", models_by_repo["numind/NuExtract-2.0-2B"]["observed_revision"],
                "HF09", "STRUCTURED_EXTRACTION", "Understand",
                None, {"brand_f1": src["brand"]["f1"], "brand_precision": src["brand"]["precision"],
                       "brand_recall": src["brand"]["recall"], "color_f1": src["color"]["f1"],
                       "color_precision": src["color"]["precision"], "color_recall": src["color"]["recall"],
                       "json_parse_failures": src.get("json_parse_failures")},
                src["n_examples"], "RUN", method=src.get("loader") + " | " + src.get("note", ""),
                wall_time_sec=src.get("wall_time_sec"),
            ))
        else:
            rows.append(cell(
                "numind/NuExtract-2.0-2B", models_by_repo["numind/NuExtract-2.0-2B"]["observed_revision"],
                "HF09", "STRUCTURED_EXTRACTION", "Understand", None, None, None, "NOT_RUN",
                reason=src.get("reason"),
            ))
    else:
        rows.append(cell(
            "numind/NuExtract-2.0-2B", models_by_repo["numind/NuExtract-2.0-2B"]["observed_revision"],
            "HF09", "STRUCTURED_EXTRACTION", "Understand", None, None, None, "NOT_RUN",
            reason="not attempted in this pass",
        ))

    # ---- HF06: RexBERT-base (linear probe) ----
    repo = "thebajajra/RexBERT-base"
    src = comparator.get("hf06_rexbert_base_linear_probe", {})
    if src.get("comparison_status") == "RUN":
        rows.append(cell(
            repo, models_by_repo[repo]["observed_revision"], "HF06", "DOMAIN_ENCODER_REQUIRES_TASK_HEAD",
            "Match", None, {"macro_f1": src["macro_f1"], "per_class": src["per_class"]},
            src["n_dev"], "RUN", method=src.get("method"), wall_time_sec=src.get("wall_time_sec"),
        ))
    else:
        rows.append(cell(
            repo, models_by_repo[repo]["observed_revision"], "HF06", "DOMAIN_ENCODER_REQUIRES_TASK_HEAD",
            "Match", None, None, None, "NOT_RUN", reason=src.get("reason", "not attempted in this pass"),
        ))

    # ---- HF06: other RexBERT sizes -- deferred, same family, same rationale ----
    for repo in ["thebajajra/RexBERT-micro", "thebajajra/RexBERT-mini", "thebajajra/RexBERT-large"]:
        rows.append(cell(
            repo, models_by_repo[repo]["observed_revision"], "HF06", "DOMAIN_ENCODER_REQUIRES_TASK_HEAD",
            "Match", None, None, None, "NOT_RUN",
            reason="deferred: outside this pass's scope (only RexBERT-base linear-probe attempted; "
                   "micro/mini/large are the same architecture family at different sizes)",
        ))

    # ---- HF07: RexReranker-0.6B ----
    repo = "thebajajra/RexReranker-0.6B"
    src = comparator.get("hf07_rexreranker_0_6b", {})
    if src.get("comparison_status") == "RUN":
        rows.append(cell(
            repo, models_by_repo[repo]["observed_revision"], "HF07", "DOMAIN_RERANKER", "Rank",
            None, {"precision_exact": src["precision_exact"], "recall_exact": src["recall_exact"],
                   "f1_exact": src["f1_exact"], "roc_auc_exact": src["roc_auc_exact"]},
            src["n"], "RUN", method=src.get("method"), wall_time_sec=src.get("wall_time_sec"),
        ))
    else:
        rows.append(cell(
            repo, models_by_repo[repo]["observed_revision"], "HF07", "DOMAIN_RERANKER", "Rank",
            None, None, None, "NOT_RUN", reason=src.get("reason", "not attempted in this pass"),
        ))

    # ---- HF07: other RexReranker sizes -- deferred ----
    for repo in ["thebajajra/RexReranker-micro", "thebajajra/RexReranker-mini",
                 "thebajajra/RexReranker-base", "thebajajra/RexReranker-large"]:
        rows.append(cell(
            repo, models_by_repo[repo]["observed_revision"], "HF07", "DOMAIN_RERANKER", "Rank",
            None, None, None, "NOT_RUN",
            reason="deferred: outside this pass's scope (only RexReranker-0.6B attempted; "
                   "micro/mini/base/large are the same architecture family at different sizes)",
        ))

    # ---- HF11: general reranking controls ----
    for repo, key in [("cross-encoder/ms-marco-MiniLM-L6-v2", "hf11_msmarco_minilm_l6_v2"),
                       ("cross-encoder/ettin-reranker-150m-v1", "hf11_ettin_reranker_150m_v1")]:
        src = comparator.get(key, {})
        if src.get("comparison_status") == "RUN":
            rows.append(cell(
                repo, models_by_repo[repo]["observed_revision"], "HF11", "RERANKING_CONTROL", "Rank",
                None, {"roc_auc_exact": src["roc_auc_exact"],
                       "median_threshold_precision_exact": src["median_threshold_precision_exact"],
                       "median_threshold_recall_exact": src["median_threshold_recall_exact"],
                       "median_threshold_f1_exact": src["median_threshold_f1_exact"]},
                src["n"], "RUN", method=src.get("method"), wall_time_sec=src.get("wall_time_sec"),
            ))
        else:
            rows.append(cell(
                repo, models_by_repo[repo]["observed_revision"], "HF11", "RERANKING_CONTROL", "Rank",
                None, None, None, "NOT_RUN", reason=src.get("reason", "not attempted in this pass"),
            ))

    # ---- HF11: Qwen3-Reranker-0.6B -- register also lists this under HF11 ----
    repo = "Qwen/Qwen3-Reranker-0.6B"
    rows.append(cell(
        repo, models_by_repo[repo]["observed_revision"], "HF11", "RERANKING_CONTROL", "Rank",
        None, None, None, "NOT_RUN",
        reason="deferred: outside this pass's scope (native interface requires a custom yes/no-logit "
               "scoring wrapper like RexReranker rather than the standard CrossEncoder API; "
               "RexReranker-0.6B, its fine-tuned derivative, was run instead as the representative "
               "of this exact scoring mechanism)",
    ))

    # ---- HF10: retrieval controls (proxy) ----
    for repo, key in [("Qwen/Qwen3-Embedding-0.6B", "hf10_qwen3_embedding_0_6b_proxy"),
                       ("intfloat/multilingual-e5-small", "hf10_multilingual_e5_small_proxy")]:
        src = comparator.get(key, {})
        if src.get("comparison_status") == "RUN":
            rows.append(cell(
                repo, models_by_repo[repo]["observed_revision"], "HF10", "TEXT_RETRIEVAL_CONTROL", "Retrieve",
                None, {"roc_auc_exact_PROXY": src["roc_auc_exact"],
                       "median_threshold_precision_exact_PROXY": src["median_threshold_precision_exact"],
                       "median_threshold_recall_exact_PROXY": src["median_threshold_recall_exact"],
                       "median_threshold_f1_exact_PROXY": src["median_threshold_f1_exact"]},
                src["n"], "RUN", method=src.get("method"), wall_time_sec=src.get("wall_time_sec"),
            ))
        else:
            rows.append(cell(
                repo, models_by_repo[repo]["observed_revision"], "HF10", "TEXT_RETRIEVAL_CONTROL", "Retrieve",
                None, None, None, "NOT_RUN", reason=src.get("reason", "not attempted in this pass"),
            ))

    # ---- HF01: eCeLLM-S/M/L ----
    ecellm_report = load_json(Path("reports/ecellm_2026-09-27/report.json"))
    for repo in ["NingLab/eCeLLM-S", "NingLab/eCeLLM-M", "NingLab/eCeLLM-L"]:
        key = repo.split("/")[-1].lower().replace("-", "_")
        src = ecellm_report.get(key, {})
        if src.get("comparison_status") == "RUN":
            rows.append(cell(
                repo, models_by_repo[repo]["observed_revision"], "HF01", "DOMAIN_MULTITASK", "Understand+Match",
                None, src.get("metrics"), src.get("n"), "RUN", method=src.get("method"),
                wall_time_sec=src.get("wall_time_sec"),
            ))
        else:
            rows.append(cell(
                repo, models_by_repo[repo]["observed_revision"], "HF01", "DOMAIN_MULTITASK", "Understand+Match",
                None, None, None, "NOT_RUN",
                reason=src.get("reason", "not attempted in this pass: see benchmark_cells reason field"),
            ))

    # ---- Frontier provider roles ----
    if frontier:
        resolved = frontier.get("resolved_snapshots", {})
        role_map = {
            "claude-haiku-4-5": "ANTHROPIC_LOWER_COST",
            "claude-sonnet-5": "ANTHROPIC_STRONG",
            "gpt-5-mini": "OPENAI_LOWER_COST",
            "gpt-5": "OPENAI_STRONG",
        }
        for model_key, role in role_map.items():
            rel = frontier.get("relevance", {}).get(model_key)
            idn = frontier.get("identity", {}).get(model_key)
            if rel and idn:
                rows.append(cell(
                    f"provider:{model_key}", resolved.get(model_key), "FRONTIER", role,
                    "Match(relevance)+Match(identity)",
                    None,
                    {"relevance_accuracy": rel["accuracy"], "relevance_invalid_or_error_rate": rel["invalid_or_error_rate"],
                     "identity_accuracy": idn["accuracy"], "identity_invalid_or_error_rate": idn["invalid_or_error_rate"]},
                    {"relevance_n": rel["n"], "identity_n": idn["n"]}, "RUN",
                    method=f"resolved_model_id={rel['model_id']}; prompted single-turn classification, "
                           f"see expansion/eval/frontier_comparison.py",
                ))
            else:
                rows.append(cell(
                    f"provider:{model_key}", resolved.get(model_key), "FRONTIER", role,
                    "Match(relevance)+Match(identity)", None, None, None, "NOT_RUN",
                    reason="frontier harness did not complete for this model in this pass",
                ))
    else:
        for model_key, role in [("claude-haiku-4-5", "ANTHROPIC_LOWER_COST"), ("claude-sonnet-5", "ANTHROPIC_STRONG"),
                                 ("gpt-5-mini", "OPENAI_LOWER_COST"), ("gpt-5", "OPENAI_STRONG")]:
            rows.append(cell(
                f"provider:{model_key}", None, "FRONTIER", role, "Match(relevance)+Match(identity)",
                None, None, None, "NOT_RUN", reason="frontier harness report not found",
            ))

    # ---- Everything else in the 38-model register not covered above: NOT_RUN, deferred ----
    covered_repos = {r["repository"] for r in rows if not r["repository"].startswith("provider:")}
    for repo, m in models_by_repo.items():
        if repo not in covered_repos:
            rows.append(cell(
                repo, m["observed_revision"], m["group"], m["comparison_role"],
                ",".join(m.get("planned_tasks", [])) or "UNSPECIFIED",
                None, None, None, "NOT_RUN",
                reason="deferred: outside this pass's scope" if m["planning_status"] not in
                       ("BLOCKED_LICENSE_AND_LOADER_VALIDATION", "BLOCKED_CAPABILITY_AND_CONFIG_VALIDATION")
                else f"blocked at plan level: {m['planning_status']}",
            ))

    # ---- External comparator (EXT01) ----
    for ext in comparators.get("external_comparators", []):
        rows.append({
            "repository": ext.get("author_repository") or ext.get("id"),
            "revision": None, "group": "EXT01", "comparison_role": ext.get("name"),
            "task": "UNSPECIFIED", "comparison_status": "NOT_RUN", "run_date": None,
            "n_examples": None, "metrics": None, "method": None, "wall_time_sec": None,
            "reason": "deferred: outside this pass's scope; " + ext.get("note", ""),
        })

    out = {
        "schema_version": 1,
        "built_date": RUN_DATE,
        "purpose": "Per-(model, task) benchmark cell register for R02. Every model in "
                   "expansion/manifests/comparators.json has at least one row here; "
                   "comparison_status=NOT_RUN rows carry a specific reason, never a fabricated score.",
        "n_rows": None,
        "cells": rows,
    }
    out["n_rows"] = len(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH} with {len(rows)} rows "
          f"({sum(1 for r in rows if r['comparison_status']=='RUN')} RUN, "
          f"{sum(1 for r in rows if r['comparison_status']=='NOT_RUN')} NOT_RUN)")


if __name__ == "__main__":
    main()
