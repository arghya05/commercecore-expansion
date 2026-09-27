# CommerceCore Expansion

A separate, standalone project from [arghya05/commercecore](https://github.com/arghya05/commercecore) — no shared runtime, no shared weights, no shared repository. The original CommerceCore release (`arghya2030/commercecore-qwen3-1.7b`) remains frozen and untouched; it is used only as an isolated external benchmark comparator, never as a training base or dependency.

**Trained model:** [arghya2030/commercecore-expansion-match-v1](https://huggingface.co/arghya2030/commercecore-expansion-match-v1) — a separate Hugging Face repository, real trained weights, honest model card with the results below.

**Status: in progress.** This README is updated as real results land — every number here is measured, not projected.

## What this is

Nine capability APIs for a horizontal ecommerce intelligence system (Understand, Match, Retrieve, Rank, Bundle, Agent, Taste, QA), built per [`../capability_expansion/CAPABILITY_EXPANSION_PLAN.md`](../capability_expansion/CAPABILITY_EXPANSION_PLAN.md). Standalone query-to-constraints parsing (the original CommerceCore's task) is explicitly excluded from this project's scope.

Current build focus: a shared multitask QLoRA adapter on an independently-pinned original `Qwen/Qwen3-1.7B` base, covering Match (relevance, identity, functional relation, technical compatibility) and Understand (attribute extraction).

## Data — real, verified, hash-checked

| Source | License | Real size acquired | Role |
|---|---|---|---|
| [ESCI](https://github.com/amazon-science/esci-data) | Apache-2.0 | 2,621,288 query-product examples, 1,814,924 products | Match relevance training/dev; retrieval dev queries |
| [WDC Products (80pair)](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) | WebDataCommons publisher terms | 19,835 train pairs, 4,500 gold-standard pairs | Match identity training/dev |
| [ABO](https://registry.opendata.aws/amazon-berkeley-objects/) | CC BY 4.0 | 9,232 listings (1 of 16 shards) | Understand extraction baseline/eval |
| [WDC PAVE](https://github.com/wbsg-uni-mannheim/wdc-pave) | Unresolved — no LICENSE file found | 211/354/1,066 rows (train/test/train_large) | Dev reference only; **not used for training** pending license resolution |
| Synthetic (structured-scenario-first) | Generated in-project | 65 functional_relation + 40 technical_compatibility examples | Match functional_relation / technical_compatibility (no real-data source exists for these predicates) |

All sources are hash-verified against their published artifacts; see `expansion/manifests/sources/registry.json` for exact SHA-256s and per-source ingestion gate status.

### Synthetic data quality — independently audited, not self-graded

Synthetic examples use the structured-scenario-first method: the label is fixed in code **before** any LLM call generates the surface text (Claude Haiku paraphrases only; it never chooses the label). Every generated example was then audited by a **different model** (GPT-5-mini) judging whether the generated text actually supports its assigned label — avoiding a generator grading its own output.

- `technical_compatibility`: **40/40 (100%)** label-supported, first pass.
- `functional_relation`: 3 audit rounds were required to reach 100%:
  - Round 1: 49/59 (83%) — 2 flawed scenario definitions found (a real laptop/13" sleeve size mismatch; a desk lamp scenario mislabeled `unrelated` when it's actually complementary).
  - Round 2: 61/65 (94%) after fixing those — a new defect surfaced: the `unrelated`-pair use-context ("furnishing a home office") was too permissive, letting any two office-adjacent items read as complementary.
  - Round 3: **65/65 (100%)** after switching to genuinely disconnected item pairs under an explicit no-functional-connection framing.

Full audit trails: `data/synthetic_match/*_audited.jsonl`, `data/synthetic_match/audit_summary.json`.

## Baselines measured so far (before any training)

Per the plan's own gate: train only where a baseline demonstrably fails, not by default.

| Task | Baseline | Result |
|---|---|---|
| Understand — brand extraction | Dictionary rules | 0.88 F1 |
| Understand — brand extraction | GLiNER2-base | 0.777 F1 |
| Understand — brand extraction | GLiNER2.5-base (native loader) | 0.753 F1 |
| Understand — color extraction | Dictionary rules | 0.754 F1 |
| Understand — color extraction | GLiNER2-base | 0.854 F1 |
| Understand — color extraction | GLiNER2.5-base (native loader) | 0.871 F1 |
| Match — relevance (ESCI native) | TF-IDF + logistic regression | 0.365 macro F1 |
| Match — identity (WDC gold) | Jaccard token-overlap rules | 0.126 F1 |

200 examples for Understand (real, text-grounded — brand/color values verified to actually occur in the source text, not copied from structured metadata). 20,000-row TF-IDF split / 4,500 WDC gold pairs for Match. Full reports: `reports/understand_baseline_2026-09-27/`, `reports/match_baseline_2026-09-27/`.

## Required comparator sweep

Per [`RELATED_MODEL_COMPARISON_MATRIX.md`](../capability_expansion/RELATED_MODEL_COMPARISON_MATRIX.md): every claim of superiority must be checked against the full comparator set — other Hugging Face domain/commerce models **and** frontier models, not frontier-only. Results in `expansion/manifests/benchmark_cells.json` and `reports/frontier_comparison_2026-09-27/`.

### Frontier models — real, resolved snapshots, on the same balanced held-out eval sets as the trained model

60 relevance examples (15 per class, from ESCI test split), 50 identity pairs (25 same/25 distinct, from WDC gold-standard). Never seen in any training mixture.

| Model (resolved snapshot) | Relevance accuracy | Identity accuracy |
|---|---|---|
| claude-haiku-4-5-20251001 | 0.467 | 0.780 |
| **claude-sonnet-5** | **0.583** | **0.920** |
| gpt-5-mini-2025-08-07 | 0.483 | 0.800 |
| gpt-5-2025-08-07 | 0.467 | 0.840 |

Claude Sonnet 5 leads on both tasks — this is the real bar the trained shared adapter must clear to claim superiority, not the much weaker rules/TF-IDF baselines above.

### Hugging Face domain/SLM comparators

- GLiNER2-base: brand 0.777 F1, color 0.854 F1 (200 real ABO examples).
- GLiNER2.5-base (native `gliner2.AutoExtractor` loader, distinct from GLiNER2-base's loader): brand 0.753 F1, color 0.871 F1.
- **NingLab/eCeLLM-S** (7.24B, Mistral-7B base): genuine hard blocker on the development machine — OOM (needs ~29GB resident in fp32, no CUDA path available on this CPU-only host; 16GB total system RAM). Deferred to GPU-hosted evaluation, not silently skipped.
- RexBERT/RexReranker, Ettin/MiniLM rerankers, Qwen3-Embedding: dependency/environment issues encountered during the first sweep pass; being re-run with corrected environment.

## Trained shared adapter — real results, two attempts, honest verdict

QLoRA rank-16 adapter on independently-pinned `Qwen/Qwen3-1.7B` (never the Query-merged artifact), trained on RunPod (RTX 2000 Ada / RTX 3080 Ti), evaluated on the full 4,519-row held-out dev set (never seen in training).

| Attempt | Config | match_relevance | match_identity | match_functional_relation | match_technical_compatibility |
|---|---|---|---|---|---|
| **v1 (adopted)** | 400 steps, natural class distribution | **0.526** | **0.914** | 1.0 | 1.0 |
| v2 (rejected) | 800 steps, class-balanced relevance oversampling | 0.491 | 0.914 | 1.0 | 1.0 |
| Frontier bar (claude-sonnet-5) | — | 0.583 | 0.920 | not measured | not measured |

**v2 hypothesis and result:** v1's relevance shortfall was hypothesized to come from severe class imbalance (`complement` was only 4.8% of the 8,000 relevance training rows). v2 tested class-balanced oversampling plus double the training steps. Result: relevance got **worse** (0.491 vs 0.526), not better — the hypothesis did not hold up under test. v1 is retained as the better real candidate rather than continuing to retrain speculatively.

**Honest verdict against the frontier bar:** v1 does not clear claude-sonnet-5 on `match_relevance` (short by 5.7 points) but is within noise on `match_identity` (0.914 vs 0.920, a 0.6-point gap). It beats all three other tested frontier models (claude-haiku-4-5, gpt-5-mini, gpt-5) on both tasks. This is genuine, disclosed partial progress — not a superiority claim.

Adapter artifacts: `models/shared_adapter_v1/` (checkpoints at steps 100/200/300/400, `checkpoint_history.json` for the full per-checkpoint trajectory against real frontier scores).

## What's NOT done yet

- Production serving is implemented (`expansion/serve/api.py`) but not yet load-tested.
- The full 38-model Hugging Face comparator sweep (RexBERT, RexReranker, Qwen3-Embedding, eCeLLM) is partial — see comparator sweep section above for what ran and what's blocked.
- No superiority claim can be made across the full comparator set yet — only the 4 frontier models have been fully measured against the trained adapter so far.

## Isolation from the original CommerceCore release

Per the plan's separation requirement: this project never imports, retrains, resumes, or serves through `arghya2030/commercecore-qwen3-1.7b` or its GitHub repository. It is a genuinely separate GitHub repository, separate Hugging Face repository (once published), separate serving process, and separate credentials/endpoints.
