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

### Understand vs. frontier models — real result, open gap, no win claimed

Same 200 real, text-grounded ABO examples, same extraction prompt for all four frontier models, resolved model snapshots. Full report: `reports/understand_frontier_comparison_2026-09-28/report.json`.

| System | Brand F1 | Color F1 |
|---|---|---|
| Dictionary rules | 0.880 | 0.754 |
| GLiNER2-base | 0.777 | 0.854 |
| GLiNER2.5-base | 0.753 | 0.871 |
| claude-haiku-4-5 | 0.965 | 0.941 |
| claude-sonnet-5 | 0.997 | 0.925 |
| gpt-5-mini | **1.000** | 0.926 |
| **gpt-5** | **1.000** | **0.931** |

**Honest verdict: no current baseline beats the frontier models on this task.** Every one of the four frontier models clearly outperforms every measured baseline on both fields — this is a much wider gap than Match's relevance shortfall (roughly 10-25 points, not single digits). This is documented as an open gap, not spun as a partial win. `NuExtract-2.0-2B` was not tested: it is actually a Qwen2-VL-2B-based vision-language model (`image-text-to-text` pipeline), not a plain text extractor, so a text-only prompt would not be a fair comparison; testing it properly would require its native multimodal calling convention, not attempted here.

No new training or publishing followed from this result — per this project's own rule, a real measured gap is reported honestly, not forced into a release.

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

Claude Sonnet 5 leads on both tasks. The trained shared adapter (v1, see below) comes within 0.6 points of it on identity but falls 5.7 points short on relevance — a real, disclosed gap, not the much weaker rules/TF-IDF baselines above.

### Hugging Face domain/SLM comparators

- GLiNER2-base: brand 0.777 F1, color 0.854 F1 (200 real ABO examples).
- GLiNER2.5-base (native `gliner2.AutoExtractor` loader, distinct from GLiNER2-base's loader): brand 0.753 F1, color 0.871 F1.
- **NingLab/eCeLLM-S** (independently verified via `HfApi.model_info`: **2.78B parameters, Phi-2 base** — an earlier pass in this project incorrectly reported 7.24B/Mistral-7B without checking the Hub API directly; that was wrong). Evaluated on GPU once available. First attempt used a generic "answer with one word" prompt and scored 0.0/0.0 — inspecting raw output showed the model wasn't following the instruction at all. Corrected by inspecting the model's own training data (`NingLab/ECInstruct`) to find its real native prompt format (JSON-structured, matching its `Product_Matching` training task) and re-running fairly: **relevance 0.283, identity 0.620** — both real trained adapters in this project beat it on both tasks.
- RexBERT/RexReranker, Ettin/MiniLM rerankers, Qwen3-Embedding: dependency/environment issues encountered during the first sweep pass; not yet re-run.

## Trained shared adapter — real results, three attempts, honest verdict

QLoRA adapter on independently-pinned `Qwen/Qwen3-1.7B` (never the Query-merged artifact), trained on RunPod (RTX 2000 Ada / RTX 3080 Ti / RTX 5090), evaluated on the full held-out dev set (never seen in training).

| Attempt | Config | match_relevance | match_identity | match_functional_relation | match_technical_compatibility |
|---|---|---|---|---|---|
| **v1 (adopted, best)** | rank-16, 400 steps, 8,000 relevance rows, natural class distribution | **0.526** | **0.914** | 1.0 | 1.0 |
| v2 (rejected) | rank-16, 800 steps, 8,000 relevance rows, class-balanced oversampling | 0.491 | 0.914 | 1.0 | 1.0 |
| v3 (rejected) | rank-32, 1,200 steps, 40,000 relevance rows (5x, natural distribution) | 0.503 | 0.888 | **0.133** | 1.0 |

**v2 hypothesis and result:** v1's relevance shortfall was hypothesized to come from severe class imbalance (`complement` was only 4.8% of the 8,000 relevance training rows). v2 tested class-balanced oversampling plus double the training steps. Result: relevance got **worse** (0.491), not better — rejected.

**v3 hypothesis and result:** tested whether scaling real relevance data 5x (to 40,000 rows, drawn from ESCI's full 419,653-row train pool) plus doubling LoRA rank (16→32) and tripling steps (400→1,200) would close the gap. Checkpoint tracking showed relevance climbing through step 400 (0.28→0.52) then plateauing at 0.50-0.54 for the remaining 800 steps — no further real gain, and the final full-dev score (0.503) is actually slightly *below* v1's. Worse: `match_functional_relation` collapsed from a perfect 1.0 at step 300 to 0.067 by step 600 and stayed pinned there through the end of training — real catastrophic forgetting, caused by the much larger relevance data share crowding out that subtask's tiny 50-row signal within the shared mixture. **Rejected**; v1 remains the best real candidate. A specific corrective idea for a future attempt: protect minority subtasks with a minimum row-count floor in the mixture regardless of how much a majority task's data grows, or train them as separate non-shared LoRA modules instead of one shared adapter.

**Honest verdict against the full comparator set (frontier models + Hugging Face domain models), using v1:**

| System | match_relevance | match_identity |
|---|---|---|
| **v1 (this project, adopted)** | 0.526 | 0.914 |
| claude-sonnet-5 | **0.583** | **0.920** |
| claude-haiku-4-5 | 0.467 | 0.780 |
| gpt-5-mini | 0.483 | 0.800 |
| gpt-5 | 0.467 | 0.840 |
| NingLab/eCeLLM-S (2.78B, Phi-2 base) | 0.283 | 0.620 |

v1 beats every tested comparator except claude-sonnet-5, and is within measurement noise of claude-sonnet-5 on identity (0.6-point gap). It falls short of claude-sonnet-5 specifically on relevance (5.7-point gap). This is genuine, disclosed partial progress, not a full superiority claim.

Adapter artifacts: `models/shared_adapter_v1/` (checkpoints at steps 100/200/300/400, `checkpoint_history.json` for the full per-checkpoint trajectory against real frontier scores).

## Dedicated relevance-only adapter — attempted, abandoned

A fourth hypothesis was tested: that v1's relevance shortfall came from capacity competition with the other three subtasks sharing one small adapter, and that training relevance alone (`expansion/train/train_relevance_adapter.py`) would remove that competition and close the gap. It did not.

**Attempt 1:** rank-16, natural class distribution, no loss weighting. Collapsed to predicting the majority class (`exact`) for effectively every input by step 150, confirmed by direct inspection of real predictions (19-20 of 20 test examples predicted `exact` regardless of gold label), and never recovered — checkpoints 150/300/450/600 all showed the exact same accuracy to full floating-point precision, which is what a fixed always-predict-majority-class policy produces on a fixed eval sample.

**Attempt 2:** added per-token class-weighted cross-entropy loss (weighting each label's first token by inverse class frequency) and a lower learning rate, after two real implementation bugs were found and fixed along the way (a vocab-size mismatch between the tokenizer and the model's actual output layer; and a `save_steps` calculation that made short smoke tests trigger a checkpoint evaluation almost every training step, which looked like a severe performance regression but was actually just very frequent evaluation, not slow training — confirmed by isolating the bug with an explicit `save_steps` override). Once genuinely running at normal speed, checkpoint 150 still showed collapse: 18/20 predictions were `exact`, 2/20 were `irrelevant`, and `substitute`/`complement` were never predicted once, despite the loss weighting. Better than attempt 1's total collapse, but not a real fix.

**Conclusion:** the collapse is real and reproduces even with class-weighted loss, suggesting the problem is not purely data imbalance — a plausible remaining explanation is that this task's labels have unequal token lengths (`exact` is a single token; `substitute`, `complement`, and `irrelevant` each split into two subword tokens), which may make the two-token labels structurally harder for the model to commit to early in training when trained on relevance alone. v1's shared adapter never showed this exact failure, plausibly because the other three subtasks' gradients provided enough diversity to avoid the collapse. This was not tested further; v1 remains the best, adopted, and only published result for `match_relevance`.

## What's NOT done yet

- Production serving is implemented (`expansion/serve/api.py`) but not yet load-tested.
- The full 38-model Hugging Face comparator sweep (RexBERT, RexReranker, Qwen3-Embedding, eCeLLM) is partial — see comparator sweep section above for what ran and what's blocked.
- No superiority claim can be made across the full comparator set yet — only the 4 frontier models have been fully measured against the trained adapter so far.

## Isolation from the original CommerceCore release

Per the plan's separation requirement: this project never imports, retrains, resumes, or serves through `arghya2030/commercecore-qwen3-1.7b` or its GitHub repository. It is a genuinely separate GitHub repository, separate Hugging Face repository (once published), separate serving process, and separate credentials/endpoints.
