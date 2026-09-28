---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
language:
- en
pipeline_tag: text-generation
tags:
- ecommerce
- product-matching
- qlora
- fine-tuned
---

# Functional-relation specialist adapter

This card describes **arghya2030/commercecore-expansion-functional-relation-v1**. The weights are unchanged; this revision corrects
and synchronizes the documented evidence. The tables identify which observations
belong to this adapter and which belong to other parameter states.

## Audited results — 2026-09-28

The historical adapter results below use the original scoring rules and local
evaluation subsets. They are not official leaderboard results. Dataset, metric
and exposure qualifications are part of each result.

| Model / task | Recorded result | Evaluation and interpretation |
|---|---|---|
| Extraction specialist | Brand F1 **1.000**; color F1 **0.955** | 200 rows; permissive field matching; 2 evaluation texts overlap training |
| Shared adapter: relevance | Accuracy **0.527** | 1,000 rows; logged v1 rescore |
| Shared adapter: identity | Accuracy **0.916** | 3,500 rows; logged v1 rescore; legacy substring scorer |
| Shared adapter: functional relation | Accuracy **0.692** (9/13) | Reassigned development set; prior exposure and scorer limitations |
| Shared adapter: compatibility | Legacy accuracy **1.000** (8/8) | Scorer accepts opposite labels; prior exposure; not validated compatibility accuracy |
| Functional specialist | Accuracy **0.821** (23/28); macro-F1 **0.780** | Exposed test; 25/28 rows reuse training scenarios; 9 checkpoints inspected |
| Letter-label relevance specialist | Accuracy **0.5115**; macro-F1 **0.436** | 4,000 rows; balanced accuracy 0.414; complement recall 0.200 |

The shared v1 rescore values **0.527 / 0.916** replace
the earlier headline pair 0.526 / 0.914. The earlier pair survives only in the
prior manuscript; it is not supported by a retained full evaluation vector.
Historical v2 identity also equals 0.914, but is a different run.

Extraction ties the strongest recorded API brand result. Its color F1 point
estimate is **1.38 percentage points** above the strongest
recorded API color baseline (Haiku, F1 0.941176). This does **not** establish a
clean or statistically significant superiority claim: scoring is permissive,
two texts overlap training, and paired field predictions are missing.

Adapter and API relevance/identity scores use different samples. The functional
specialist and API results also use different evaluation sets. No cross-sample
frontier ranking is claimed. Historical API names are report labels, not verified
current endpoint snapshots. The retained eCeLLM functional run produced zero
strictly valid responses; its interface failure does not establish model inferiority.

The four adapters are distinct parameter states. Their combined task coverage
is not evidence that one shared model performs all tasks well. Human naturalness
validation, controlled forgetting/retention ablations, and universal retailer
suitability have not been demonstrated.

## Separate RunPod development baseline

This is **untuned Qwen3-1.7B**, revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, not one of the four trained adapters.
It was evaluated on a new development suite with strict scoring, so its values
must not be subtracted from the historical adapter scores as an adaptation gain.

| Task / metric | Development result | Rows |
|---|---|---|
| Relevance accuracy / macro-F1 | 0.3361 / 0.2540 | 595 |
| Identity accuracy / macro-F1 | 0.3340 / 0.2865 | 500 |
| Extraction exact brand / color F1 | 0.7250 / 0.7181 | 400 |
| Extraction normalized brand / color F1 | 0.7275 / 0.7914 | 400 |
| Extraction normalized joint accuracy | 0.5800 | 400 |

All 1,495 rows have retained raw outputs. The evaluation loop took
218.90 seconds on an RTX 3090, using batch size one,
bfloat16, greedy decoding, thinking disabled and a 64-token output cap.
This is sequential offline inference time, excluding model loading and provisioning;
it is **not** a concurrent serving benchmark or total campaign cost.
No new model training or frontier API calls were performed in this pilot.

## Serving, data quality and publication limits

The historical load-test CSVs contain **zero completed inference requests**.
Production throughput, tail latency and cost advantage remain unmeasured.
Neither historical token-cost assumptions nor the new offline inference timing
establish a production service-level claim.

New training remains blocked until independent human label/naturalness review
passes the quality gate. The new test suite has not been scored or used for
model selection. Machine checks and family proxies do not replace human review.

The paper is an empirical audit with a new development baseline and evaluation
infrastructure. It does not establish a new foundation-model architecture,
universal frontier superiority, or conference acceptance.

- [Detailed historical preprint](https://github.com/arghya05/commercecore-expansion/blob/main/paper/CommerceCore_Expansion_Paper.pdf)
- [Current anonymous SIGIR-style draft](https://github.com/arghya05/commercecore-expansion/blob/main/paper/CommerceCore_Expansion_SIGIR_Draft.pdf)
- [Alternative WSDM-style draft](https://github.com/arghya05/commercecore-expansion/blob/main/paper/CommerceCore_Expansion_WSDM_Draft.pdf)
- [ACM LaTeX source package](https://github.com/arghya05/commercecore-expansion/blob/main/paper/CommerceCore_Expansion_ACM_Source.zip)
- [Historical evidence audit](https://github.com/arghya05/commercecore-expansion/blob/main/paper/evidence/audit_results.json)
- [New development report and raw outputs](https://github.com/arghya05/commercecore-expansion/tree/main/research_v2/results/runpod_pilot_20260928/qwen_base_gpu_dev_v2)
- [Publication gaps and related work](https://github.com/arghya05/commercecore-expansion/blob/main/paper/REVIEW_AND_PUBLICATION_STATUS.md)

## Follow-up RunPod experiments — 2026-09-28

New **development-only** measurements; no retraining, final-test scoring or new frontier API calls. Base and adapters receive identical examples within each interface. All full paired runs use batch size eight, BF16, greedy decoding and a 64-token output cap.

| Task | Interface | Base primary score | Adapter primary score | Base / adapter invalid rate |
|---|---|---:|---:|---:|
| relevance | legacy | 0.0000 | 0.3333 | 1.0000 / 0.0000 |
| relevance | chat | 0.2568 | 0.2665 | 0.0000 / 0.0000 |
| identity | legacy | 0.0000 | 0.8617 | 1.0000 / 0.0000 |
| identity | chat | 0.3110 | 0.8886 | 0.0000 / 0.0000 |
| extraction | legacy | 0.0000 | 0.9150 | 1.0000 / 0.0000 |
| extraction | chat | 0.5825 | 0.9025 | 0.0000 / 0.0000 |

Primary scores are macro-F1 for relevance/identity and normalized exact joint accuracy for extraction. Legacy uses the original training completion prompts; chat uses the common strict task prompt. Interface differences are part of the experiment, not hidden preprocessing.

NuExtract-2.0-2B, using its native text/schema interface on the same 400 extraction rows: normalized joint accuracy **0.4675**, normalized brand/color F1 **0.5667 / 0.8283**, invalid rate **0.0000**. Its immutable revision, schema and raw outputs are recorded. This is one comparator/interface, not a comprehensive leaderboard.

The known-source audit found zero exact-text matches for these development rows; identity entity-level overlap remains unknown. These retrospective adapter diagnostics do not establish an independent final-test win. Raw outputs, per-class scores and exploratory paired correctness intervals are retained.

### HTTP serving prototype

| Batch limit | Concurrency | Completed req/s | Correct fraction | Mean run p95 (s) | GPU-rate estimate / 1,000 valid requests |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.01 | 0.9000 | 0.116 | $0.00678 |
| 1 | 4 | 8.96 | 0.9000 | 0.466 | $0.00682 |
| 1 | 8 | 9.06 | 0.9000 | 0.933 | $0.00674 |
| 8 | 1 | 8.57 | 0.9000 | 0.124 | $0.00713 |
| 8 | 4 | 29.11 | 0.9050 | 0.148 | $0.00210 |
| 8 | 8 | 40.56 | 0.9000 | 0.287 | $0.00151 |

One RTX 3090; same-pod loopback HTTP client/server; shared identity adapter; three repeats of 200 requests per cell; 50 warmup requests excluded per engine/repeat. Batching waits at most 5 ms, with an eight-token output cap. All responses and errors are retained. These are closed-loop, single-task prototype measurements, not an optimized production server or open-loop SLO test. Tail estimates are exploratory.

Cost uses the $0.22/hour GPU quote during measured windows. It excludes setup, idle time, storage and network; it is not a full invoice or frontier-cost advantage. No new architecture, controlled training improvement, universal frontier win or conference acceptance is established.

A slow batch-one run was interrupted for runtime reasons before completing; its journal is preserved and excluded from full-run tables. The planned large serving campaign was narrowed to 200 requests/cell/repeat; broader hardware, public-benchmark, human-review and controlled-training experiments remain open.

[Raw follow-up evidence](https://github.com/arghya05/commercecore-expansion/tree/main/research_v2/results/followup_20260928) · [Experiment protocol](https://github.com/arghya05/commercecore-expansion/blob/main/research_v2/EXPERIMENT_CAMPAIGN.md)

The base produced invalid outputs for every legacy-prompt row; those zero scores diagnose interface failure, not absent task knowledge. The chat comparison is therefore essential. Identity has only five connected evaluation components, so its cluster-bootstrap interval is suppressed. The shared adapter predicts no complement labels in either interface. Batch-eight base scores are separate from the earlier batch-one pilot.

## Use and scope

Use this adapter with its historical task-specific prompt interface; see the
[GPU inference guide](https://github.com/arghya05/commercecore-expansion/blob/main/docs/INFERENCE.md). The new development
pilot uses an untuned base and a separate chat interface. The original
[CommerceCore release](https://huggingface.co/arghya2030/commercecore-qwen3-1.7b)
is a separate project and is not modified by this work.
