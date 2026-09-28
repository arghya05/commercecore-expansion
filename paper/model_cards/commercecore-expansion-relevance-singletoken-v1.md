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

# Letter-label relevance specialist adapter

This card describes **arghya2030/commercecore-expansion-relevance-singletoken-v1**. The weights are unchanged; this revision corrects
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

## Use and scope

Use this adapter with its historical task-specific prompt interface; see the
[GPU inference guide](https://github.com/arghya05/commercecore-expansion/blob/main/docs/INFERENCE.md). The new development
pilot uses an untuned base and a separate chat interface. The original
[CommerceCore release](https://huggingface.co/arghya2030/commercecore-qwen3-1.7b)
is a separate project and is not modified by this work.
