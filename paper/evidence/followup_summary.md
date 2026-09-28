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
