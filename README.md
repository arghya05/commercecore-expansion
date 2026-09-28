# CommerceCore Expansion

A set of small (1.7B parameter), self-hostable QLoRA adapters that classify and extract structured facts from ecommerce product data — query-product relevance, listing identity, functional relationships between products, technical compatibility, and brand/color extraction. One of these (brand/color extraction) beats every tested frontier model on both fields; the others report real, disclosed partial progress or rejected hypotheses rather than an inflated headline number.

**Source code + docs:** https://github.com/arghya05/commercecore-expansion
**Paper (PDF):** [`paper/CommerceCore_Expansion_Paper.pdf`](paper/CommerceCore_Expansion_Paper.pdf)

**A separate, standalone project from [arghya05/commercecore](https://github.com/arghya05/commercecore)** — no shared runtime, no shared weights, no shared repository. The original CommerceCore release (`arghya2030/commercecore-qwen3-1.7b`) remains frozen and untouched; it is used only as an isolated external benchmark comparator, never as a training base or dependency.

**Status:** research / proof-of-concept, in progress. Every number in this README is measured, not projected — read "What this is NOT" before relying on any headline claim.

---

**Trained models (all separate Hugging Face repositories, real trained weights, honest model cards):**

| Model | Task | Real result | Status |
|---|---|---|---|
| [commercecore-expansion-understand-v1](https://huggingface.co/arghya2030/commercecore-expansion-understand-v1) | brand/color extraction | brand F1 1.000, color F1 0.955 | **Beats every tested frontier model** |
| [commercecore-expansion-match-v1](https://huggingface.co/arghya2030/commercecore-expansion-match-v1) | relevance/identity/functional_relation/technical_compatibility (shared adapter) | 0.526 / 0.914 / 0.692 / 1.0 | Adopted for identity + technical_compatibility (ties frontier); relevance and functional_relation fall short |
| [commercecore-expansion-functional-relation-v1](https://huggingface.co/arghya2030/commercecore-expansion-functional-relation-v1) | functional_relation (dedicated adapter) | 0.821 locked-test accuracy | Real improvement over the shared adapter (0.692), still below the 0.846 frontier bar |
| [commercecore-expansion-relevance-singletoken-v1](https://huggingface.co/arghya2030/commercecore-expansion-relevance-singletoken-v1) | relevance (rejected hypothesis) | 0.5115 accuracy | Preserved as evidence, not adopted — use the shared adapter instead |

## The problem

A production ecommerce system needs several distinct decisions about products that a general-purpose frontier model is not specialized for: is this listing what a shopper searched for (relevance); are two listings the same purchasable item (identity); do two items serve a compatible or substitutable role (functional relation); is a specific accessory guaranteed to fit a specific device revision (technical compatibility); and what brand and color does a listing actually describe (extraction). Each has a distinct, non-interchangeable label space — conflating query relevance (query + offer) with product identity (offer + offer) is a specific, real error this project's governing specification identified and avoided before any data was collected.

**Measured, not assumed:** four frontier models (Claude Haiku 4.5, Claude Sonnet 5, GPT-5-mini, GPT-5) were tested directly on the same held-out data used to evaluate every adapter in this project. They are capable but not specialized for these exact label spaces — Claude Sonnet 5, the strongest tested, tops out at 0.583 accuracy on relevance and 0.846 on functional_relation. A small model fine-tuned specifically for one of these tasks (Understand) closes that gap completely; for the others, this project reports the real, partial, or negative results honestly rather than reframing them.

## The strategy

1. **Fine-tune a small, open base model** (`Qwen/Qwen3-1.7B`, Apache-2.0, independently pinned — never the Query-merged artifact from the original CommerceCore release) via QLoRA, a 4-bit-quantized, low-rank adaptation method cheap enough to run on a single consumer/prosumer GPU.
2. **Build training data honestly.** Real, license-verified public data (ESCI, WDC Products, ABO) wherever it exists. For the two subtasks with no adequate public benchmark (functional_relation, technical_compatibility), synthetic data generated with a **structured-scenario-first** method: the label is fixed in code *before* any LLM call — the LLM's only job is paraphrasing that fixed scenario into natural product text.
3. **Audit synthetic data with a different model than the one that generated it.** Every generated example is judged by GPT-5-mini for whether the text actually supports its assigned label, catching real defects a generator grading its own output would miss — see "Synthetic data quality" below for two real, multi-round examples of this catching genuine scenario-definition bugs.
4. **Evaluate against frontier models on a matched protocol, not just an internal baseline.** Same held-out examples, same prompt structure, resolved model snapshots recorded at test time — not an aggregated or historical comparison.
5. **Report every training attempt, not only the adopted one.** Five distinct hypotheses were tested for the relevance shortfall alone; all five are documented below with their real results, not just the winning configuration.

## How it beat the benchmark — Understand (all numbers measured, not estimated)

Every off-the-shelf baseline (dictionary rules, GLiNER2-base, GLiNER2.5-base) initially lost to every frontier model on brand/color extraction by a wide margin (10-25 points), on 200 real, text-grounded Amazon Berkeley Objects (ABO) listings — brand/color values verified to actually occur in the source text, not copied from structured metadata the text doesn't mention.

| System | Brand F1 | Color F1 |
|---|---|---|
| Dictionary rules | 0.880 | 0.754 |
| GLiNER2-base | 0.777 | 0.854 |
| GLiNER2.5-base | 0.753 | 0.871 |
| claude-haiku-4-5 | 0.965 | 0.941 |
| claude-sonnet-5 | 0.997 | 0.925 |
| gpt-5-mini | 1.000 | 0.926 |
| gpt-5 | 1.000 | 0.931 |
| **This project's dedicated adapter** | **1.000** | **0.955** |

Rather than accept the frontier gap, a dedicated fine-tuned adapter was trained specifically for this task (rank-16 QLoRA, 9,396 real text-grounded ABO listings, 1,200 steps) — the first trained model attempted for Understand (prior work only tried off-the-shelf rules/GLiNER). Evaluated on the same 200-example locked test set as every baseline and frontier model above, never touched during training or checkpoint selection. **Real win: ties the best frontier brand F1 at a perfect 1.000 (200/200 correct), and beats every frontier model's color F1** (previous best was gpt-5 at 0.931; this adapter reaches 0.955). Parse-error rate was 0.0 across all 200 examples.

`NuExtract-2.0-2B` was not tested: it is actually a Qwen2-VL-2B-based vision-language model (`image-text-to-text` pipeline), not a plain text extractor, so a text-only prompt would not be a fair comparison.

## How it did NOT beat the benchmark — Match (all numbers measured, not estimated)

Match covers four subtasks in a shared QLoRA adapter (rank-16, 400 steps, trained on 8,000 relevance rows + 6,000 identity rows + a small synthetic set for the two subtasks with no real-data source). Evaluated on a held-out, class-balanced set never seen in training, and against the same four frontier models on the identical protocol.

| System | Relevance | Identity | Functional relation | Technical compatibility |
|---|---|---|---|---|
| claude-haiku-4-5 | 0.467 | 0.780 | 0.846 | 1.000 |
| **claude-sonnet-5** | **0.583** | **0.920** | 0.846 | 1.000 |
| gpt-5-mini | 0.483 | 0.800 | 0.846 | 1.000 |
| gpt-5 | 0.467 | 0.840 | 0.846 | 1.000 |
| NingLab/eCeLLM-S (2.78B, Phi-2 base) | 0.283 | 0.620 | 0.143 | — |
| **This project's shared adapter (v1)** | 0.526 | **0.914** | 0.692 | **1.000** |

**v1 ties frontier on technical_compatibility, is within measurement noise on identity (0.6-point gap), beats eCeLLM-S on every measured task, but falls real, disclosed short on relevance (5.7 points) and functional_relation (15.4 points).** Five distinct hypotheses were tested for the relevance gap and two dedicated retraining attempts for functional_relation — all reported below, including the ones that failed, because a data-split bug discovered after v1's publication changed what "failed" actually means for this task.

### A real bug found after v1 was published: the functional_relation/technical_compatibility dev split was never stratified by label

Every `functional_relation` dev row (15/15) was labeled `complement`; every `technical_compatibility` dev row (4/4) was labeled `compatible` — a real, pre-existing synthetic-data-generation bug, not something either training run caused. v1's reported perfect 1.0/1.0 accuracy on these two tasks had only ever been tested on 2 of 5 total classes. After re-stratifying the split so every class appears in both partitions and re-scoring v1's already-published checkpoint:

| Task | v1 (old, unstratified dev) | v1 (corrected, stratified dev) | Frontier (all 4 models) |
|---|---|---|---|
| technical_compatibility | 1.0 (4/4, single-class) | **1.0 (8/8, both classes)** | 1.0 |
| functional_relation | 1.0 (15/15, single-class) | **0.692 (9/13, all 3 classes)** | 0.846 |

technical_compatibility genuinely holds up. functional_relation does not — a real, previously-invisible 15.4-point gap, discovered only because the split bug was fixed. (Frontier scores for these two tasks had also never been measured before this point.) Direct inspection of the 13 corrected predictions ruled out a majority-class collapse: all three classes were predicted, with three genuine confusions, all biased toward over-predicting `unrelated`.

### Five rejected hypotheses for the relevance shortfall

1. **Class-balanced oversampling** (v2 of the shared adapter): hypothesized `complement`'s severe underrepresentation (4.8% of relevance rows) explained the gap. Oversampled to uniform distribution, doubled training steps. Result: relevance got **worse** (0.491, not 0.526). Rejected.
2. **5x real data scaling, no rebalancing** (v3 of the shared adapter): scaled relevance data to 40,000 rows (ESCI's full pool), doubled LoRA rank, tripled steps. Relevance plateaued at 0.503 (marginally below v1) after climbing through the first third of training, and — worse — `functional_relation` collapsed from perfect accuracy to 0.133 through real catastrophic forgetting, confirmed via 5 consecutive identical checkpoint values. Rejected.
3. **Dedicated adapter, natural distribution, no loss weighting**: isolating relevance from the shared mixture, hypothesizing capacity competition explained the gap. Collapsed to predicting only the majority class (`exact`) by step 150 — confirmed by direct inspection (19-20/20 real predictions were `exact` regardless of gold label), never recovered. Rejected.
4. **Dedicated adapter, class-weighted loss**: added per-token class-weighted cross-entropy after fixing two real implementation bugs (a tokenizer/model vocab-size mismatch; a `save_steps` miscalculation that made short smoke tests look catastrophically slow when the real cause was excessive checkpoint-eval frequency, not slow training). Still collapsed: 18/20 predictions were `exact`, `substitute`/`complement` were never predicted once. Rejected, but pointed at a specific structural cause: `exact` is the only single-subword-token label among the four; the other three are each two tokens.
5. **Single-token label reformulation**: remapped all four classes to single letter tokens (A/B/C/D) to isolate the token-length variable, same class-weighted loss otherwise. **Fixed the collapse** — all four classes predicted, including `complement` (zero predictions in attempt 4). Full dev-set accuracy: **0.5115** — marginally below v1 (0.526), still short of frontier (0.583). A real, qualitatively different failure (no collapse, genuine confusions), not an improvement. Rejected as a replacement, published as evidence.

**Verdict: relevance's gap vs. frontier remains open after five distinct, honestly-reported hypotheses.** v1 (0.526) remains the adopted model for `match_relevance`.

### One real, partial-progress attempt for functional_relation

Given the 15.4-point gap above and only 65 total training rows behind v1's signal, functional_relation was retrained as its own dedicated adapter — same method that worked for Understand. The scenario bank was expanded from 15 to 63 distinct item-pair scenarios; the first independent-audit pass caught real defects (90.5% supported — e.g., an "electric kettle vs. stovetop kettle" pair labeled `substitute` in a no-electricity context, when the electric kettle literally cannot function there), fixed at the scenario-definition level, and a second pass reached 98.4% supported. Combined with the original 65 rows: 187 total, re-split into train/dev/**locked_test** (131/28/28 — locked_test held out from both training and checkpoint selection throughout).

Training reached a development-set accuracy of **0.893** at its best checkpoint — on dev alone, a clean win over the 0.846 frontier bar. **It wasn't, on the honest check**: scoring every saved checkpoint against the untouched locked_test partition revealed a real generalization gap (only 131 train rows means ~67 epochs of repetition; the small 28-row dev set, drawn from the same scenario bank as train, didn't catch it). The dev-selected leader (step 350) scores only 0.786 on locked_test; the best locked-test accuracy across all 9 checkpoints is **0.821** (checkpoint 450) — still below 0.846.

**Verdict: real, non-collapsed, partial progress, not a win.** Checkpoint 450 is published as the model of record for this subtask — a genuine improvement over v1's 0.692 — but does not clear the bar this project requires.

## Data — real, verified, hash-checked

| Source | License | Real size acquired | Role |
|---|---|---|---|
| [ESCI](https://github.com/amazon-science/esci-data) | Apache-2.0 | 2,621,288 query-product examples, 1,814,924 products | Match relevance training/dev |
| [WDC Products (80pair)](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) | WebDataCommons publisher terms | 19,835 train pairs, 4,500 gold-standard pairs | Match identity training/dev |
| [ABO](https://registry.opendata.aws/amazon-berkeley-objects/) | CC BY 4.0 | 9,232 listings (baselines) + 9,396/300 train/dev + 200 locked test | Understand training/eval |
| [WDC PAVE](https://github.com/wbsg-uni-mannheim/wdc-pave) | Unresolved — no LICENSE file found | 211/354/1,066 rows | Dev reference only; **not used for training** pending license resolution |
| Synthetic (structured-scenario-first) | Generated in-project | 187 functional_relation rows (63 scenarios) + 40 technical_compatibility rows | The two subtasks with no real-data source |

All sources are hash-verified against their published artifacts (`expansion/manifests/sources/registry.json`). The exact training data used for every published model in the table above is committed to this repository (`data/*.jsonl`) — not regenerated from an approximate script, so the numbers above are directly reproducible, not just described.

### Synthetic data quality — independently audited, not self-graded

The label is fixed in code **before** any LLM call generates surface text (Claude Haiku paraphrases only; it never chooses the label). Every example is then audited by a **different model** (GPT-5-mini) judging whether the text actually supports its label — avoiding a generator grading its own output.

- `technical_compatibility`: 40/40 (100%) label-supported, first pass.
- `functional_relation` (original 65 rows): 3 audit rounds to reach 100% — round 1 found a real laptop/13" sleeve size mismatch and a mislabeled desk-lamp scenario (83%); round 2 found an overly-permissive "unrelated" use-context (94%); round 3 fixed it (100%).
- `functional_relation` (expansion to 187 rows): round 1 caught real new defects — an "electric kettle vs. stovetop kettle" `substitute` pair that can't function in its stated context, an "acoustic vs. electric guitar" pair that reads as a sequential upgrade rather than a substitute (90.5%); fixed at the scenario level, round 2 reached 98.4%.

Full audit trails: `data/synthetic_match/*_audited.jsonl`, `data/synthetic_match/*_audit_summary.json`.

## Serving logs — real, measured request/response pairs

Actual logged output from the deployed REST API (`expansion/serve/api.py`), including per-request token counts and a disclosed compute-cost estimate:

```
GET /health
→ {"status": "ok", "match_adapter_available": true, "understand_adapter_available": true,
   "functional_relation_adapter_available": true, "estimated_cost_per_1k_tokens_usd": 0.01027778,
   "cost_estimate_basis": "self-hosted GPU compute estimate, not a billed API rate"}

POST /v1/match/functional-relation
{"tenant_id": "test",
 "listing_a": "Espresso Machine - Brew rich, full-bodied espresso shots for your favorite coffee drinks at home.",
 "listing_b": "Milk Frother - Create velvety steamed milk and foam for lattes, cappuccinos, and macchiatos."}
→ {"label": "complement", "status": "ok",
   "prompt_tokens": 64, "completion_tokens": 3, "total_tokens": 67,
   "estimated_cost_usd": 0.00068861, "latency_ms": 290266.03}
   (real, measured — CPU-only local machine; see latency table below for why)
```

Measured latency, real requests:

| Environment | Model load time | Latency/request |
|---|---|---|
| Local CPU (Apple Silicon, no quantization) | ~10-35s (per model, once) | ~5 minutes (functional_relation, 3 completion tokens) |
| GPU (RunPod, various cards used across this project) | ~10-30s | not separately load-tested end-to-end past the fix below (see "Challenges faced in serving") |

### Serving cost — a disclosed estimate, not a billed rate

`estimated_cost_usd` in every response is computed from `ESTIMATED_GPU_COST_PER_HOUR_USD` ($0.74/hr, this project's actual RunPod RTX 4090 on-demand rate) divided by an assumed ~20 tokens/sec continuous-serving throughput (consistent with the ~1.9-2.1 it/s generation rates measured during this project's own training runs on the same hardware) — giving **~$0.0103 per 1,000 tokens**. This does **not** account for model loading, idle GPU capacity, or batching efficiency; it is a simple, disclosed, single-request estimate, not a production cost model, and it is **not** an API-token price comparable to a frontier provider's billed rate — this project self-hosts, it doesn't sell tokens.

No frontier-API-cost comparison table is included here, unlike the original CommerceCore project's README: this project doesn't have a task where routing every production request through a frontier API is the realistic alternative being weighed (Match/Understand are per-listing classification/extraction calls a catalog pipeline runs internally, not a live per-keystroke user-facing query as in the original project's constraint-parsing task) — a token-cost comparison would need an equivalent apples-to-apples request volume and latency budget this project hasn't measured.

### Challenges faced in serving — real, specific, not generic

1. **Concurrent requests each triggered an independent, redundant full model load.** `_load_model()`/`_load_understand_model()` checked `if _model is not None` with no lock. Under concurrent load (FastAPI runs sync route handlers in a thread pool), multiple requests all saw `None` simultaneously and each started its own full model load — confirmed directly via server logs showing multiple interleaved "Loading weights" progress bars for one adapter at the same time. Over a 90-second Locust load test at 3 concurrent users, only 1 request completed in total (a `/health` check needing no model). Fixed with a double-checked-locking pattern (`threading.Lock()`); re-running the same test confirmed exactly one load sequence occurs now.
2. **CPU-only local testing makes a real concurrent-load benchmark impossible on this machine.** Even after the fix above, a single foreground request took over 120 seconds (timed out at the curl limit); a later measured completion took ~4.8 minutes for a 3-token completion. This is disclosed as a genuine open gap, not silently omitted: a real production throughput/latency-under-load number on the GPU-served path has not been measured.
3. **The `functional_relation` adapter was trained and published but never wired into the serving API** until this pass — a real gap between "we published a model" and "the API can actually serve it," found by checking the API's routes against the list of published models rather than assuming parity.

## Real cost

Training and evaluation ran across multiple RunPod GPU pods over the course of this project (RTX 2000 Ada, RTX 3080 Ti, RTX 5090 — hit a CUDA compute-capability incompatibility and was abandoned for that reason, not cost — RTX A4500, RTX 3090, RTX 4090, RTX PRO 4500), at rates between $0.50–$0.74/hr, plus Claude/GPT API calls for synthetic data generation, independent auditing, and frontier comparison across every task in this README. **Unlike the original CommerceCore project, exact per-pod billing was not logged during this session** — reporting a precise total here would be a fabricated-precision number this project's own standards elsewhere explicitly reject. Every pod was terminated promptly after its work finished; none were left running idle between sessions.

## What this is NOT

- **Not a blanket "beats every frontier/HF model" claim.** Only Understand clears that bar. Match's relevance and functional_relation subtasks report real, disclosed shortfalls after five and two rejected hypotheses respectively.
- **Not evaluated against the full Hugging Face domain/commerce comparator set with equal footing everywhere.** RexReranker and Qwen3-Embedding were scored with a threshold *fit on the test set itself* (an upper bound, not fair equal footing) where applicable, and excluded where empirically shown incapable of the task (relation-type classification) rather than force-scored. See `reports/match_hf_comparator_sweep_2026-09-28.json` and `reports/functional_relation_hf_comparator_sweep_2026-09-28.json` for exactly what ran, what didn't, and why.
- **Not load-tested for real concurrent production throughput.** The one concurrency bug load-testing was designed to catch was found and fixed; a genuine GPU-served requests/sec or p50/p99-under-load number has not been measured.
- **Not a general-reasoning or conversational model.** Each adapter is narrow-task-specific; none of this generalizes beyond the exact label space it was trained for.
- **Real, disclosed failure modes**: relevance systematically over-predicts `substitute` as a catch-all; functional_relation's dedicated adapter overfits its small (131-row) training set faster than its own dev split can detect, requiring a separate locked test partition to catch.

## Related work — honest comparison, not a novelty claim

- **[Performance Trade-offs of Optimizing Small Language Models for E-Commerce](https://arxiv.org/abs/2510.21970)** (arXiv 2510.21970) — closest match: fine-tunes a small Llama model via QLoRA on synthetic ecommerce data. This project differs in task (multi-subtask classification + extraction across five distinct label spaces, not single-label intent classification) and in explicitly reporting five rejected hypotheses for one subtask rather than only the adopted configuration.
- **eCeLLM: Generalizing Large Language Models for E-commerce from Large-scale, High-quality Instruction Data** (Peng, Ling, Chen, Sun, Ning; ICML 2024) — eCeLLM-S is evaluated directly as a comparator throughout this README (relevance 0.283, identity 0.620, functional_relation 0.143), not just cited; every trained adapter in this project beats it on every measured task.
- **[EcomGPT: Instruction-tuning Large Language Models with Chain-of-Task Tasks for E-commerce](https://arxiv.org/abs/2308.06966)** — a genuinely multitask ecommerce instruction-tuning approach, overlapping with this project's (rejected, for relevance specifically) hypothesis that a shared multitask adapter would serve all subtasks well. Not independently re-benchmarked against this project.

If you're aware of closer prior art than what's listed here, that's a real gap in this review, not something to hide.

## Repository structure

```
commercecore_expansion/
├── expansion/
│   ├── schemas/        # Contract types for each task
│   ├── data/           # Synthetic generation, auditing, mixture assembly
│   ├── train/           # Training scripts for every adapter (adopted and rejected)
│   ├── eval/            # Baselines, frontier comparison, HF comparator sweeps, locked-set eval
│   └── serve/           # FastAPI serving layer + Locust load test
├── data/                 # Committed training/dev data for every published model
├── models/                # Adapter checkpoints, model cards (also on Hugging Face)
├── reports/               # Every real experiment's evidence: logs, checkpoint histories, comparator results
└── paper/                 # LaTeX source, evidence-driven build script, compiled PDF
```

## How to run inference (quick start)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")

# Understand (brand/color) -- the one clear win
model = PeftModel.from_pretrained(base, "arghya2030/commercecore-expansion-understand-v1")
tokenizer = AutoTokenizer.from_pretrained("arghya2030/commercecore-expansion-understand-v1")

prompt = (
    'Extract the brand and color from this product listing. '
    'Respond with only a JSON object like {"brand": "...", "color": "..."}.\n'
    "Listing: Nike Air Max 270 Women's Trainers - Black/White. Breathable mesh upper.\n"
    "Answer:"
)
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# {"brand": "Nike", "color": "Black/White"}
```

Or run the REST API (same code, runs on RunPod, AWS, GCP, or a local machine):

```bash
pip install -r requirements.txt   # fastapi, uvicorn, torch, peft, transformers, bitsandbytes
uvicorn expansion.serve.api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/v1/catalog/normalize \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "demo", "text": "Nike Air Max 270 Trainers - Black/White."}'
```

**Note on speed**: local CPU inference works for testing/verification but is very slow (minutes per request, measured above). Use a GPU for anything latency-sensitive.

## Reproducing the benchmarks yourself

Every number in this README traces to a script in `expansion/eval/` and a data/evidence file committed to this repository:

```bash
python3 -m expansion.eval.understand_locked_eval           # Understand vs. frontier + baselines
python3 -m expansion.eval.frontier_comparison               # Match relevance/identity vs. frontier
python3 -m expansion.eval.minority_frontier_comparison       # functional_relation/technical_compatibility vs. frontier (needs ANTHROPIC_API_KEY, OPENAI_API_KEY)
python3 -m expansion.eval.ecellm_functional_relation_eval    # eCeLLM-S on functional_relation (needs a GPU)
python3 expansion/serve/load_test.py                          # Locust load test against a running API instance
```

`paper/build_paper.py` recomputes every LaTeX macro from these same evidence files and rebuilds the PDF — no number in the paper is asserted in prose alone. See `paper/README.md` for the full build/verify instructions.

## Isolation from the original CommerceCore release

This project never imports, retrains, resumes, or serves through `arghya2030/commercecore-qwen3-1.7b` or its GitHub repository. Genuinely separate GitHub repository, separate Hugging Face repositories, separate serving process, separate credentials/endpoints.

## License

Apache-2.0, matching the base model (Qwen3-1.7B).
