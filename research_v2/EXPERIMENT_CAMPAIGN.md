# Follow-up experiments for the CommerceCore Expansion paper

Prepared 2026-09-28. Status: full research design with a completed, scoped first
follow-up; see [measured results](../paper/evidence/followup_summary.md).
All computation, downloads, tests, training, analysis and PDF builds run on
RunPod. Local activity is limited to document editing, connections, transfers
and publishing. The previous pilot pod was terminated. The user authorized
an additional $25 ceiling for the follow-up. The scoped run covers paired
base/adapter development comparisons, NuExtract text extraction, and a bounded
HTTP serving prototype; it does not complete the entire design below.

The original CommerceCore repository and commercecore-qwen3-1.7b weights remain
unchanged. This campaign lives under research_v2, with new versioned outputs.
The accompanying README restoration retains the previous Expansion problem
statement, section order, examples and model/paper links, correcting facts in place.

## Questions and order of execution

| ID | Research question | Experimental comparison | Gate and deliverable |
|---|---|---|---|
| E1 | Does adaptation improve each task under a fair interface? | Pinned base versus existing shared/specialist adapters on identical development IDs; original task prompts and a common interface reported separately | Budget and remote protocol tests; paired raw outputs, per-class metrics and interface-validity table |
| E2 | How competitive are the models against appropriate alternatives? | Commerce model, extraction specialist, compact supervised baselines and current Claude/OpenAI endpoints on the same task-specific contracts | Verified interfaces/revisions, budget, licenses; matched comparison table, never mixed historical populations |
| E3 | Does scenario breadth outperform paraphrase repetition? | Many scenarios/few paraphrases versus fewer scenarios/more paraphrases with fixed rows and training-token budget | Independent human quality gate and new scenario-disjoint data; three seeds and paired unseen-scenario outcomes |
| E4 | Does the proposed development-guided task sampling preserve weak tasks? | Fixed mixture versus established loss-based sampling versus proposed retention-constrained sampling, with shared/specialist controls | Quality gate, frozen update rule, identical data/compute; learning curves, worst-task retention and compute cost |
| E5 | What is the actual quality/latency/cost trade-off? | Reference GPU inference versus a compatible optimized server; BF16 first, quantization only as a separately scored condition | Working endpoints and quality checks; repeated load runs, failures, p50/p95/p99 and cost per successful completion |
| E6 | Do selected results generalize? | Frozen candidate cohort on untouched tests plus separately specified official public benchmark protocols | All development choices frozen; single final scoring stage, confidence intervals and explicit wins/ties/losses |

E1 and a small E5 measurement come first. E2 requires additional model-specific
integration; API comparisons consume a separate token budget. E3/E4 require
annotation and training and must not be presented as covered by a small inference
pilot. A $25 or $100 authorization is a spending ceiling, not a promise to finish
this entire campaign within that amount.

## E1: repair the comparison before expanding it

Use the existing 1,495-row development suite: 595 relevance, 500 identity and
400 extraction. The completed untuned-base run is a reference only when the
prompt, decoding, scorer and row hashes match. Otherwise rerun the base.

Evaluate the shared match adapter on relevance/identity and the Understand
adapter on extraction. Preserve their actual task interfaces; do not silently
wrap a completion-trained adapter in an unvalidated chat prompt. Compare each
adapter with the base under the identical interface to estimate adaptation
effects. Report the common-chat condition separately as an interface robustness
experiment. Freeze any prompt selection using development data alone, including
the number of prompts tried for every candidate.

Before interpreting these legacy-model scores, audit new development/test texts
and entity/scenario groups against every available historical adapter training
source. A new split is not necessarily unseen by an already-trained adapter.
If overlap exists, report these evaluations as retrospective diagnostics and
construct a separately versioned clean evaluation population before final
testing. Unknown pretraining contamination remains a limitation.

Primary classification metric: macro-F1; also accuracy, balanced accuracy,
per-class precision/recall/F1, confusion matrix, invalid outputs and abstentions.
Primary extraction metric: normalized exact joint accuracy; also exact and
normalized field F1, null handling and schema validity. Keep legacy permissive
scores only as labeled secondary historical results. Failed/oversized/invalid
rows stay in the denominator. Never use substring matching for label scoring.

Functional relation and technical compatibility are excluded from clean
generalization claims until new independently reviewed scenario-disjoint data
exists. Historical examples can diagnose their interfaces but cannot replace
that evaluation.

## E2: fair compact, commerce and frontier comparators

Candidate screening list, not a frozen or completed cohort:

- Qwen/Qwen3-1.7B at the existing immutable revision, with the selected adapter
  revisions recorded separately.
- NingLab/eCeLLM-S, using its documented task interface. Investigate the earlier
  malformed-output run on development examples before drawing quality conclusions.
- numind/NuExtract-2.0-2B for extraction, using its documented text/schema
  interface; screen the newer NuExtract release before cohort freeze.
- A supervised compact relevance classifier and Ditto-style identity matcher,
  trained on the same permissible training partition after quality approval.
- GLiNER2-family extraction baseline, with schema and any tuning budget disclosed.
- One current high-capability and one cost-oriented endpoint from each of
  Anthropic and OpenAI, subject to budget. Resolve exact available model IDs and
  prices when preparing the paid run; old manuscript names are not identities.

For every candidate record repository/endpoint, immutable revision where
available, license, tokenizer/processor, prompt/schema, precision, decoding,
input/output limits, request date and returned model ID. Review any required
remote model code before enabling it. An unavailable snapshot is a recorded
limitation, not an invented revision. Do not truncate evidence differently
across candidates. Report native-interface token counts and cost separately.

All candidates for a task receive the same input evidence and IDs. Equivalent
task instructions may have documented model-native serialization; identical
strings are not a substitute for a valid interface. Fix output normalization
before final testing. Provide the same development prompt-selection budget.
Log billed attempts and retries; do not selectively repair incorrect predictions.
Malformed outputs count as failures, with interface diagnostics shown separately.

If API cost forces a smaller sample, freeze one stratified subset before any
API outputs are inspected and rerun all comparison candidates on that subset.
Label its estimand and sampling weights; do not compare subset API scores with
full-suite adapter scores. The wider suite remains a separate experiment.

## E3: scenario breadth versus paraphrase volume

This is a proposed contribution to test, not a certified novel method. Prior
commerce instruction tuning already exists. The narrower hypothesis is that,
at fixed budget and audited text quality, scenario breadth improves unseen
scenario performance more than repeated descriptions of the same scenarios.

First construct and review a scenario taxonomy with explicit labels, product
constraints and sufficient evidence. Assign entire scenario families to
train/development/test before generating text. Counterfactual siblings must stay
together. Test scenarios must never be used as generation demonstrations.

Target design after a feasibility/annotation gate: 1,200 training rows per arm,
balanced over declared labels and matched over categories/difficulty. Breadth
arm: 300 scenario families x 4 paraphrases. Repetition arm: 60 scenario families
x 20 paraphrases, drawn as a nested balanced subset. Use the same generator,
generation policy and quality thresholds. Match non-padding training-token
budgets and report row-length distributions and unique-token exposure.
Freeze the final feasible counts before training; disclose any deviation from
these targets. Do not fabricate families to satisfy counts.

Use three paired seeds (17, 29, 43), identical initialization/configuration and
checkpoint schedule. Hold out at least 100 development and 200 test scenario
families if feasible; multiple descriptions of a family are not independent
statistical observations. Estimate the unseen-family effect with cluster-aware
resampling. Provide category/label breakdowns and confidence intervals. Assess
statistical precision on development data before committing the test size.

Independent raters see randomized text without generator, condition or expected
label. They assign label, evidence sufficiency and naturalness; disagreements
are adjudicated and preserved. Follow the existing training_gate minimums:
at least 100 reviewed examples/task, two independent raters, adjudicated label
support >=98%, naturalness >=95%, zero unresolved critical issues, and matching
manifest/evidence hashes. Report uncertainty and agreement, not just threshold
passing. Evaluation gold also needs independent annotation. Never deliberately
train an unsafe/failed-quality arm to create a more favorable comparison.

## E4: adaptive sampling and forgetting

Treat this as a second hypothesis, with lower priority than completing E1-E3.
Compare uniform/task-fixed sampling, a declared loss-based sampler, and a
retention-constrained sampler that increases exposure to weak tasks based only
on development metrics. Include a fixed shared-adapter and task-specialist
control. Finalize the update equation, smoothing, sampling bounds, update
frequency and tuning budget in a machine-readable config before execution;
this document does not claim that controller is implemented yet.

Use the same training corpus, initial checkpoint, optimizer, effective batch
tokens, non-padding token budget and three paired seeds. Evaluate at fixed
token-budget intervals (10% increments) rather than differently sized epochs.
Charge evaluation overhead to the cost comparison. Primary selection metric:
equal-weight mean of task primary metrics; development tie-break: worst-task
score, then earlier checkpoint. Preregister any non-inferiority margin using
application needs and development precision before inspecting final tests.

Report each task's best-to-final decline, full curves, worst-task score and a
separate general-capability retention set. A horizontal claim requires one
shared parameter state to pass across tasks; a collection of specialists is
reported as an adapter system. Test results never trigger an automatic retrain
in the same experiment. Development adaptation stops at the approved budget;
failure to beat a comparator is a publishable outcome, not an infinite loop.

## E5: production-serving measurements

Start with the existing supported endpoints on one on-demand GPU. Select the
instance from live total pricing and availability; the prior RTX 3090 rate is
historical, not a new quote. Compare a second GPU only if budget allows. Record
GPU model, memory, runtime/container versions, engine, precision and adapter
loading strategy. Unsupported runtime/adapter combinations are excluded with
their reason recorded.

Separate three questions: cold start/load/download time; warm single-request
latency; concurrent sustained service. Use a fixed public development workload
with task and input/output-length strata. No final-test outputs are needed for
systems tuning. Compare systems on identical request traces and score their
outputs; a faster configuration that changes quality must show that trade-off.

Warm-up: 50 completed requests, recorded but excluded from steady-state metrics.
Measure concurrency 1, 4, 8 and 16 for at least 500 completions per cell and
three repetitions, within budget. Report small-sample p99 as exploratory;
require >=10,000 completions per selected configuration for a strong tail claim.
Add open-loop arrival-rate sweeps around observed capacity to expose queuing;
log offered, accepted, completed, failed, timed-out and outstanding requests.
Use explicit deadlines and bounded queues. Preserve request timestamps so
closed-loop throughput cannot conceal overload or coordinated omission.

Report end-to-end p50/p95/p99, throughput, goodput at preregistered latency and
quality targets, error rate, peak GPU memory and cold/warm adapter switching.
Report time-to-first-token only when actual streaming timestamps exist.
Quantization is a separate condition with a repeated quality evaluation.

Compute cost from measured billable GPU/storage duration and applicable rates,
then reconcile against available invoice records. Report observed utilization
and idle/provisioning costs. Cost per 1,000 schema-valid completions and per
1,000 correct labeled completions are distinct. Token-price assumptions alone
are not a production cost result. API latency is measured from the same remote
client with region/network limitations disclosed; it is not hardware parity.

## E6: statistical analysis and final test

Freeze candidates, prompts, exact revisions, scorer, selection policy and all
development reports through the existing experiment.freeze contract. Screen
historical exposure before this freeze. The current 3,000-row test is usable
only for candidates for which the required independence is established.
Use one recorded test access for the whole frozen cohort. Preserve test outputs
and failures; do not repair a model or select checkpoints from this stage.

Use paired cluster bootstrap intervals (10,000 resamples, fixed analysis seed),
grouping relevance by query, identity by connected entity group and extraction
by item/family. Use scenario-family groups for synthetic tasks. Report the
grouping proxies and their limitations. Account separately for training-seed
variation; do not treat seed x example pairs as independent observations.
Predeclare comparison families and use Holm adjustment for confirmatory multiple
comparisons. Report effect sizes and confidence intervals, including ties and
losses; statistical significance is not practical significance.

Public benchmark claims require publisher splits, metrics and permissible
training protocols, run as separately named evaluations. A local subset of
ESCI/WDC/ABO does not become an official benchmark result by using the dataset
name. If official coverage is incomplete, restrict the paper's claims accordingly.

## Execution, artifacts and stopping rules

Before provisioning: record approved total ceiling, GPU/storage reserve, any API
cap, and a provider-supported shutdown mechanism verified for the chosen pod.
Recheck available balance/rates and make a conservative total-cost estimate.
Allow only one active experiment pod initially. Do not leave a GPU idle while
waiting for human annotation. A stopped pod may still incur storage charges.

On RunPod: restore the isolated suite; verify source/model hashes; execute the
protocol/scorer tests; run an interface smoke test before each full model run.
Test fixtures cover opposite-label collisions, invalid JSON/null extraction,
overlength inputs, duplicate predictions and missing/error rows. Audit remaining
implementation gaps before claiming the campaign is executable end to end.

Each run saves its configuration, source revision, dataset and model hashes,
environment, raw predictions, errors, token counts, timing, quality report and
cost ledger. Use new run IDs; do not overwrite historical artifacts. Journal
and checkpoint for restart, preserving attempts and excluding no failed rows.
Stop on budget exhaustion, unverified data, failed protocol tests or a persistent
runtime fault; do not spend through repeated blind retries.

Transfer evidence back, verify hashes remotely, then terminate only the campaign
pod. Generate paper tables, README numerical updates and HF-card summaries from
the same evidence registry. Preserve the restored README structure. Compile and
verify PDFs on RunPod before publishing. Report incomplete experiments as such.

No claimed new experiment result is added to the manuscript before its raw
outputs and evaluation artifacts exist. Neither novelty nor a win over every
frontier model nor conference acceptance is guaranteed by this design.

## Primary references and existing implementation

- [eCeLLM paper](https://arxiv.org/abs/2402.08831) and
  [eCeLLM-S model card](https://huggingface.co/NingLab/eCeLLM-S), checked 2026-09-28.
- [NuExtract-2.0-2B model card](https://huggingface.co/numind/NuExtract-2.0-2B),
  checked 2026-09-28; use its text extraction interface.
- [Existing publication review](../paper/REVIEW_AND_PUBLICATION_STATUS.md).
- [Protocol and training gates](experiment.py), [current development runner](run_local.py),
  [RunPod lifecycle policy](runpod/README.md).

The current runner is pinned to Qwen3-1.7B and uses a chat interface. It must not
be treated as a ready-made eCeLLM/NuExtract evaluator or a validated legacy
completion-interface comparison. Training controllers, comparator integrations,
human annotation and load measurements remain work to execute under this design.
