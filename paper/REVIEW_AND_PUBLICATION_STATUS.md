# Research review and publication status

Review date: 2026-09-28. This document concerns the completed manuscript revision and the evidence available for it, not a guarantee of conference acceptance.

The rewrite is substantially deeper than the previous expansion draft. It is an empirical small-model study with an executable retrospective audit. The current artifacts do **not** support presenting it as a new general commerce foundation model, a universal frontier-model winner, a new adaptive training algorithm, or a proven low-cost serving system. Stronger prose cannot substitute for the missing experiments below.

## Research positioning

The closest methods were inspected through their primary papers and, for eCeLLM and extraction comparators, model documentation. This is a focused review of relevant work, not an exhaustive search of every paper or Hugging Face repository.

| Prior work | Why it matters to this manuscript | Consequence for originality |
|---|---|---|
| [eCeLLM, ICML 2024](https://arxiv.org/abs/2402.08831) | Broad commerce instruction tuning with explicit generalization experiments | Multitask commerce tuning already exists; reproduce its relevant tasks or distinguish the contribution |
| [EcomGPT](https://arxiv.org/abs/2308.06966) | Commerce instruction construction through task transformations | Synthetic commerce instruction data alone is not new |
| [CASLIE](https://arxiv.org/abs/2410.17337) | Contextual captions and quality selection for commerce models | No multimodal or category-transfer superiority can be inferred from this text-only study |
| [Licardo and Tanković](https://arxiv.org/abs/2510.21970) | Small-model commerce adaptation plus measured optimization trade-offs | Low-cost QLoRA is prior art; actual hardware/precision measurements are needed for a systems contribution |
| [GLiNER2](https://arxiv.org/abs/2507.18546) | Compact schema-driven extraction/classification | Include task-tuned compact extraction baselines, not only off-the-shelf encoders |
| [Ditto](https://arxiv.org/abs/2004.00584) and [WDC Products](https://arxiv.org/abs/2301.09521) | Supervised matching and unseen-entity evaluation | The identity baseline and entity-disjoint evaluation remain incomplete |
| [NuExtract model card](https://huggingface.co/numind/NuExtract-2.0-2B) | Structured extraction supports text inputs | Its absent completed result is a coverage gap; a vision-language backbone does not justify exclusion |

The newly implemented work in this revision is the artifact audit: source verification, scenario/text overlap analysis, class-level metric reconstruction, real-scorer counterexamples, and a partial-identification calculation for paired binary correctness from aggregate counts. The paper does not claim these statistical or evaluation principles were invented here. It makes their concrete application and resulting corrections inspectable.

Potential original studies are specified in Appendix F: scenario diversity versus paraphrase volume at a fixed data budget; development-only adaptation with minority-task retention controls; and quality-constrained adapter scheduling against established serving implementations. These are research hypotheses, not completed methods or validated novelty claims.

## What must change before a strong SIGIR/WSDM empirical submission

| Priority | Evidence gap | Required experiment or artifact | Acceptance condition for the claim |
|---|---|---|---|
| P0 | API and adapter Match samples differ | Rerun all candidates on identical frozen IDs, gold, and evidence | Paired outputs and hashes exist; no cross-population score subtraction |
| P0 | Compatibility substring collision; permissive extraction | Strict ontology parser, normalized exact match, separately reported permissive metric | Opposite-label and partial-field scorer fixtures fail correctly; all raw outputs retained |
| P0 | Scenario reuse and post-training split reassignment | New scenario-disjoint train/dev/test groups; do not recycle the exposed test as final evidence | Group overlap zero; frozen test is opened only after development choices are fixed |
| P0 | Two extraction evaluation texts occur in training | New item/text/family-separated evaluation; rerun all extraction baselines | Same clean sample across candidates with per-row predictions |
| P0 | Functional checkpoint selected using test results | Predeclare development metric and tie-break, then evaluate once on new test | Test does not control retraining, ranking, stopping, or checkpoint selection |
| P1 | Sequential trials change several factors | Fixed-data, fixed-budget shared/specialist and label/weighting ablations with repeated seeds | Effect estimates and variance support the proposed mechanism |
| P1 | Broad relevance/identity baseline coverage incomplete | Unfine-tuned base, trained compact classifiers, Ditto-style matcher, and relevant commerce models | Task-appropriate interface, dev-only threshold/prompt selection, equal test contract |
| P1 | Naturalness and label validity judged automatically | Blinded stratified human annotation with explicit evidence sufficiency | Agreement, adjudication, and failure taxonomy reported |
| P1 | No public leaderboard-equivalent protocol | Execute official task splits/metrics or label local subsets unambiguously | Public-benchmark claims correspond to the publisher's protocol |
| P1 | No useful serving completion distribution | GPU cold/warm concurrent workload with sufficient completions, error accounting, quality checks, actual billing | Reproducible throughput, tail latency, memory, and cost per valid completion |
| P1 | Missing run identity/environment | Immutable base/adapter revisions, complete configs, dependency lock, GPU/runtime record | Each metric is traceable to exact data, scorer, model, and execution |

The offline audit cannot recover generations that were never retained, retroactively hide a test set, or measure a GPU service that was never benchmarked. Those items are disclosed as missing; they are not silently filled with estimates.

## Format and venue rules

The detailed historical preprint retains the earlier paper's style and attribution.
Separate anonymous ACM SIGIR-style and WSDM-style drafts now use the unmodified
official class. Formatting checks do not close the scientific gaps above.
The ACM drafts add the untuned-base RunPod development baseline; historical
adapter and frontier scores are not reclassified as that new experiment.

The checked [SIGIR 2027 full-paper rules](https://sigir2027.org/pages/submit-full.html) call for anonymous ACM `sigconf` formatting and a nine-page content limit excluding references; the content limit includes appendices. The checked [WSDM 2027 full-paper rules](https://www.wsdm-conference.org/2027/cffp.html) likewise require anonymous ACM review formatting and a nine-page main-content limit, with their stated exclusions. Its posted full-paper deadline has already passed as of this review. Confirm the intended future cycle and its rules before submission.

For an eventual submission, close the experimental gaps, refine the contribution, recheck the intended format, and prepare anonymized permitted artifacts. The expanded public preprint is useful for completeness but should not be uploaded unchanged to a nine-page full-paper track. No conference or arXiv submission has been made by this workflow.

## Completed checks for this revision

- Inspected the earlier public PDF and verified it matches the local reference copy; retained its preprint style.
- Audited original source/data/report files plus archived local evidence; generated reproducible tables and CSV plots.
- Added seven offline tests, including exhaustive small-case checks of the pairing bounds and execution of actual scorer counterexamples without loading models.
- Checked primary related-work papers and corrected the ABO citation and incomplete comparator interpretations.
- Compiled and visually reviewed the final PDF, and separately compiled its extracted LaTeX package (see build manifest for final artifact hashes).
- Kept historical runtime/data/results intact; the paper describes their limitations without rewriting experimental evidence.


## RunPod follow-up and synchronized publication

The 2026-09-28 remote pilot completed 1,495 strict development evaluations with
untuned Qwen3-1.7B. Raw outputs, model/config hashes, software and GPU records are
retained in `research_v2/results/runpod_pilot_20260928`. This is not retraining
or a matched adapter/API comparison. The final test has not been scored.

The README and four Expansion model-card drafts derive historical and new
development values from separately identified evidence. Shared v1 rescore is
0.527 relevance / 0.916 identity; earlier headline values are labeled as
prior-manuscript-only observations. Extraction brand ties the strongest recorded
API result; color has a 1.38 percentage-point historical margin, qualified by
permissive scoring, overlap and missing paired predictions.

The focused primary-source review was refreshed on 2026-09-28 for eCeLLM,
EcomGPT, CASLIE, the small-model optimization study and NuExtract documentation.
The NuExtract-2.0 card links to [NuExtract3](https://huggingface.co/numind/NuExtract3);
screen that newer release when freezing future extraction comparators. No
result against it is claimed. This is not an exhaustive novelty certification.
