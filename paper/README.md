# CommerceCore Expansion research paper

**Canonical manuscript:** [CommerceCore Expansion: Auditing Small-Model Adaptation for Product Matching and Attribute Extraction](CommerceCore_Expansion_Paper.pdf).

This revision reconstructs the experiments from the repository and preserved local logs, adds new offline analyses, and corrects claims that the saved evidence does not support. Its results and limitations supersede the earlier paper and conflicting historical README/model-card narratives. No new model training, paid API evaluation, or GPU serving experiment was performed for this revision.

- [PDF](CommerceCore_Expansion_Paper.pdf)
- [Main LaTeX source](commercecore_expansion_paper.tex) and [section sources](sections/)
- [Self-contained LaTeX ZIP](CommerceCore_Expansion_arXiv_Source.zip)
- [Offline evidence audit](audit_evidence.py) and [machine-readable findings](evidence/audit_results.json)
- [Publication-readiness review](REVIEW_AND_PUBLICATION_STATUS.md)
- [Source/PDF/package hashes](evidence/build_manifest.json)
- [Standalone package verification](evidence/package_verification.json)

The public PDF keeps the same NeurIPS preprint style as the earlier [CommerceCore paper](https://github.com/arghya05/commercecore/tree/main/paper), as requested. It does not claim NeurIPS acceptance, ACM submission readiness, or completed arXiv submission. SIGIR/WSDM submission requires the relevant anonymous ACM format and a shorter main manuscript; the review document explains the more important outstanding experimental work.

## What the revision adds

The manuscript includes precise task definitions; historical versus current split counts; QLoRA objectives and configuration details; related-work positioning; full available loss and checkpoint plots; relevance and functional confusion matrices; extraction TP/FP/FN tables; comparison-validity analysis; synthetic-scenario and extraction-text overlap checks; scorer counterexamples; paired-correctness uncertainty bounds; and an analysis of censored serving measurements. Appendices preserve protocol details and a concrete controlled-study design, clearly labeled as prospective work.

The clearest positive observation is extraction: brand F1 1.000 and color F1 0.955 on the recorded common 200-row sample under permissive field matching. Brand ties the best API result; color is 1.38 percentage points above the strongest measured API color baseline. Two evaluation texts overlap training and paired field outputs are missing, so neither clean generalization nor statistical superiority is established.

The matching results do not support a universal frontier-model win. Adapter and API relevance/identity samples differ; a compatibility scorer accepts `compatible` inside `incompatible`; 25/28 functional test rows reuse training scenarios; and the functional test set was used for checkpoint selection. No completed inference requests occur in the retained load-test CSVs, so speed and cost advantage remain unmeasured.

## Reproduce the analysis and paper

From the repository root, using Python 3.9+:

```sh
python3 -m unittest discover -s paper -p 'test_*.py'
python3 paper/audit_evidence.py
python3 paper/build_paper.py
python3 paper/verify_package.py
```

The audit and tests use the standard library and make no inference or API calls. The build additionally needs [Tectonic](https://tectonic-typesetting.github.io/), which may download LaTeX packages on first use. It fails on undefined citations/references, missing glyphs, or overfull boxes, then packages all LaTeX, style, and generated table/plot dependencies. The package verifier also needs Poppler's `pdftotext` and `pdfinfo`; it checks hashes, compiles the ZIP in a temporary directory, and compares the resulting PDF text. Ordinary underfull-box layout notices are distinct from missing or overflowing content.

To compile only the packaged manuscript, extract the source ZIP and run:

```sh
tectonic --keep-logs commercecore_expansion_paper.tex
```

The ZIP supports standalone typesetting; regenerating the statistical analysis requires the full repository. The legacy ZIP filename contains `arXiv` for link continuity, but server-side arXiv compilation and moderation have not been performed.

## Evidence preservation

The originally observed baseline is commit `c7deee76caa8c22058088d501befc32c57513d41`. Its source tree is identical to public commit [`2af4b2c914284920d2d674e68a7293d304bc8fef`](https://github.com/arghya05/commercecore-expansion/commit/2af4b2c914284920d2d674e68a7293d304bc8fef), as recorded in [revision equivalence](evidence/revision_equivalence.json). The inventory in [evidence/archive](evidence/archive/) records the original non-paper files and hashes. Runtime, dataset, and result files are checked against those hashes; the README is allowed to evolve. Additional archived local artifacts supply trainer states, adapter configurations, the original v1 dev split, and the functional checkpoint history. The prior manuscript and macros remain archived for traceability.

Generated numbers are reconstructed where the saved artifacts permit it; prose configuration claims are supported by source inspection. This is reproducibility of the retrospective analysis, not an exact replay of every historical training run. Missing raw predictions, incomplete runtime/version metadata, and edited historical trainer scripts are disclosed in the paper.

Author: [Arghya Mukherjee](https://orcid.org/0009-0008-3423-8574). The original CommerceCore repository and its model release are separate and are not modified by this paper revision.
