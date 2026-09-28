# CommerceCore Expansion paper

Author: [Arghya Mukherjee](https://orcid.org/0009-0008-3423-8574) · [arghya05@gmail.com](mailto:arghya05@gmail.com) · ORCID: 0009-0008-3423-8574

This folder contains the research manuscript, its complete LaTeX source, and the evidence audit, following the same build-and-verify pattern as [the original CommerceCore paper](https://github.com/arghya05/commercecore/tree/main/paper) — entirely independent of it: no shared LaTeX, no shared evidence files, no shared repo. The PDF is a public preprint. It is not an accepted conference paper and has not been submitted to arXiv by this workflow.

- [Manuscript PDF](CommerceCore_Expansion_Paper.pdf)
- [Main LaTeX source](commercecore_expansion_paper.tex)
- [arXiv source ZIP](CommerceCore_Expansion_arXiv_Source.zip)
- [Evidence macros](evidence/numbers.tex) — every number in the paper is computed by [`build_evidence.py`](build_evidence.py) from a recorded evidence file, never asserted in prose alone
- [Build manifest](evidence/build_manifest.json) — sha256 hashes of every source file, the built PDF, and the source zip

## Build and verify

From this directory, with Python 3.10+ and [Tectonic](https://tectonic-typesetting.github.io/) installed:

```sh
python3 build_paper.py
```

This recomputes every evidence macro from recorded evidence files (`build_evidence.py`, standard library only — no model inference or paid API calls), compiles the PDF with Tectonic, fails the build on any undefined reference/citation, missing glyph, or overfull box, then sha256-verifies and packages the LaTeX source into `CommerceCore_Expansion_arXiv_Source.zip`.

The source is also compatible with a conventional PDFLaTeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error commercecore_expansion_paper.tex
```

## Evidence scope

The manuscript reports, in full, every training attempt run across three subtask families in this project, not only the adopted configurations:

- **Match** (relevance, identity, functional relation, technical compatibility): three shared-adapter training runs, a real data-split-stratification bug found and fixed after the first run's publication, and five distinct, falsifiable hypotheses tested for the relevance shortfall (all rejected) plus a dedicated functional-relation retraining attempt (real, partial progress, not a win).
- **Understand** (brand/color extraction): one dedicated adapter, the first attempted for this subtask, which beats every tested frontier model on both fields.
- A completed Hugging Face domain-model comparator sweep (RexBERT, RexReranker, Qwen3-Embedding, eCeLLM-S) across every task where each model is architecturally applicable, with exclusions justified by direct empirical checks where not.
- A serving-API load test that found and fixed one real concurrency bug.

Every evidence file referenced by `build_evidence.py` is committed to this repository under `reports/` and `models/*/checkpoint_history.json` — see [`build_evidence.py`](build_evidence.py) for the exact list. The manuscript does **not** claim blanket superiority over frontier models: two of three Match subtasks and the single-token relevance retry explicitly do not clear the strongest tested comparator, and this is reported as the honest outcome rather than reframed or omitted.

## Template

`neurips_2026.sty` is unmodified from [the official 2026 author kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip). The manuscript uses `preprint`, preserving author attribution without claiming conference acceptance.
