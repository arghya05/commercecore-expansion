# CommerceCore Expansion paper

Author: [Arghya Mukherjee](https://orcid.org/0009-0008-3423-8574) · [arghya05@gmail.com](mailto:arghya05@gmail.com) · ORCID: 0009-0008-3423-8574

This folder contains the research manuscript and its complete LaTeX source, following the same build-and-verify pattern as [the original CommerceCore paper](https://github.com/arghya05/commercecore/tree/main/paper). The PDF is a preprint. It is not an accepted conference paper and has not been submitted to arXiv.

- [Manuscript PDF](commercecore_expansion_paper.pdf)
- [LaTeX source](commercecore_expansion_paper.tex)
- [Evidence macros](evidence/numbers.tex) — every number in the paper is computed by [`build_evidence.py`](build_evidence.py) from a recorded evidence file, never asserted in prose alone.

## Build and verify

From this directory, with Python 3.10+ and [Tectonic](https://tectonic-typesetting.github.io/) installed:

```sh
python3 build_evidence.py
tectonic commercecore_expansion_paper.tex
```

`build_evidence.py` uses only the Python standard library, performs no model inference or paid API calls, and reads only files already committed to this repository (`reports/frontier_comparison_2026-09-27/report.json`, `reports/match_baseline_2026-09-27/report.json`, `reports/understand_baseline_2026-09-27/report.json`, `models/shared_adapter_v1/checkpoint_history.json`, `reports/shared_adapter_v2_checkpoint_history.json`).

## Evidence scope

The manuscript reports two full training/evaluation runs (referred to in-text as "the trained adapter" and "a second training run"), a frontier-model comparison against four models with resolved model snapshots, and a synthetic-data audit trail with three rounds of independent review. It does **not** claim superiority over frontier models: the primary result explicitly does not clear the strongest tested comparator on the relevance subtask, and this is reported as the honest outcome rather than reframed.

The broader Hugging Face domain/commerce model comparator sweep specified in the project's own release gate ([`../../capability_expansion/RELATED_MODEL_COMPARISON_MATRIX.md`](../../capability_expansion/RELATED_MODEL_COMPARISON_MATRIX.md)) is only partially complete at the time of this manuscript; see [`../expansion/manifests/benchmark_cells.json`](../expansion/manifests/benchmark_cells.json) for exactly which comparators ran, which did not, and why.

## Template

`neurips_2026.sty` is unmodified from [the official 2026 author kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip), copied from the original CommerceCore paper folder. The manuscript uses `preprint`, preserving author attribution without claiming conference acceptance.
