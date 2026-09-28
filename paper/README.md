# CommerceCore Expansion research papers

Current conference-format drafts:
- [Anonymous SIGIR-style PDF](CommerceCore_Expansion_SIGIR_Draft.pdf)
- [Alternative WSDM-style PDF](CommerceCore_Expansion_WSDM_Draft.pdf)
- [ACM LaTeX source ZIP](CommerceCore_Expansion_ACM_Source.zip)
- [Source and dependency guide](acm/README.md)

The [detailed historical preprint](CommerceCore_Expansion_Paper.pdf) retains the
same preprint format as the earlier CommerceCore paper. Its
[source package](CommerceCore_Expansion_arXiv_Source.zip) and
[section sources](sections/) preserve the longer retrospective analysis.
The new ACM drafts add a separate, strict development baseline run on RunPod.
Neither document claims universal frontier superiority or conference acceptance.
No conference or arXiv submission has occurred.

## One numerical source of truth

[The audit](evidence/audit_results.json) supplies historical paper macros and
historical values in the GitHub README and four Expansion model cards.
[The new development report](../research_v2/results/runpod_pilot_20260928/qwen_base_gpu_dev_v2/report.json)
supplies the separate untuned-base development table. Different datasets,
scorers and parameter states are not presented as paired comparisons.

Run **on RunPod**, from the repository root:

```sh
python3 -m unittest discover -s paper -p 'test_audit_evidence.py' -v
python3 paper/audit_evidence.py
python3 paper/sync_public_results.py
python3 paper/sync_public_results.py --check
python3 paper/build_acm.py
python3 paper/verify_acm_package.py
```

The ACM build uses XeLaTeX, the unmodified official ACM class and the fonts in
[BUILD.txt](acm/BUILD.txt). It checks references, fonts, overflow, page limits and
identifying text/metadata. Package verification compiles the ZIP in isolation
and compares its PDF text with the delivered PDFs.

[Build hashes](acm/build_manifest.json), [package verification](acm/package_verification.json),
[publication review](REVIEW_AND_PUBLICATION_STATUS.md) and
[generated numerical summary](evidence/public_results.json) accompany the drafts.

Historical runtime, data and reports remain intact. The baseline source-tree
equivalence is recorded in [revision_equivalence.json](evidence/revision_equivalence.json).
The audit cannot reconstruct missing predictions or retroactively hide an
exposed test. GPU development inference does not fill the remaining production
serving, controlled-training, comparator or human-validation gaps.

This public repository identifies the author; use a properly anonymized review
artifact. The venue drafts are alternatives, not simultaneous submissions.
Check the intended cycle's current rules before submitting.

Author: [Arghya Mukherjee](https://orcid.org/0009-0008-3423-8574).
The original CommerceCore repository and model release are unchanged.
