# Anonymous ACM drafts

Entry points: `../commercecore_acm.tex` (SIGIR-style sigconf) and
`../commercecore_wsdm.tex` (alternative with review line numbering).
The shared manuscript is `body.tex`; references are `references.tex`.
The original detailed preprint remains available separately.

Build and verify **on RunPod only**:

```sh
python3 paper/build_acm.py
python3 paper/verify_acm_package.py
```

See [BUILD.txt](BUILD.txt) for dependencies. The unmodified official ACM class
and source/license materials are included. `build_manifest.json` records output,
source and archive hashes; `package_verification.json` records compilation from
an isolated archive and matching delivered PDF text.

Passing format checks does not establish a strong scientific contribution.
Matched baselines, controlled ablations, independent human validation and serving
measurements remain incomplete. These are drafts for alternative venues; no
submission has occurred.
