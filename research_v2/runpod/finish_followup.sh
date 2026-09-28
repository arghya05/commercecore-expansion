#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux && -n "${RUNPOD_POD_ID:-}" ]] || exit 2
cd /workspace/commercecore_expansion_research_v2
export HF_HOME=/workspace/hf_cache
export HF_XET_HIGH_PERFORMANCE=1
PY=/workspace/cc-research-venv/bin/python
OUT=research_v2/results/followup_20260928
while pgrep -f '^bash research_v2/runpod/followup.sh$' > /dev/null; do sleep 10; done
[[ "$(cat "$OUT/exit_code.txt")" == 0 ]] || { echo 'Primary campaign failed; inspect logs.'; exit 1; }
"$PY" -u -m research_v2.runpod.nuextract_probe > "$OUT/nuextract.log" 2>&1
"$PY" -m research_v2.runpod.summarize_followup
"$PY" paper/sync_public_results.py
"$PY" paper/sync_public_results.py --check
"$PY" paper/build_acm.py > "$OUT/acm_build.log" 2>&1
"$PY" paper/verify_acm_package.py > "$OUT/acm_package.log" 2>&1
"$PY" paper/build_paper.py > "$OUT/preprint_build.log" 2>&1
echo 'FOLLOWUP_BUILDS_COMPLETE'
