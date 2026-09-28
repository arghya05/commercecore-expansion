#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux && -n "${RUNPOD_POD_ID:-}" ]] || exit 2
cd /workspace/commercecore_expansion_research_v2
export HF_HOME=/workspace/hf_cache
export HF_XET_HIGH_PERFORMANCE=1
PY=/workspace/cc-research-venv/bin/python
mkdir -p research_v2/results/followup_20260928
OUT=research_v2/results/followup_20260928
exec > >(tee -a "$OUT/campaign.log") 2>&1
trap 'printf "%s\n" "$?" > "$OUT/exit_code.txt"' EXIT
"$PY" -m unittest discover -s research_v2/tests -v
"$PY" -m research_v2.experiment verify
"$PY" -m research_v2.runpod.exposure_audit
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B',revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
PY
SNAP=/workspace/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
for interface in legacy chat; do
  "$PY" -u -m research_v2.run_local --backend hf --snapshot "$SNAP" --batch-size 8 --interface "$interface" --per-task 3 --output "$OUT/base_${interface}_smoke"
  "$PY" -u -m research_v2.run_local --backend hf --snapshot "$SNAP" --batch-size 8 --interface "$interface" --output "$OUT/base_${interface}"
  "$PY" -u -m research_v2.run_local --backend hf --snapshot "$SNAP" --batch-size 8 --interface "$interface" --adapter models/shared_adapter_v1 --tasks relevance identity --output "$OUT/shared_${interface}"
  "$PY" -u -m research_v2.run_local --backend hf --snapshot "$SNAP" --batch-size 8 --interface "$interface" --adapter models/understand_adapter_v1 --tasks extraction --output "$OUT/understand_${interface}"
done
"$PY" -m research_v2.runpod.analyze_followup
"$PY" -m research_v2.runpod.serving_probe --snapshot "$SNAP"
