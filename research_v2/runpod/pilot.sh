#!/usr/bin/env bash
# Execute ONLY on the selected RunPod. Does not create or terminate a paid pod.
set -euo pipefail
if [[ "$(uname -s)" != Linux || -z "${RUNPOD_POD_ID:-}" ]]; then
  printf '%s\n' 'Refusing execution: this job must run inside RunPod.' >&2
  exit 2
fi
: "${CC_REMOTE_PYTHON:?Set CC_REMOTE_PYTHON to the prepared RunPod Python executable}"
: "${CC_JOB_ID:?Set a unique job identifier}"
if [[ ! "$CC_JOB_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  printf '%s\n' 'CC_JOB_ID must contain only letters, numbers, underscores and hyphens.' >&2
  exit 2
fi
cd /workspace/commercecore_expansion_research_v2
cc_result_dir="research_v2/runs/$CC_JOB_ID"
mkdir "$cc_result_dir"
exec > >(tee -a "$cc_result_dir/job.log") 2>&1
trap 'cc_exit_code=$?; printf "%s\n" "$cc_exit_code" > "$cc_result_dir/exit_code.txt"' EXIT
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$cc_result_dir/started_utc.txt"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$cc_result_dir/gpu.csv"
"$CC_REMOTE_PYTHON" -m pip freeze > "$cc_result_dir/dependencies.txt"
"$CC_REMOTE_PYTHON" -m unittest discover -s research_v2/tests -v
"$CC_REMOTE_PYTHON" -m research_v2.experiment verify
cc_snapshot_path=$("$CC_REMOTE_PYTHON" - <<'PY'
import torch
from huggingface_hub import snapshot_download
if not torch.cuda.is_available():
    raise SystemExit('CUDA is required; no CPU inference fallback is allowed.')
print(snapshot_download('Qwen/Qwen3-1.7B',
    revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'))
PY
)
timeout --signal=TERM --kill-after=60s 20m \
  "$CC_REMOTE_PYTHON" -u -m research_v2.run_local \
  --backend hf --device cuda --snapshot "$cc_snapshot_path" \
  --per-task 3 --output "$cc_result_dir/base_development_smoke"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$cc_result_dir/finished_utc.txt"
