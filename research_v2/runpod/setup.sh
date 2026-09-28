#!/usr/bin/env bash
# Only RunPod performs installation, verification and compilation.
set -euo pipefail
if [[ "$(uname -s)" != Linux || -z "${RUNPOD_POD_ID:-}" ]]; then
  printf '%s\n' 'Run this script on RunPod, not on the laptop.' >&2
  exit 2
fi
cd /workspace/commercecore_expansion_research_v2
mkdir -p research_v2/runs research_v2/remote_artifacts
python -m venv --system-site-packages /workspace/cc-research-venv
/workspace/cc-research-venv/bin/python -m pip install \
  'transformers==4.57.6' 'peft==0.17.1' 'accelerate==1.10.1' \
  'huggingface_hub==0.36.0' 'pandas==2.3.3' 'pyarrow==21.0.0'
/workspace/cc-research-venv/bin/python -m pip freeze > research_v2/remote_artifacts/environment.txt
/workspace/cc-research-venv/bin/python - <<'PY'
import json,os,platform,torch
from pathlib import Path
if not torch.cuda.is_available():
    raise SystemExit('RunPod GPU is unavailable. No local/CPU model fallback.')
report={'pod_id':os.environ['RUNPOD_POD_ID'],'python':platform.python_version(),
        'torch':torch.__version__,'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(0),
        'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory}
Path('research_v2/remote_artifacts/hardware.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
