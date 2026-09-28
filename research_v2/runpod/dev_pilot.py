"""Run the development-only GPU pilot without reading training or test labels."""
import json
import os
import sys
from pathlib import Path
from research_v2.core import digest
from research_v2.execution import require_runpod
from research_v2.experiment import read_rows
from research_v2.run_local import main as run


def main():
    require_runpod()
    root=Path(__file__).resolve().parents[2]
    suite=root/'research_v2/work/suite_v2'
    manifest=json.loads((suite/'manifest.json').read_text())
    rows=read_rows(suite/'dev.jsonl')
    if manifest['status']!='MACHINE_GATES_PASS' or digest(rows)!=manifest['partition_contract_hashes']['dev']:
        raise SystemExit('Development contract failed its saved machine-gate/hash check')
    snapshot=Path('/workspace/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
    job=os.environ.get('CC_JOB_ID','qwen_base_gpu_smoke_v2')
    if not job.replace('_','').replace('-','').isalnum():raise SystemExit('Invalid job identifier')
    sys.argv=['run_local','--backend','hf','--device','cuda','--snapshot',str(snapshot),
              '--dev',str(suite/'dev.jsonl'),'--output',str(root/'research_v2/runs'/job)]
    if os.environ.get('CC_FULL_DEV')!='1':sys.argv.extend(['--per-task','3'])
    run()


if __name__=='__main__':main()
