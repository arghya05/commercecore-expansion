"""Fetch authorized Expansion cards remotely; token arrives only on SSH stdin."""
import json,sys
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from research_v2.execution import require_runpod
require_runpod()
token=sys.stdin.read().strip()
if not token:raise SystemExit('No Hugging Face credential supplied')
api=HfApi(token=token)
root=Path(__file__).resolve().parents[1]
dest=root/'work/hf_card_backups';dest.mkdir(parents=True,exist_ok=True)
records={}
for short in ['understand-v1','match-v1','functional-relation-v1','relevance-singletoken-v1']:
    name='commercecore-expansion-'+short
    repo='arghya2030/'+name
    try:
        info=api.model_info(repo)
        path=Path(hf_hub_download(repo,'README.md',revision=info.sha,token=token,local_dir=dest/name))
        raw=path.read_text()
        (dest/(name+'.md')).write_text(raw)
        records[repo]={'revision':info.sha,'private':info.private}
        print('Saved card: '+repo+' at '+info.sha)
    except Exception as exc:
        print('Model-card read failed for '+repo+': '+type(exc).__name__)
        raise SystemExit(1)
(dest/'revisions.json').write_text(json.dumps(records,indent=2)+'\n')
