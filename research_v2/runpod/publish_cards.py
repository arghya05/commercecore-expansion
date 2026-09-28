"""Update only the four authorized Expansion README cards from generated drafts."""
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from research_v2.execution import require_runpod
require_runpod()
root=Path(__file__).resolve().parents[2]
token=sys.stdin.read().strip()
if not token:raise SystemExit('No Hugging Face credential supplied')
api=HfApi(token=token)
backups=json.loads((root/'research_v2/work/hf_card_backups/revisions.json').read_text())
receipt_path=root/'paper/evidence/hf_publication_sync.json'
receipt={'updated_utc':datetime.now(timezone.utc).isoformat(),'operation':'README.md only; weights and visibility unchanged','repositories':{}}
def signature(info):
    return {f.rfilename:{'blob_id':f.blob_id,'size':f.size,
            'lfs_sha256':f.lfs.sha256 if f.lfs else None}
            for f in info.siblings if f.rfilename!='README.md'}
for repo,previous in backups.items():
    if not repo.startswith('arghya2030/commercecore-expansion-'):
        raise SystemExit('Repository outside authorized Expansion scope')
    short=repo.rsplit('/',1)[1]
    draft=root/'paper/model_cards'/(short+'.md')
    content=draft.read_bytes();before=api.model_info(repo,files_metadata=True)
    if before.sha!=previous['revision']:raise SystemExit('Model repository changed since review: '+repo)
    try:
        result=api.upload_file(path_or_fileobj=content,path_in_repo='README.md',repo_id=repo,
            repo_type='model',parent_commit=before.sha,
            commit_message='Synchronize model card with audited paper and qualified results')
        after=api.model_info(repo,revision=result.oid,files_metadata=True)
        if signature(before)!=signature(after):raise RuntimeError('Unexpected non-card file change')
        if before.private!=after.private:raise RuntimeError('Unexpected repository visibility change')
        cached=hf_hub_download(repo,'README.md',revision=result.oid,token=token,
                              cache_dir='/workspace/hf_card_verification')
        if Path(cached).read_bytes()!=content:raise RuntimeError('Published card content mismatch')
        receipt['repositories'][repo]={'previous_revision':before.sha,'revision':result.oid,
            'readme_sha256':hashlib.sha256(content).hexdigest(),'non_card_files_unchanged':True,
            'visibility_unchanged':True,'published_content_verified':True}
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        print('Verified README update: '+repo+' at '+result.oid,flush=True)
    except Exception as exc:
        print('Publication failed for '+repo+': '+type(exc).__name__,file=sys.stderr)
        raise SystemExit(1)
