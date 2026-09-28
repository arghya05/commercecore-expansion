"""Publish only the reviewed follow-up code, documents and evidence via GitHub.

Credential arrives on stdin and is never persisted or printed.
"""
import argparse
import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from research_v2.execution import require_runpod
from research_v2.core import file_sha

def main():
    require_runpod()
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');a=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    expected='bfd8e87b931d2a2ab75e5d184db931b9f86b23a7'
    changed=subprocess.check_output(['git','ls-files','-m','-o','--exclude-standard'],cwd=root,text=True).splitlines()
    paths=sorted({p for p in changed if p=='README.md' or p.startswith('paper/') or p.startswith('research_v2/')})
    for p in paths:
        if '/work/' in p or '/remote_artifacts/' in p or '__pycache__' in p or p.endswith(('.pyc','.aux','.out')):
            raise ValueError('Unexpected publication file: '+p)
    listing={p:{'bytes':(root/p).stat().st_size,'sha256':file_sha(root/p)} for p in paths}
    dest=root/'research_v2/work/followup_publication';dest.mkdir(parents=True,exist_ok=True)
    (dest/'files.json').write_text(json.dumps(listing,indent=2)+'\n')
    print(json.dumps({'files':paths,'bytes':sum(v['bytes'] for v in listing.values())},indent=2),flush=True)
    if not a.publish:return
    token=sys.stdin.read().strip()
    if not token:raise ValueError('Missing GitHub credential')
    prefix='https://api.github.com/repos/arghya05/commercecore-expansion/'
    def api(endpoint,data=None,method=None):
        req=urllib.request.Request(prefix+endpoint,data=json.dumps(data).encode() if data is not None else None,
            method=method,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'commercecore-followup'})
        try:
            with urllib.request.urlopen(req,timeout=90) as response:return json.load(response)
        except urllib.error.HTTPError as exc:raise RuntimeError('GitHub request failed: '+str(exc.code)) from None
    current=api('git/ref/heads/main')['object']['sha']
    if current!=expected:raise ValueError('GitHub main changed; review before publishing')
    tree=api('git/commits/'+current)['tree']['sha'];entries=[]
    for path in paths:
        raw=(root/path).read_bytes()
        blob=api('git/blobs',{'content':base64.b64encode(raw).decode(),'encoding':'base64'},'POST')['sha']
        entries.append({'path':path,'mode':'100644','type':'blob','sha':blob})
    newtree=api('git/trees',{'base_tree':tree,'tree':entries},'POST')['sha']
    commit=api('git/commits',{'message':'Add RunPod matched evaluations, NuExtract comparator and measured HTTP serving; update papers and public summaries','tree':newtree,'parents':[current]},'POST')['sha']
    # Recheck before advancing; force is always false.
    if api('git/ref/heads/main')['object']['sha']!=current:raise ValueError('Main changed during upload')
    api('git/refs/heads/main',{'sha':commit,'force':False},'PATCH')
    remote=api('git/trees/'+newtree+'?recursive=1')
    mapping={r['path']:r['sha'] for r in remote['tree']}
    if remote.get('truncated') or any(mapping.get(r['path'])!=r['sha'] for r in entries):raise ValueError('Published tree verification failed')
    receipt={'commit':commit,'parent':current,'tree':newtree,'files':listing,'verified_tree':True}
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('PUBLISHED '+commit,flush=True)

if __name__=='__main__':main()
