"""Lightweight RunPod control plane. Never imports ML libraries or runs local tests."""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'work/runpod_pilot_state.json'


def credential():
    text = (Path.home()/'.runpod/config.toml').read_text()
    match = re.search(r'^\s*apikey\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not match:
        raise SystemExit('RunPod credential is not configured')
    return match.group(1)


def query(document, variables=None):
    key = credential()
    url = 'https://api.runpod.io/graphql?' + urllib.parse.urlencode({'api_key': key})
    request = urllib.request.Request(url, data=json.dumps({'query':document,'variables':variables or {}}).encode(),
        headers={'Content-Type':'application/json','User-Agent':'commercecore-research/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail=json.loads(exc.read()).get('errors',[])
        except (ValueError,AttributeError):
            detail=[]
        messages=[x.get('message','API error').replace(key,'[REDACTED]') for x in detail]
        raise SystemExit('RunPod HTTP status '+str(exc.code)+' '+json.dumps(messages))
    except urllib.error.URLError:
        raise SystemExit('RunPod network request failed (URL suppressed)')
    if result.get('errors'):
        raise SystemExit(json.dumps([x.get('message','API error').replace(key,'[REDACTED]') for x in result['errors']]))
    return result['data']


def main():
    # Migrate the small bookkeeping files created before the research path fix.
    for name in ['runpod_pilot_state.json','initial_pod_cancelled.json']:
        old=ROOT.parent/'work'/name;new=ROOT/'work'/name
        if old.is_file() and not new.exists():
            value=json.loads(old.read_text())
            if value.get('pod',{}).get('name')=='commercecore-research-pilot-20260928':
                new.parent.mkdir(parents=True,exist_ok=True);old.replace(new)
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['inventory','schema','status','create','stop','terminate','cancel-empty'])
    parser.add_argument('--gpu');parser.add_argument('--image');parser.add_argument('--hourly-ceiling',type=float,default=1.)
    parser.add_argument('--cloud',choices=['SECURE','COMMUNITY','ALL'],default='SECURE')
    parser.add_argument('--type-name',default='PodFindAndDeployOnDemandInput')
    args=parser.parse_args()
    if args.action=='inventory':
        result=query('query { gpuTypes { id displayName memoryInGb securePrice communityPrice lowestPrice(input: {gpuCount: 1}) { stockStatus uninterruptablePrice minimumBidPrice } } myself { pods { id name desiredStatus costPerHr } } }')
        result['gpuTypes']=[g for g in result['gpuTypes'] if 20<=g['memoryInGb']<=48]
    elif args.action=='schema':
        result=query('query($name: String!) { __type(name:$name) { name inputFields { name type { kind name ofType {kind name} } } fields {name} } }',{'name':args.type_name})
    elif args.action=='create':
        if STATE.exists():raise SystemExit('A pilot state already exists. Inspect it before creating any new pod.')
        if not args.gpu or not args.image:raise SystemExit('Explicit GPU and image are required')
        if not 0<args.hourly_ceiling<=1.:raise SystemExit('Pilot compute ceiling is $1/hour')
        public_key=(Path.home()/'.ssh/id_ed25519.pub').read_text().strip()
        payload={'cloudType':args.cloud,'gpuCount':1,'volumeInGb':30,'containerDiskInGb':40,
            'minVcpuCount':4,'minMemoryInGb':24,'gpuTypeId':args.gpu,
            'name':'commercecore-research-pilot-20260928','imageName':args.image,
            'ports':'22/tcp','volumeMountPath':'/workspace','startSsh':True,'supportPublicIp':True,
            'stopAfter':datetime.fromtimestamp(time.time()+7200,timezone.utc).isoformat(),
            'env':[{'key':'PUBLIC_KEY','value':public_key}]}
        result=query('mutation($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input:$input) {id name desiredStatus costPerHr imageName machine {gpuDisplayName} } }',{'input':payload})
        pod=result['podFindAndDeployOnDemand']
        STATE.parent.mkdir(parents=True,exist_ok=True)
        state={'pod':pod,'created_unix':time.time(),'deadline_unix':time.time()+7200,
               'user_budget_usd':25,'compute_hourly_ceiling':args.hourly_ceiling,
               'max_pilot_seconds':7200,'storage_gb':30,'status':'CREATED',
               'provider_stop_after':payload['stopAfter']}
        STATE.write_text(json.dumps(state,indent=2)+'\n')
        if float(pod['costPerHr'])>args.hourly_ceiling:
            query('mutation($id: String!) {podStop(input:{podId:$id}) {id desiredStatus}}',{'id':pod['id']})
            raise SystemExit('Pod exceeded declared hourly ceiling and was stopped immediately; inspect state.')
    else:
        state=json.loads(STATE.read_text());pod_id=state['pod']['id']
        if args.action=='status':
            result=query('query($id: String!) {pod(input:{podId:$id}) { id name desiredStatus costPerHr lastStatusChange imageName podType ports machine {gpuDisplayName podHostId} runtime {uptimeInSeconds ports {ip isIpPublic privatePort publicPort type} gpus {gpuUtilPercent memoryUtilPercent} } }}',{'id':pod_id})
        elif args.action=='stop':
            result=query('mutation($id: String!) {podStop(input:{podId:$id}) {id desiredStatus}}',{'id':pod_id})
            state['status']='STOPPED';state['stopped_unix']=time.time();STATE.write_text(json.dumps(state,indent=2)+'\n')
        elif args.action=='cancel-empty':
            if state.get('status')!='CREATED' or state.get('provider_stop_after'):
                raise SystemExit('Only the untouched initial provisioning attempt without its provider deadline can be cancelled here.')
            result=query('mutation($id: String!) {podTerminate(input:{podId:$id})}',{'id':pod_id})
            state['status']='CANCELLED_BEFORE_TRANSFER';state['terminated_unix']=time.time()
            STATE.with_name('initial_pod_cancelled.json').write_text(json.dumps(state,indent=2)+'\n')
            STATE.unlink()
        else:
            if not (ROOT/'work/runpod_backup_verified.json').is_file():
                raise SystemExit('Verified artifact backup marker is required before terminating this dedicated pilot pod')
            result=query('mutation($id: String!) {podTerminate(input:{podId:$id})}',{'id':pod_id})
            state['status']='TERMINATED';state['terminated_unix']=time.time();STATE.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
