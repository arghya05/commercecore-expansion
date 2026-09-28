"""RunPod development evaluation; filename retained for existing command references.

The optional row cap is a deterministic smoke test, NOT a performance benchmark.
This runner cannot execute on the laptop. It does not provision paid services.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from research_v2.core import canonical,digest,file_sha,prompt,score_rows,validate_contract,LABELS
from research_v2.experiment import read_rows
from research_v2.execution import require_runpod


def select_dev(rows, per_task=None):
    validate_contract(rows)
    if any(r['split']!='dev' for r in rows):raise ValueError('This runner accepts development rows only')
    selected=[]
    for task in sorted({r['task'] for r in rows}):
        subset=sorted((r for r in rows if r['task']==task),key=lambda r:digest(['smoke-order',r['id']]))
        selected.extend(subset[:per_task] if per_task is not None else subset)
    if not selected:raise ValueError('Empty development evaluation')
    return selected


def main():
    require_runpod()
    p=argparse.ArgumentParser();p.add_argument('--dev',default='research_v2/work/suite_v2/dev.jsonl')
    p.add_argument('--output',required=True);p.add_argument('--backend',choices=['majority','hf'],default='majority')
    p.add_argument('--train',default='research_v2/work/suite_v2/train.jsonl')
    p.add_argument('--snapshot');p.add_argument('--adapter');p.add_argument('--per-task',type=int)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--max-input-tokens',type=int,default=2048);p.add_argument('--device',choices=['cuda'],default='cuda')
    a=p.parse_args()
    if a.per_task is not None and a.per_task<1:p.error('--per-task must be positive')
    if a.max_new_tokens<1 or a.max_input_tokens<1:p.error('Token caps must be positive')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a new empty run directory; never overwrite a prediction journal')
    rows=select_dev(read_rows(a.dev),a.per_task)
    prompts={r['id']:prompt(r) for r in rows}
    manifest={'phase':'development_smoke' if a.per_task else 'development',
              'backend':a.backend,'rows':len(rows),'input_sha256':digest(rows),'prompt_sha256':digest(prompts),
              'scorer_sha256':file_sha(Path(__file__).with_name('core.py')),
              'runner_sha256':file_sha(__file__),'decode':{'temperature':0,'max_new_tokens':a.max_new_tokens,
              'max_input_tokens':a.max_input_tokens,'truncation':'reject','enable_thinking':False},
              'environment':{'python':platform.python_version(),'system':platform.platform()},
              'test_accessed':False,'paid_compute':True,'execution_location':'RunPod',
              'pod_id':os.environ['RUNPOD_POD_ID']}
    load_start=time.perf_counter()
    if a.backend=='majority':
        from collections import Counter
        train=read_rows(a.train);validate_contract(train)
        if any(r['split']!='train' for r in train):raise ValueError('Majority must fit train only')
        manifest['model_identity']='train-majority/null-output sanity control'
        manifest['train_sha256']=digest(train)
        majority={t:Counter(r['gold'] for r in train if r['task']==t).most_common(1)[0][0]
                  for t in LABELS if any(r['task']==t for r in train)}
        def generate(r):
            return (canonical({'brand':None,'color':None}) if r['task']=='extraction' else majority[r['task']],None,None)
    else:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        if not a.snapshot:raise ValueError('Pass a locally cached immutable snapshot directory')
        snapshot=Path(a.snapshot).resolve()
        if snapshot.name!='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e':
            raise ValueError('Expected the declared immutable Qwen3-1.7B revision')
        if not snapshot.is_dir():raise ValueError('Snapshot unavailable locally')
        torch.set_num_threads(a.threads);torch.manual_seed(20260928)
        if a.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
        manifest['model_identity']={'repository':'Qwen/Qwen3-1.7B','revision':snapshot.name,
            'files':{f.name:file_sha(f) for f in sorted(snapshot.iterdir()) if f.is_file()}}
        for pkg in ['torch','transformers','peft','safetensors']:
            manifest['environment'][pkg]=importlib.metadata.version(pkg)
        manifest['environment'].update({'device':a.device,'threads':a.threads,
            'dtype':'float32' if a.device=='cpu' else 'bfloat16'})
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
                   dtype=torch.float32 if a.device=='cpu' else torch.bfloat16).to(a.device)
        if a.adapter:
            from peft import PeftModel
            adapter=Path(a.adapter)
            files=[adapter/'adapter_config.json',adapter/'adapter_model.safetensors']
            if not all(f.is_file() for f in files):raise ValueError('Missing adapter files')
            manifest['adapter']={f.name:file_sha(f) for f in files}
            model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True)
        model.eval()
        def generate(r):
            text=tokenizer.apply_chat_template([{'role':'user','content':prompts[r['id']]}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
            batch=tokenizer(text,return_tensors='pt',truncation=False).to(a.device)
            n=batch['input_ids'].shape[1]
            if n>a.max_input_tokens:raise ValueError(f'Input length {n} exceeds cap; no silent truncation')
            with torch.inference_mode():
                result=model.generate(**batch,max_new_tokens=a.max_new_tokens,do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
            tokens=result[0,n:]
            return tokenizer.decode(tokens,skip_special_tokens=True),n,len(tokens)
    manifest['load_seconds']=time.perf_counter()-load_start
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results=[];begin=time.perf_counter()
    with (out/'predictions.jsonl').open('x') as stream:
        for i,row in enumerate(rows):
            start=time.perf_counter()
            try:
                raw,nt,ng=generate(row);rec={'id':row['id'],'status':'ok','raw_output':raw,
                    'input_tokens':nt,'output_tokens':ng}
            except Exception as exc:
                rec={'id':row['id'],'status':'error','raw_output':None,'error_type':type(exc).__name__,
                     'error':str(exc)[:500]}
            rec['seconds']=time.perf_counter()-start;results.append(rec)
            stream.write(canonical(rec)+'\n');stream.flush()
            print(f"{i+1}/{len(rows)} {row['task']} {rec['status']} {rec['seconds']:.2f}s",flush=True)
    report={'manifest_sha256':file_sha(out/'manifest.json'),'phase':manifest['phase'],
            'elapsed_seconds':time.perf_counter()-begin,'metrics':score_rows(rows,results),
            'predictions_sha256':file_sha(out/'predictions.jsonl'),
            'interpretation':'Development only. A capped run validates execution, not model quality or serving throughput.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
