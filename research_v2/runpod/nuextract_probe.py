"""Documented native text/schema interface, development-only comparator."""
import json
import time
from pathlib import Path
from research_v2.execution import require_runpod
from research_v2.experiment import read_rows
from research_v2.core import digest,file_sha,score_rows

def main():
    require_runpod()
    import torch
    from huggingface_hub import HfApi,snapshot_download
    from transformers import AutoTokenizer,AutoModelForVision2Seq
    root=Path(__file__).resolve().parents[2];out=root/'research_v2/results/followup_20260928/nuextract'
    out.mkdir(exist_ok=False)
    repo='numind/NuExtract-2.0-2B';revision=HfApi().model_info(repo).sha
    path=Path(snapshot_download(repo,revision=revision))
    # Native Transformers implementation only; no repository Python is executed.
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=False,padding_side='left')
    tok.pad_token=tok.eos_token
    model=AutoModelForVision2Seq.from_pretrained(path,trust_remote_code=False,dtype=torch.bfloat16).to('cuda').eval()
    rows=[r for r in read_rows(root/'research_v2/work/suite_v2/dev.jsonl') if r['task']=='extraction']
    template=json.dumps({'brand':'verbatim-string','color':'verbatim-string'},indent=4)
    texts=[tok.apply_chat_template([{'role':'user','content':[{'type':'text','text':r['input']['text']}]}],
        template=template,tokenize=False,add_generation_prompt=True) for r in rows]
    manifest={'repository':repo,'revision':revision,'phase':'development','test_accessed':False,
        'input_sha256':digest(rows),'prompt_sha256':digest(texts),'template':template,
        'batch_size':8,'max_new_tokens':64,'max_input_tokens':2048,'dtype':'bfloat16','decode':'greedy',
        'scorer_sha256':file_sha(root/'research_v2/core.py'),'runner_sha256':file_sha(__file__),
        'files_sha256':{p.name:file_sha(p) for p in path.iterdir() if p.is_file()},
        'interface':'publisher native text/schema template; no examples, no prompt tuning'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    predictions=[];start=time.perf_counter()
    with (out/'predictions.jsonl').open('x') as journal:
        for offset in range(0,len(rows),8):
            group=rows[offset:offset+8];begin=time.perf_counter()
            try:
                inp=tok(texts[offset:offset+8],padding=True,return_tensors='pt',truncation=False).to('cuda')
                if inp['input_ids'].shape[1]>2048:raise ValueError('Input cap exceeded')
                with torch.inference_mode():generated=model.generate(**inp,max_new_tokens=64,do_sample=False,pad_token_id=tok.eos_token_id)
                outputs=tok.batch_decode(generated[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)
                records=[{'id':r['id'],'status':'ok','raw_output':text} for r,text in zip(group,outputs)]
            except Exception as exc:records=[{'id':r['id'],'status':'error','raw_output':None,'error':str(exc)[:500]} for r in group]
            for rec in records:
                rec['batch_seconds']=time.perf_counter()-begin;predictions.append(rec);journal.write(json.dumps(rec)+'\n')
            journal.flush();print(offset+len(group),len(rows),flush=True)
    report={'metrics':score_rows(rows,predictions),'elapsed_seconds':time.perf_counter()-start,
        'manifest_sha256':file_sha(out/'manifest.json'),'predictions_sha256':file_sha(out/'predictions.jsonl')}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
