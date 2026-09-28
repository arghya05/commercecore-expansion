"""Loopback HTTP serving experiment, serial queue versus bounded microbatching.

This is a reference prototype, not vLLM or a production SLO certification.
"""
import argparse
import concurrent.futures
import http.server
import json
import queue
import threading
import time
import urllib.request
from pathlib import Path
from research_v2.execution import require_runpod
from research_v2.experiment import read_rows
from research_v2.core import digest,file_sha,parse_label
from research_v2.run_local import legacy_prompt

def main():
    require_runpod()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/'research_v2/results/followup_20260928/serving'
    out.mkdir(parents=True,exist_ok=False)
    rows=sorted([r for r in read_rows(root/'research_v2/work/suite_v2/dev.jsonl') if r['task']=='identity'],key=lambda r:digest(r['id']))[:200]
    by_id={r['id']:r for r in rows}
    begin=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True,padding_side='left')
    tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(a.snapshot,local_files_only=True,dtype=torch.bfloat16).to('cuda')
    model=PeftModel.from_pretrained(base,str(root/'models/shared_adapter_v1'),is_trainable=False).eval()
    torch.cuda.synchronize();load_seconds=time.perf_counter()-begin
    requests=queue.Queue(maxsize=32);stop=threading.Event();batch_limit=[1]
    def worker():
        while not stop.is_set():
            try:first=requests.get(timeout=.1)
            except queue.Empty:continue
            batch=[first];deadline=time.perf_counter()+.005
            while len(batch)<batch_limit[0]:
                try:batch.append(requests.get(timeout=max(.00001,deadline-time.perf_counter())))
                except queue.Empty:break
                if time.perf_counter()>=deadline:break
            try:
                prompts=[legacy_prompt(by_id[item[0]]) for item in batch]
                inp=tok(prompts,padding=True,return_tensors='pt',truncation=False).to('cuda')
                if inp['input_ids'].shape[1]>2048:raise ValueError('Input cap exceeded')
                with torch.inference_mode():
                    generated=model.generate(**inp,max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)
                raw=tok.batch_decode(generated[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)
                for item,text in zip(batch,raw):item[1].set_result({'raw_output':text,'batch_size':len(batch),'status':'ok'})
            except Exception as exc:
                for item in batch:item[1].set_result({'raw_output':None,'status':'error','error':str(exc)[:300]})
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                ident=payload['id']
                if ident not in by_id:raise ValueError('Unknown row')
                future=concurrent.futures.Future();requests.put((ident,future),timeout=1)
                result=future.result(timeout=60);status=200
            except Exception as exc:result={'status':'error','raw_output':None,'error':str(exc)[:300]};status=503
            data=json.dumps(result).encode();self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    serving=threading.Thread(target=server.serve_forever,daemon=True);serving.start()
    url='http://127.0.0.1:'+str(server.server_port)
    def request(row):
        start=time.perf_counter();wall=time.time()
        try:
            req=urllib.request.Request(url,data=json.dumps({'id':row['id']}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=65) as response:result=json.load(response)
        except Exception as exc:result={'status':'error','raw_output':None,'error':str(exc)[:300]}
        result.update({'id':row['id'],'started_unix':wall,'seconds':time.perf_counter()-start})
        label=parse_label(result['raw_output'],'identity') if result['status']=='ok' else None
        result.update({'valid':label is not None,'correct':label==row['gold']})
        return result
    cells=[]
    manifest={'scope':'development loopback HTTP prototype; closed-loop only','request_count_per_cell':200,'repetitions':3,
        'concurrency':[1,4,8],'batch_limits':[1,8],'max_wait_ms':5,'max_new_tokens':8,'load_seconds':load_seconds,
        'rows_sha256':digest(rows),'adapter_sha256':file_sha(root/'models/shared_adapter_v1/adapter_model.safetensors'),
        'gpu':torch.cuda.get_device_name(0),'dtype':'bfloat16','hourly_gpu_quote_usd':.22,
        'limitations':'Client and server share a pod; exploratory tails; no open-loop test, invoice or production SLO. Training/adaptation is unchanged.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        with (out/'requests.jsonl').open('x') as journal:
            for repetition in range(3):
                for limit in ([1,8] if repetition%2==0 else [8,1]):
                    batch_limit[0]=limit
                    for row in rows[:50]:
                        rec=request(row);rec.update({'warmup':True,'batch_limit':limit,'repetition':repetition});journal.write(json.dumps(rec)+'\n')
                    for concurrency in [1,4,8]:
                        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:records=list(pool.map(request,rows))
                        elapsed=time.perf_counter()-started
                        for rec in records:
                            rec.update({'warmup':False,'batch_limit':limit,'repetition':repetition,'concurrency':concurrency});journal.write(json.dumps(rec)+'\n')
                        journal.flush();lat=sorted(r['seconds'] for r in records)
                        cells.append({'batch_limit':limit,'repetition':repetition,'concurrency':concurrency,'attempted':len(records),
                            'completed':sum(r['status']=='ok' for r in records),'valid':sum(r['valid'] for r in records),
                            'correct':sum(r['correct'] for r in records),'elapsed_seconds':elapsed,
                            'completed_per_second':sum(r['status']=='ok' for r in records)/elapsed,
                            'p50_seconds':lat[99],'p95_seconds':lat[189],'p99_seconds':lat[197],
                            'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),
                            'gpu_quote_cost_usd':elapsed*.22/3600})
                        print(json.dumps(cells[-1]),flush=True)
    finally:
        server.shutdown();stop.set();thread.join(timeout=10);server.server_close()
    result={'cells':cells,'request_sha256':file_sha(out/'requests.jsonl'),'manifest_sha256':file_sha(out/'manifest.json')}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
