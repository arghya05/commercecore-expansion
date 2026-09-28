"""Analyze paired DEVELOPMENT results on RunPod; no final-test access."""
import json
from pathlib import Path
from research_v2.core import paired_correctness,score_rows,file_sha
from research_v2.experiment import read_rows
from research_v2.execution import require_runpod

def main():
    require_runpod()
    root=Path(__file__).resolve().parents[2];out=root/'research_v2/results/followup_20260928'
    dev=read_rows(root/'research_v2/work/suite_v2/dev.jsonl')
    exposure=json.loads((out/'exposure.json').read_text())
    unseen={r['id'] for r in exposure['records'] if not r['exact_text_seen'] and r['recorded_group_seen'] is not True}
    results={}
    for interface in ['legacy','chat']:
        for task in ['relevance','identity','extraction']:
            name=('understand' if task=='extraction' else 'shared')+'_'+interface
            rows=[r for r in dev if r['task']==task];ids={r['id'] for r in rows}
            left=[r for r in read_rows(out/name/'predictions.jsonl') if r['id'] in ids]
            right=[r for r in read_rows(out/('base_'+interface)/'predictions.jsonl') if r['id'] in ids]
            clean=[r for r in rows if r['id'] in unseen]
            clean_ids={r['id'] for r in clean}
            results[task+'_'+interface]={'base':score_rows(rows,right)[task],'adapter':score_rows(rows,left)[task],
                'paired_adapter_minus_base':paired_correctness(rows,left,right),
                'known_exposure_filtered':{'n':len(clean),'base':score_rows(clean,[r for r in right if r['id'] in clean_ids])[task],
                    'adapter':score_rows(clean,[r for r in left if r['id'] in clean_ids])[task]} if clean else None,
                'prediction_hashes':{'adapter':file_sha(out/name/'predictions.jsonl'),'base':file_sha(out/('base_'+interface)/'predictions.jsonl')}}
    result={'phase':'development','test_accessed':False,'results':results,
        'limitations':['Retrospective pretrained adapters; historical exposure is explicitly audited but not fully recoverable.',
        'Identity entity overlap unknown; filtered is not certified unseen.',
        'Single deterministic run per interface; no new training or causal ablation.',
        'Paired intervals apply to accuracy/joint correctness, not macro-F1.',
        'Exploratory 2,000-resample existing cluster bootstrap, not the proposed confirmatory 10,000-resample test analysis.']}
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
