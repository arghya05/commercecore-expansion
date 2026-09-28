"""Retrospective development exposure audit. No test labels or model loading."""
import json
from pathlib import Path
from research_v2.core import normalize,file_sha
from research_v2.experiment import read_rows
from research_v2.execution import require_runpod

def main():
    require_runpod()
    root=Path(__file__).resolve().parents[2]
    dev=read_rows(root/'research_v2/work/suite_v2/dev.jsonl')
    shared=root/'data/training_mixture_v1.jsonl'
    understand=root/'data/understand_train.jsonl'
    training=read_rows(shared)
    text={normalize(r['input']) for r in training if r.get('split')=='train'}
    queries={str(r['query_id']) for r in training if r.get('split')=='train' and 'query_id' in r}
    abo=read_rows(understand)
    abo_text={normalize(r['text']) for r in abo}
    abo_ids={str(r['item_id']) for r in abo}
    records=[]
    for r in dev:
        x=r['input'];task=r['task']
        if task=='relevance':
            exact=normalize('Query: '+x['query']+'\nProduct: '+x['product_title']) in text
            group=any(g.startswith('esci:query:') and g.split(':')[-1] in queries for g in r['groups'])
        elif task=='identity':
            exact=normalize('Listing A: '+x['listing_a']+'\nListing B: '+x['listing_b']) in text
            group=None
        elif task=='extraction':
            exact=normalize(x['text']) in abo_text
            group=str(r.get('source_id')) in abo_ids or any(g.split(':')[-1] in abo_ids for g in r['groups'])
        else:raise ValueError(task)
        records.append({'id':r['id'],'task':task,'exact_text_seen':exact,'recorded_group_seen':group})
    summary={t:{'n':sum(r['task']==t for r in records),
        'exact_text_seen':sum(r['task']==t and r['exact_text_seen'] for r in records),
        'recorded_group_seen':sum(r['task']==t and r['recorded_group_seen'] is True for r in records),
        'group_status_unknown':sum(r['task']==t and r['recorded_group_seen'] is None for r in records)} for t in {r['task'] for r in records}}
    out=root/'research_v2/results/followup_20260928';out.mkdir(parents=True,exist_ok=True)
    (out/'exposure.json').write_text(json.dumps({'source_hashes':{str(p.relative_to(root)):file_sha(p) for p in [shared,understand]},
        'summary':summary,'records':records,'interpretation':'Known historical source audit only; identity entity overlap and pretraining exposure remain unknown. All outcomes are development diagnostics, not final generalization.'},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
