"""Offline evidence audit; standard library only, no model inference or API calls."""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/evidence'
ARCHIVE = 'paper/evidence/archive/local_artifacts/'
SOURCES = {}


def read(name):
    raw = (ROOT / name).read_bytes()
    SOURCES[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return raw.decode()


def js(name):
    return json.loads(read(name))


def rows(name):
    return [json.loads(line) for line in read(name).splitlines() if line.strip()]


def norm(value):
    return ' '.join(value.lower().split())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def table(name, data):
    (OUT / name).write_text('\n'.join(' & '.join(map(str, row)) + r' \\' for row in data) + '\n')


def csvfile(name, data):
    assert data, name
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def log_json(name):
    text = read(name)
    decoder = json.JSONDecoder()
    found = []
    for match in re.finditer(r'^\{', text, re.M):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and '_n_by_task' in obj:
            found.append(obj)
    assert len(found) == 1, name
    return found[0]


def classification(confusion, labels):
    cells = {tuple(k.split('->')): v for k, v in confusion.items()}
    n = sum(cells.values())
    per = {}
    for label in labels:
        support = sum(v for (g, _), v in cells.items() if g == label)
        predicted = sum(v for (_, p), v in cells.items() if p == label)
        tp = cells.get((label, label), 0)
        per[label] = dict(support=support, predicted=predicted, tp=tp,
                         precision=tp/predicted if predicted else 0.0,
                         recall=tp/support if support else 0.0,
                         f1=2*tp/(support+predicted) if support+predicted else 0.0)
    return dict(n=n, accuracy=sum(cells.get((c, c), 0) for c in labels)/n,
                macro_f1=sum(v['f1'] for v in per.values())/len(labels),
                balanced_accuracy=sum(v['recall'] for v in per.values())/len(labels),
                per_class=per, confusion=confusion)


def exact_mcnemar(b, c):
    n = b+c
    return min(1.0, 2*sum(math.comb(n, k) for k in range(min(b,c)+1))/2**n) if n else 1.0


def aggregate_pairing_bounds(n, correct_a, correct_b):
    if not all(isinstance(v, int) for v in (n, correct_a, correct_b)) or not (0 <= correct_a <= n and 0 <= correct_b <= n):
        raise ValueError('Correctness margins must be integer counts between zero and n')
    possibilities = [dict(both_correct=x, a_only=correct_a-x, b_only=correct_b-x,
                          two_sided_exact_p=exact_mcnemar(correct_a-x, correct_b-x))
                     for x in range(max(0,correct_a+correct_b-n), min(correct_a,correct_b)+1)]
    return dict(n=n, correct_a=correct_a, correct_b=correct_b, possible_pairings=possibilities,
                p_min=min(x['two_sided_exact_p'] for x in possibilities),
                p_max=max(x['two_sided_exact_p'] for x in possibilities),
                interpretation='Possible pairings for binary correctness, not observed predictions or a paired F1 test')


def overlap(left, right, key):
    a, b = {key(r) for r in left}, {key(r) for r in right}
    return dict(shared_keys=len(a & b), right_rows_with_left_key=sum(key(r) in a for r in right), right_rows=len(right))


def split_summary(data):
    return {s: dict(n=sum(r['split']==s for r in data),
                    labels=dict(Counter(r.get('target',r.get('label')) for r in data if r['split']==s)))
            for s in sorted({r['split'] for r in data})}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = js('paper/evidence/archive/repository_inventory.json')
    for item in inventory['files']:
        if item['path'] == 'README.md':
            continue
        read(item['path'])
        assert SOURCES[item['path']]['sha256'] == item['sha256'], 'Historical artifact changed: '+item['path']
    archive = js('paper/evidence/archive/manifest.json')
    for item in archive['files']:
        read(item['archive_path'])
        assert SOURCES[item['archive_path']]['sha256'] == item['sha256']
    results = {name: log_json(path) for name,path in [
        ('shared_v1_rescore','reports/v1_dev_split_correction_2026-09-28/v1_corrected_eval.log'),
        ('shared_v2','reports/batched_eval_v2.log'),('shared_v3','reports/batched_eval_v3.log')]}
    old = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^\n]+)\}',read('paper/evidence/archive/pre_revision_numbers.tex')))
    results['shared_v1_prior_manuscript_only'] = dict(relevance=float(old['AdapterOneRelevanceAcc']),
        identity=float(old['AdapterOneIdentityAcc']), evidence='Prior manuscript macros; original full evaluation log/vector absent')
    final, scaled = rows('data/training_mixture_final.jsonl'), rows('data/training_mixture_v3.jsonl')
    saved_dev = js(ARCHIVE+'models/shared_adapter_v1/dev_rows_for_eval.json')
    mixture_counts = {name:{f'{t}/{s}':n for (t,s),n in sorted(Counter((r['task'],r.get('split','dev')) for r in data).items())}
                      for name,data in [('current_final',final),('scaled_v3',scaled),('saved_v1_dev',saved_dev)]}
    task_names = {'match_relevance':'Relevance','match_identity':'Identity','match_functional_relation':'Functional relation','match_technical_compatibility':'Compatibility'}
    data_table=[]
    for task,label in task_names.items():
        cur=[r for r in final if r['task']==task];olddev=[r for r in saved_dev if r['task']==task]
        data_table.append([label,sum(r['split']=='train' for r in cur),sum(r['split']=='dev' for r in cur),len(olddev),len({r['target'] for r in olddev})])
    table('mixture_rows.tex',data_table)
    relevance_overlap={}
    for name,data in [('current',final),('scaled',scaled)]:
        train=[r for r in data if r['task']=='match_relevance' and r['split']=='train']
        dev=[r for r in data if r['task']=='match_relevance' and r['split']=='dev']
        relevance_overlap[name]={'query_id':overlap(train,dev,lambda r:r['query_id']),'input':overlap(train,dev,lambda r:norm(r['input']))}
    frontier_inputs={t:js('data/frontier_eval_'+t+'.json') for t in ['relevance','identity']}
    frontier_overlap={}
    for task,data in frontier_inputs.items():
        fmt=(lambda r:f"Query: {r['query']}\nProduct: {r['product_title']}") if task=='relevance' else (lambda r:f"Listing A: {r['title_a']}\nListing B: {r['title_b']}")
        frontier_overlap[task]={}
        for split in ['train','dev']:
            src={norm(r['input']) for r in final if r['task']=='match_'+task and r['split']==split}
            frontier_overlap[task][split]=sum(norm(fmt(r)) in src for r in data)
    synthetic={}
    for short,fn in [('functional_original','functional_relation.jsonl'),('compatibility','technical_compatibility.jsonl'),('functional_expanded','functional_relation_v3_final.jsonl')]:
        data=rows('data/synthetic_match/'+fn); train=[r for r in data if r['split']=='train']
        info=dict(split_counts=split_summary(data),unique_scenarios=len({fingerprint(r['scenario']) for r in data}),overlap={})
        for split in sorted({r['split'] for r in data}-{'train'}):
            rr=[r for r in data if r['split']==split]
            info['overlap'][split]={'scenario':overlap(train,rr,lambda r:fingerprint(r['scenario'])),
                                   'text':overlap(train,rr,lambda r:norm(r['generated_text']))}
        if short!='functional_expanded':
            task='match_functional_relation' if short=='functional_original' else 'match_technical_compatibility'
            oldtrain=[r for r in scaled if r['task']==task and r['split']=='train']
            corrected=[r for r in data if r['split']=='dev']
            info['corrected_dev_seen_in_original_train']=overlap(oldtrain,corrected,lambda r:norm(r['input']))
        if short=='functional_original':
            dev=[r for r in data if r['split']=='dev']
            info['frontier_empty_second_listing']=sum(len(r['generated_text'].split('\n'))<2 or not r['generated_text'].split('\n')[1].strip() for r in dev)
        synthetic[short]=info
    extraction={name:(js(fn) if fn.endswith('.json') else rows(fn)) for name,fn in [
        ('train','data/understand_train.jsonl'),('dev','data/understand_dev.jsonl'),('eval','data/abo/understand_eval_grounded.json')]}
    extract_audit={'counts':{},'overlap':{}}
    for name,data in extraction.items():
        extract_audit['counts'][name]=dict(n=len(data),unique_ids=len({r['item_id'] for r in data}),unique_texts=len({r['text'] for r in data}),brands=len({r['brand'] for r in data}),
            brand_literal=sum(r['brand'].lower() in r['text'].lower() for r in data),color_literal=sum(r['color'].lower() in r['text'].lower() for r in data))
    for a,b in [('train','dev'),('train','eval'),('dev','eval')]:
        extract_audit['overlap'][a+'_'+b]={k:overlap(extraction[a],extraction[b],lambda r,k=k:r[k]) for k in ['item_id','text']}
    extract_audit['nonliteral_eval_color_examples']=[dict(item_id=r['item_id'],color=r['color'],text=r['text']) for r in extraction['eval'] if r['color'].lower() not in r['text'].lower()][:3]
    frontier=js('reports/frontier_comparison_2026-09-27/report.json');frontier_stats={};frontier_table=[]
    for model in frontier['relevance']:
        values=[]
        for task in ['relevance','identity']:
            rec=frontier[task][model];gold=[r['gold'] for r in frontier_inputs[task]];pred=rec['predictions']
            assert len(pred)==len(gold)==rec['n']
            metrics=classification(dict(Counter(f'{g}->{p}' for g,p in zip(gold,pred))),sorted(set(gold)))
            assert math.isclose(metrics['accuracy'],rec['accuracy'])
            frontier_stats[task+'/'+model]=metrics
            values += [f"{sum(g==p for g,p in zip(gold,pred))}/{len(gold)}",f"{metrics['accuracy']:.3f}"]
        frontier_table.append([model,*values])
    ec=js('reports/ecellm_result.json')
    frontier_table.append(['eCeLLM-S (aggregate)',f"{round(ec['relevance_accuracy']*ec['n_relevance'])}/{ec['n_relevance']}",f"{ec['relevance_accuracy']:.3f}",f"{round(ec['identity_accuracy']*ec['n_identity'])}/{ec['n_identity']}",f"{ec['identity_accuracy']:.3f}"])
    table('frontier_rows.tex',frontier_table)
    ext=js('reports/understand_locked_eval_result.json');api_ext=js('reports/understand_frontier_comparison_2026-09-28/report.json')
    base=js('reports/understand_baseline_2026-09-27/report.json');g25=js('reports/extraction_comparators_2026-09-27/gliner25_only.json')
    ext_table=[]
    for name,rec in [('Dictionary',base['rules']),('GLiNER2',base['gliner2_base']),('GLiNER2.5',g25),*api_ext['results'].items(),('Dedicated adapter',ext)]:
        values=[]
        for field in ['brand','color']:
            c=rec[field];f=2*c['tp']/(2*c['tp']+c['fp']+c['fn']);assert math.isclose(f,c['f1'])
            values += [f"{c['tp']}/{c['fp']}/{c['fn']}",f"{f:.3f}"]
        ext_table.append([name,*values])
    table('extraction_rows.tex',ext_table)
    best=max(api_ext['results'],key=lambda k:api_ext['results'][k]['color']['f1'])
    pairing=aggregate_pairing_bounds(ext['n'],ext['color']['tp'],api_ext['results'][best]['color']['tp']);pairing['comparator']=best
    rel=js('reports/relevance_singletoken_experiment_2026-09-28/final_full_dev_scores_singletoken.json')
    rel_stats=classification(rel['confusion'],['exact','substitute','complement','irrelevant'])
    assert math.isclose(rel_stats['accuracy'],rel['accuracy']) and rel_stats['n']==rel['n']
    table('relevance_class_rows.tex',[[k,v['support'],v['predicted'],f"{v['precision']:.3f}",f"{v['recall']:.3f}",f"{v['f1']:.3f}"] for k,v in rel_stats['per_class'].items()])
    labels=list(rel_stats['per_class']);table('relevance_confusion_rows.tex',[[g,*[rel['confusion'].get(g+'->'+p,0) for p in labels]] for g in labels])
    fr=js('reports/functional_relation_locked_eval_result.json');fr_stats=classification(fr['confusion'],['substitute','complement','unrelated'])
    assert math.isclose(fr_stats['accuracy'],fr['accuracy'])
    labels=['substitute','complement','unrelated'];table('functional_confusion_rows.tex',[[g,*[fr['confusion'].get(g+'->'+p,0) for p in labels]] for g in labels])
    fr_ec=js('reports/ecellm_functional_relation_result.json');mapping={'a':'substitute','b':'complement','c':'unrelated'}
    parser=dict(n=fr_ec['n'],legacy_correct=0,strict_valid=0,strict_correct=0,raw_counts=dict(Counter(r['raw'] for r in fr_ec['predictions'])))
    for rec in fr_ec['predictions']:
        raw=rec['raw'].strip().lower();legacy=mapping.get(next((c for c in raw if c in 'abc'),None),'invalid');assert legacy==rec['pred']
        parser['legacy_correct']+=legacy==rec['gold'];valid=re.fullmatch(r'[abc][.)]?',raw)
        parser['strict_valid']+=bool(valid);parser['strict_correct']+=bool(valid) and mapping[raw[0]]==rec['gold']
    history_paths={'shared_v1':'models/shared_adapter_v1/checkpoint_history.json','shared_v2':'reports/shared_adapter_v2_checkpoint_history.json',
        'shared_v3':'reports/shared_adapter_v3_checkpoint_history.json','single_token':'reports/relevance_singletoken_experiment_2026-09-28/checkpoint_history.json',
        'functional':ARCHIVE+'models/functional_relation_adapter_v1/checkpoint_history.json','understand':'models/understand_adapter_v1/checkpoint_history.json'}
    trajectories={}
    for name,path in history_paths.items():
        h=js(path);trajectories[name]=h
        flat=[dict(step=r['step'],brand=r['match_understand_brand'],color=r['match_understand_color'],parse=r['parse_rate']) for r in h] if name=='understand' else [{'step':r['step'],**{k.replace('match_',''):v for k,v in r['dev_accuracy_by_task'].items()}} for r in h]
        csvfile('checkpoint_'+name+'.csv',flat)
    sweep=js('reports/functional_relation_locked_eval_checkpoint_sweep.json')
    fdev={str(r['step']):r['dev_accuracy_by_task']['match_functional_relation'] for r in trajectories['functional']}
    table('functional_checkpoints.tex',[[s,f"{fdev[s]:.3f}",f"{v:.3f}"] for s,v in sweep['checkpoints'].items()])
    csvfile('functional_sweep.csv',[dict(step=int(s),dev=fdev[s],exposed_test=v) for s,v in sweep['checkpoints'].items()])
    losses={};runtime={}
    for name,path in [('shared_v1',ARCHIVE+'models/shared_adapter_v1/checkpoint-400/trainer_state.json'),('understand',ARCHIVE+'models/understand_adapter_v1/checkpoint-1200/trainer_state.json')]:
        state=js(path);losses[name]=[{k:r[k] for k in ['step','loss','epoch']} for r in state['log_history'] if 'loss' in r]
        runtime[name]=dict(global_step=state['global_step'],epoch=state['epoch'],train_batch_size=state['train_batch_size'],runtime_seconds=None)
    for name,path in [('functional','reports/functional_relation_experiment_2026-09-28/functional_relation_train.log'),('single_token','reports/relevance_singletoken_experiment_2026-09-28/singletoken_train.log')]:
        entries=[]
        for match in re.finditer(r'\{[^{}\n]*\}',read(path)):
            try:rec=ast.literal_eval(match.group())
            except (ValueError,SyntaxError):continue
            if isinstance(rec,dict) and 'loss' in rec:entries.append(dict(step=10*(len(entries)+1),loss=float(rec['loss']),epoch=float(rec['epoch'])))
            if isinstance(rec,dict) and 'train_runtime' in rec:runtime[name]={k:float(v) for k,v in rec.items()}
        losses[name]=entries
    loss_table=[]
    for name,entries in losses.items():
        csvfile('loss_'+name+'.csv',entries)
        loss_table.append([name.replace('_',' '),len(entries),f"{entries[0]['loss']:.4f}",f"{entries[-1]['loss']:.4f}",f"{entries[-1]['epoch']:.3f}"])
    table('loss_summary_rows.tex',loss_table)
    load={name:list(csv.DictReader(io.StringIO(read('reports/load_test_2026-09-28/'+fn)))) for name,fn in [('before','result_stats.csv'),('after','result_after_fix_stats.csv')]}
    macros=dict(CurrentMixtureRows=str(len(final)),CurrentTrainRows=str(sum(r['split']=='train' for r in final)),CurrentDevRows=str(sum(r['split']=='dev' for r in final)),SavedDevRows=str(len(saved_dev)),
        SharedRelevance=f"{results['shared_v1_rescore']['match_relevance']:.3f}",SharedIdentity=f"{results['shared_v1_rescore']['match_identity']:.3f}",
        SingleAccuracy=f"{rel_stats['accuracy']:.4f}",SingleMacroF=f"{rel_stats['macro_f1']:.3f}",SingleBalanced=f"{rel_stats['balanced_accuracy']:.3f}",
        FunctionalAccuracy=f"{fr_stats['accuracy']:.3f}",FunctionalMacroF=f"{fr_stats['macro_f1']:.3f}",ExtractionBrand=f"{ext['brand']['f1']:.3f}",ExtractionColor=f"{ext['color']['f1']:.3f}",
        ExtractionColorDelta=f"{100*(ext['color']['f1']-api_ext['results'][best]['color']['f1']):.2f}",PairPMin=f"{pairing['p_min']:.4f}",PairPMax=f"{pairing['p_max']:.4f}",
        ScenarioTestSeen=str(synthetic['functional_expanded']['overlap']['locked_test']['scenario']['right_rows_with_left_key']),ScenarioDevSeen=str(synthetic['functional_expanded']['overlap']['dev']['scenario']['right_rows_with_left_key']),
        StrictEcellmValid=str(parser['strict_valid']),EcellmRawN=str(parser['n']))
    (OUT/'numbers.tex').write_text('% Generated by audit_evidence.py; do not edit.\n'+''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k,v in macros.items()))
    audit=dict(audit_date='2026-09-28',baseline_commit=inventory['baseline_commit'],scope='Offline reanalysis only; no new inference, training, API or GPU use',results=results,mixture_counts=mixture_counts,relevance_overlap=relevance_overlap,frontier_input_overlap=frontier_overlap,synthetic=synthetic,extraction=extract_audit,frontier_prediction_metrics=frontier_stats,extraction_pairing_bounds=pairing,single_token_metrics=rel_stats,functional_metrics=fr_stats,ecellm_parser=parser,training_runtime=runtime,load_test=load,macros=macros,source_files=dict(sorted(SOURCES.items())))
    (OUT/'audit_results.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(audited_source_files=len(SOURCES),macros=macros,query_overlap=relevance_overlap,frontier_overlap=frontier_overlap,parser=parser,runtime=runtime),indent=2))


if __name__ == '__main__':
    main()
