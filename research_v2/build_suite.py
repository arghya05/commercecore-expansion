"""Build a new real-data suite, preserving original artifacts and excluding historical evaluation exposure.

Run with the repository .venv. Outputs are local, including a separately stored test gold file.
These are local subsets of public corpora, not their official leaderboard protocols.
"""
from __future__ import annotations
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from research_v2.core import SCHEMA_VERSION, LABELS, canonical, digest, file_sha, normalize, validate_contract, group_overlap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_v2/work/suite_v2'
PUBLIC=ROOT/'research_v2/manifests'


def read_jsonl(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(canonical(row)+'\n' for row in rows))


class Components:
    def __init__(self):self.parent={}
    def root(self,key):
        self.parent.setdefault(key,key)
        while self.parent[key]!=key:
            self.parent[key]=self.parent[self.parent[key]];key=self.parent[key]
        return key
    def join(self,keys):
        roots=sorted({self.root(k) for k in keys})
        for r in roots[1:]:self.parent[r]=roots[0]
        return roots[0]


def row(task, source, source_id, x, gold, groups, split):
    return {'id':digest([SCHEMA_VERSION,task,source,source_id]),'task':task,'source':source,
            'source_id':str(source_id),'input':x,'gold':gold,'groups':sorted(set(groups)),
            'split':split,'quality':{'gold_origin':'publisher_metadata_or_annotation','human_reaudit':False}}


def take_groups(records, limit, seed):
    groups=defaultdict(list)
    for rec in records:groups[rec['sampling_group']].append(rec)
    result=[]
    for key in sorted(groups,key=lambda x:digest([seed,x])):
        batch=groups[key]
        if len(batch)>limit:continue
        if len(result)+len(batch)>limit:continue
        result.extend(batch)
    return result


def historical():
    mixture=[]
    for name in ['training_mixture_v1.jsonl','training_mixture_final.jsonl','training_mixture_v3.jsonl','relevance_rows_v3.jsonl']:
        mixture.extend(read_jsonl(ROOT/'data'/name))
    qids={str(r['query_id']) for r in mixture if r['task']=='match_relevance'}
    queries=set();titles=set()
    for r in mixture:
        if r['task']=='match_relevance':queries.add(normalize(r['input'].split('\nProduct:')[0].removeprefix('Query: ')))
        if r['task']=='match_identity':
            for line in r['input'].split('\n'):
                if line.startswith(('Listing A: ','Listing B: ')):titles.add(normalize(line[11:]))
    for r in json.loads((ROOT/'data/frontier_eval_relevance.json').read_text()):queries.add(normalize(r['query']))
    for r in json.loads((ROOT/'data/frontier_eval_identity.json').read_text()):
        titles.update([normalize(r['title_a']),normalize(r['title_b'])])
    ext=read_jsonl(ROOT/'data/understand_train.jsonl')+read_jsonl(ROOT/'data/understand_dev.jsonl')+json.loads((ROOT/'data/abo/understand_eval_grounded.json').read_text())
    return {'query_ids':qids,'queries':queries,'identity_titles':titles,'abo_ids':{r['item_id'] for r in ext},'abo_texts':{normalize(r['text']) for r in ext}}


def relevance(hist, inputs):
    import pandas as pd
    folder=ROOT/'data/esci/esci-data/shopping_queries_dataset'
    ex_path=folder/'shopping_queries_dataset_examples.parquet';prod_path=folder/'shopping_queries_dataset_products.parquet'
    inputs.extend([ex_path,prod_path])
    ex=pd.read_parquet(ex_path,columns=['example_id','query','query_id','product_id','product_locale','esci_label','small_version','split'])
    ex=ex[(ex.product_locale=='us') & ex.small_version].copy()
    old=ex.query_id.astype(str).isin(hist['query_ids']) | ex['query'].map(lambda q:normalize(q) in hist['queries'])
    historical_products=set(ex.loc[old,'product_id'])
    historical_source_splits=dict(Counter(ex.loc[old,'split']))
    products=pd.read_parquet(prod_path,columns=['product_id','product_locale','product_title'])
    products=products[products.product_locale=='us'].drop_duplicates('product_id').set_index('product_id').product_title.to_dict()
    labels={'E':'exact','S':'substitute','C':'complement','I':'irrelevant'}
    partitions={s:[] for s in ['train','dev','test']};taken_products=set()
    for split,limit in [('train',6000),('dev',600),('test',1200)]:
        pool=ex[ex.split==('train' if split=='train' else 'test')]
        selected=[]
        grouped=pool.groupby('query_id',sort=False)
        ordered=sorted(grouped.groups,key=lambda q:digest(['relevance-'+split,'esci:query:'+str(q)]))
        for qid in ordered:
            frame=grouped.get_group(qid)
            if len(selected)+len(frame)>limit:continue
            if split!='train':
                if str(qid) in hist['query_ids'] or normalize(str(frame.iloc[0]['query'])) in hist['queries']:continue
                if set(frame.product_id)&(historical_products|taken_products):continue
                if ('dev' if int(digest(['esci-split',str(qid)])[:8],16)%3==0 else 'test')!=split:continue
            if frame.product_id.duplicated().any():continue
            batch=[]
            for rec in frame.to_dict('records'):
                title=products.get(rec['product_id'])
                if not isinstance(title,str) or not title.strip():break
                r=row('relevance','ESCI_US_SMALL',rec['example_id'],{'query':rec['query'],'product_title':title},labels[rec['esci_label']],
                      ['esci:query:'+str(qid),'esci:product:'+str(rec['product_id'])],split)
                r['sampling_group']='esci:query:'+str(qid);r['publisher_split']=rec['split'];batch.append(r)
            if len(batch)==len(frame):selected.extend(batch)
            if len(selected)==limit:break
        partitions[split]=selected
        for r in selected:taken_products.update(g.removeprefix('esci:product:') for g in r['groups'] if g.startswith('esci:product:'))
    return partitions,{'historical_query_rows_by_publisher_split':historical_source_splits,'historical_product_exclusion_count':len(historical_products),'sampling':'whole queries; natural label mix; cross-partition product exclusion; local subset, not official full benchmark'}


def identity(hist, inputs):
    paths=list(sorted((ROOT/'data/wdc').glob('*.json.gz')));inputs.extend(paths)
    files={p.name:read_jsonl(p) for p in paths}
    banned=set()
    for records in files.values():
        for r in records:
            for side in ['left','right']:
                title=r.get('title_'+side)
                if isinstance(title,str) and normalize(title) in hist['identity_titles']:banned.add(str(r['cluster_id_'+side]))
    pools={'train':files['wdcproducts80cc20rnd000un_train_large.json.gz'],
           'dev':files['wdcproducts80cc20rnd100un_gs.json.gz'],
           'test':files['wdcproducts80cc20rnd100un_gs.json.gz']}
    out={};seen_clusters=set();seen_texts=set()
    for split,limit in [('train',6000),('dev',500),('test',1000)]:
        candidates=[]
        for x in pools[split]:
            clusters={str(x['cluster_id_left']),str(x['cluster_id_right'])}
            a,b=x.get('title_left'),x.get('title_right')
            if not isinstance(a,str) or not isinstance(b,str) or not a.strip() or not b.strip():continue
            key=digest(sorted([normalize(a),normalize(b)]))
            if split!='train':
                if clusters & (banned|seen_clusters) or key in seen_texts:continue
                roles={'dev' if int(digest(['wdc-split',c])[:8],16)%3==0 else 'test' for c in clusters}
                if roles!={split}:continue
            r=row('identity','WDC_PRODUCTS_80PAIR',x['pair_id'],{'listing_a':a,'listing_b':b},'same' if x['label']==1 else 'distinct',
                  ['wdc:cluster:'+c for c in clusters]+['wdc:text-pair:'+key],split)
            r['sampling_group']='wdc:pair:'+str(x['pair_id']);candidates.append(r)
        # Identical text pairs carry one observation; contradictory gold is rejected.
        keyed=defaultdict(list)
        for r in candidates:keyed[digest(sorted([normalize(r['input']['listing_a']),normalize(r['input']['listing_b'])]))].append(r)
        candidates=[sorted(rs,key=lambda r:r['id'])[0] for rs in keyed.values() if len({r['gold'] for r in rs})==1]
        out[split]=take_groups(candidates,limit,'identity-'+split)
        for r in out[split]:
            seen_clusters.update(g.removeprefix('wdc:cluster:') for g in r['groups'] if g.startswith('wdc:cluster:'))
            seen_texts.update(g.removeprefix('wdc:text-pair:') for g in r['groups'] if g.startswith('wdc:text-pair:'))
    return out,{'historically_exposed_clusters':len(banned),'sampling':'publisher train and 100%-unseen gold source; entity hashes split dev/test; cross-role edges excluded; natural label proportions'}


def english(record, key):
    return [r['value'] for r in record.get(key,[]) if isinstance(r.get('value'),str) and (r.get('language_tag','en').startswith('en'))]


def extraction(hist, inputs):
    full=ROOT/'research_v2/work/abo_full'
    paths=sorted((full if full.exists() else ROOT/'data/abo').glob('listings_*.json.gz'));inputs.extend(paths)
    components=Components();records=[];old_keys=set()
    for p in paths:
        with gzip.open(p,'rt') as stream:
            for line in stream:
                x=json.loads(line);titles=english(x,'item_name');brands=english(x,'brand');colors=english(x,'color')
                if not titles:continue
                title=titles[0];text='. '.join([title]+english(x,'bullet_point'))
                brand=brands[0] if brands else None;color=colors[0] if colors else None
                keys=['abo:item:'+x['item_id'],'abo:text:'+digest(normalize(text))]
                if x.get('main_image_id'):keys.append('abo:image:'+x['main_image_id'])
                if brand:
                    # Model names can be generic categories (e.g. "shirt").
                    # Use manufacturer model identifiers, not category-like names.
                    for field in ['model_number']:
                        for value in english(x,field):
                            if normalize(value) not in {'n/a','na','unknown','none','0'}:keys.append('abo:'+field+':'+digest([normalize(brand),normalize(value)]))
                components.join(keys)
                if x['item_id'] in hist['abo_ids'] or normalize(text) in hist['abo_texts']:old_keys.update(keys)
                if not brand or not color or normalize(brand) not in normalize(text) or normalize(color) not in normalize(text):continue
                rec=row('extraction','ABO',x['item_id'],{'text':text},{'brand':brand,'color':color},keys,'unassigned')
                rec['family_evidence']='metadata/image proxy; not human-verified family labels'
                records.append(rec)
    banned={components.root(k) for k in old_keys}
    by_id=defaultdict(list)
    for r in records:by_id[r['id']].append(r)
    records=[rs[0] for rs in by_id.values() if len({canonical([r['input'],r['gold']]) for r in rs})==1]
    text_groups=defaultdict(list)
    for r in records:text_groups[normalize(r['input']['text'])].append(r)
    records=[sorted(rs,key=lambda r:r['id'])[0] for rs in text_groups.values() if len({canonical(r['gold']) for r in rs})==1]
    partitions={s:[] for s in ['train','dev','test']}
    for r in records:
        component=components.root(r['groups'][0]);h=int(digest(['abo-role',component])[:8],16)%10
        # Historical exposure may contribute only to training, never to new dev/test.
        split='train' if component in banned or h<6 else ('dev' if h<8 else 'test')
        r['split']=split;r['sampling_group']='abo:component:'+digest(component)
        r['groups'].append(r['sampling_group']);partitions[split].append(r)
    for split,limit in [('train',6000),('dev',400),('test',800)]:
        partitions[split]=take_groups(partitions[split],limit,'extraction-'+split)
    return partitions,{'source_shards':len(paths),'historically_exposed_components':len(banned),'candidate_components':len({components.root(r['groups'][0]) for r in records}),'family_independence':'manufacturer model-number/image connected components only; generic model names excluded; true product-family independence requires annotation','grounding':'nonempty brand and color must occur after NFKC/case/whitespace normalization','null_cases':'not covered by this initial publisher-derived subset'}


def main():
    from research_v2.execution import require_runpod
    require_runpod()
    if (OUT/'selection_lock.json').exists() or (OUT/'test_access.json').exists():
        raise SystemExit('A selection/test access lock exists; create a new version instead of changing this suite.')
    print('Reading historical exposure inventory',flush=True)
    hist=historical();inputs=[ROOT/'data'/name for name in ['training_mixture_v1.jsonl','training_mixture_final.jsonl','training_mixture_v3.jsonl','relevance_rows_v3.jsonl','frontier_eval_relevance.json','frontier_eval_identity.json','understand_train.jsonl','understand_dev.jsonl','abo/understand_eval_grounded.json']];partitions={s:[] for s in ['train','dev','test']};notes={}
    for name,fn in [('relevance',relevance),('identity',identity),('extraction',extraction)]:
        print('Building '+name,flush=True)
        split_rows,notes[name]=fn(hist,inputs)
        for split,records in split_rows.items():partitions[split].extend(records)
        print(name,{s:len(v) for s,v in split_rows.items()},flush=True)
    conflicts=group_overlap(partitions)
    if conflicts:raise ValueError(f'Cross-split group overlap: {len(conflicts)}')
    issues=[]
    for split,records in partitions.items():
        validate_contract(records)
        for task in ['relevance','identity','extraction']:
            rr=[r for r in records if r['task']==task]
            if len(rr)<100:issues.append(f'{split}/{task}: fewer than 100 rows')
            if task!='extraction':
                counts=Counter(r['gold'] for r in rr)
                if any(counts[label]<10 for label in LABELS[task]):issues.append(f'{split}/{task}: class support below 10 for at least one required label')
    counts={s:dict(Counter(r['task'] for r in rows)) for s,rows in partitions.items()}
    label_counts={s:{t:dict(Counter(r['gold'] for r in rows if r['task']==t)) for t in ['relevance','identity']} for s,rows in partitions.items()}
    for split in ['train','dev']:write_jsonl(OUT/(split+'.jsonl'),partitions[split])
    write_jsonl(OUT/'test.inputs.jsonl',[{k:v for k,v in r.items() if k!='gold'} for r in partitions['test']])
    write_jsonl(OUT/'private/test.gold.jsonl',partitions['test'])
    manifest={'schema_version':SCHEMA_VERSION,'suite_id':'real_sources_v2','scope':'local corpus subsets; not official benchmark scores',
              'status':'MACHINE_GATES_PASS' if not issues else 'BLOCKED','blocking_issues':issues,'counts':counts,'labels':label_counts,
              'partition_contract_hashes':{s:digest(rows) for s,rows in partitions.items()},'cross_partition_group_overlap':0,
              'source_sha256':{str(p.relative_to(ROOT)):file_sha(p) for p in sorted(set(inputs))},
              'notes':notes,'synthetic_tasks':'BLOCKED: need new scenario-disjoint, independently reviewed data',
              'human_naturalness_validation':'NOT_PERFORMED','historical_exposure_policy':'all prior train/dev/eval inputs excluded from new evaluation by available identifiers and group proxies',
              'test_status':'NOT_RUN; test labels isolated from training inputs'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    PUBLIC.mkdir(exist_ok=True)
    (PUBLIC/'real_sources_v2.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'issues':issues,'counts':counts,'labels':label_counts},indent=2))


if __name__=='__main__':main()
