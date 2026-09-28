"""Versioned input contracts, exact parsing, and paired evaluation. Standard library only."""
from __future__ import annotations
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 'commerce-research-v2.1'
LABELS = {
    'relevance': ('exact', 'substitute', 'complement', 'irrelevant'),
    'identity': ('same', 'distinct'),
    'functional': ('substitute', 'complement', 'unrelated', 'unknown'),
    'compatibility': ('compatible', 'incompatible', 'unknown'),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(value):
    """Conservative normalization: no substring credit, synonyms, or unit guessing."""
    if not isinstance(value, str):
        raise TypeError('Expected a string')
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def parse_label(raw, task):
    if not isinstance(raw, str):
        return None
    answer = normalize(raw)
    # Permit a single terminal period, but no explanatory text or multiple labels.
    if answer.endswith('.'):
        answer = answer[:-1].strip()
    return answer if answer in LABELS[task] else None


def parse_fields(raw):
    def no_duplicates(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError('Duplicate JSON key')
            out[k] = v
        return out
    try:
        obj = json.loads(raw, object_pairs_hook=no_duplicates)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {'brand', 'color'}:
        return None
    if any(v is not None and (not isinstance(v, str) or not v.strip()) for v in obj.values()):
        return None
    return obj


def classification(gold, pred, labels):
    if len(gold) != len(pred) or not gold:
        raise ValueError('Expected nonempty aligned predictions')
    if any(g not in labels for g in gold):
        raise ValueError('Gold label is outside the ontology')
    cells = Counter((g, p if p in labels else 'INVALID') for g, p in zip(gold, pred))
    per = {}
    for label in labels:
        support = sum(g == label for g in gold)
        predicted = sum(p == label for p in pred)
        tp = cells[label, label]
        per[label] = {'support': support, 'predicted': predicted,
                      'precision': tp/predicted if predicted else 0.,
                      'recall': tp/support if support else 0.,
                      'f1': 2*tp/(support+predicted) if support+predicted else 0.}
    return {'n': len(gold), 'accuracy': sum(g == p for g, p in zip(gold, pred))/len(gold),
            'macro_f1': sum(r['f1'] for r in per.values())/len(labels),
            'balanced_accuracy': sum(r['recall'] for r in per.values())/len(labels),
            'invalid_rate': sum(p not in labels for p in pred)/len(pred),
            'per_class': per, 'confusion': {f'{a}->{b}': n for (a,b),n in sorted(cells.items())}}


def field_counts(gold, pred, normalize_values=True):
    """Null is absence, not a false negative when the gold field is absent."""
    if len(gold) != len(pred):
        raise ValueError('Unaligned fields')
    tp = fp = fn = tn = 0
    transform = normalize if normalize_values else (lambda x: x)
    for g, p in zip(gold, pred):
        if g is None:
            tn += p is None
            fp += p is not None
        elif p is None:
            fn += 1
        elif transform(g) == transform(p):
            tp += 1
        else:
            fp += 1
            fn += 1
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def score_rows(rows, outputs):
    """A complete one-to-one join is mandatory; failures stay in the denominator."""
    expected = {r['id']: r for r in rows}
    actual = {r['id']: r for r in outputs}
    if len(expected) != len(rows) or len(actual) != len(outputs) or set(expected) != set(actual):
        raise ValueError('Duplicate, missing, or unexpected example IDs')
    tasks = {}
    for task in sorted({r['task'] for r in rows}):
        subset = [r for r in rows if r['task'] == task]
        raw = [actual[r['id']]['raw_output'] if actual[r['id']].get('status') == 'ok' else None for r in subset]
        if task == 'extraction':
            parsed = [parse_fields(x) for x in raw]
            task_result = {'n':len(subset),'parse_error_rate':sum(x is None for x in parsed)/len(subset)}
            for mode in ['exact','normalized_exact']:
                task_result[mode] = {field:field_counts([r['gold'][field] for r in subset],
                    [p[field] if p is not None else None for p in parsed], mode=='normalized_exact')
                    for field in ['brand','color']}
            task_result['joint_accuracy'] = sum(p is not None and all(
                (p[k] is None and r['gold'][k] is None) or
                (p[k] is not None and r['gold'][k] is not None and normalize(p[k])==normalize(r['gold'][k]))
                for k in ['brand','color']) for p,r in zip(parsed,subset))/len(subset)
            tasks[task] = task_result
        else:
            tasks[task] = classification([r['gold'] for r in subset],
                                         [parse_label(x,task) for x in raw], LABELS[task])
    return tasks


def prompt(row):
    """All backends receive these same evidence fields and label definitions."""
    x, task = row['input'], row['task']
    if task == 'relevance':
        return ('Classify query-product relevance. exact: satisfies the query; substitute: a reasonable alternative; '
                'complement: an accessory or item used with the requested product; irrelevant: none of these. '
                'Return only one label: exact, substitute, complement, irrelevant.\n'
                f"Query: {x['query']}\nProduct: {x['product_title']}\nAnswer:")
    if task == 'identity':
        return ('Do these listings describe the same purchasable item and variant? '
                'Return only same or distinct.\n'
                f"Listing A: {x['listing_a']}\nListing B: {x['listing_b']}\nAnswer:")
    if task == 'extraction':
        return ('Extract brand and color explicitly supported by the listing. Do not guess absent values. '
                'Return only a JSON object with exactly brand and color as strings or null.\n'
                f"Listing: {x['text']}\nAnswer:")
    if task in ('functional','compatibility'):
        return (f"Classify {task}. Return only one label: {', '.join(LABELS[task])}. "
                'Use unknown if the evidence is insufficient.\n'
                f"Context: {x['context']}\nListing A: {x['listing_a']}\nListing B: {x['listing_b']}\nAnswer:")
    raise ValueError(task)


def validate_contract(rows, require_gold=True):
    if not rows:
        raise ValueError('Empty evaluation contract')
    ids = set()
    for r in rows:
        for key in ['id','task','input','groups','source','split']:
            if key not in r:
                raise ValueError(f'Missing {key}')
        if r['id'] in ids or not r['groups'] or r['split'] not in {'train','dev','test'}:
            raise ValueError('Duplicate ID or missing group metadata')
        ids.add(r['id'])
        prompt(r)
        if require_gold:
            if r['task']=='extraction':
                if parse_fields(json.dumps(r['gold'])) is None:
                    raise ValueError('Malformed extraction gold')
            elif r['gold'] not in LABELS[r['task']]:
                raise ValueError('Gold outside ontology')
    return digest(rows)


def group_overlap(partitions):
    ownership = {}
    conflicts = []
    for split, rows in partitions.items():
        for row in rows:
            for group in row['groups']:
                previous = ownership.setdefault(group, split)
                if previous != split:
                    conflicts.append({'group':group,'left':previous,'right':split,'row_id':row['id']})
    return conflicts


def paired_correctness(rows, left, right):
    """Require aligned contracts before pairing; fail on unobserved rows."""
    for values in (left,right):
        score_rows(rows,values)
    lhs={r['id']:r for r in left};rhs={r['id']:r for r in right}
    if len({r['task'] for r in rows}) != 1:
        raise ValueError('Pair task-specific outcomes; do not pool different tasks')
    # Connect all shared entities/queries, rather than arbitrarily taking one key.
    parent = {}
    def root(key):
        parent.setdefault(key,key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for r in rows:
        roots = sorted({root(g) for g in r['groups']})
        for key in roots[1:]:
            parent[key] = roots[0]
    pairs=[]
    for row in rows:
        def correct(output):
            result=score_rows([row],[output])[row['task']]
            return result['joint_accuracy'] if row['task']=='extraction' else result['accuracy']
        pairs.append((row['id'],root(row['groups'][0]),correct(lhs[row['id']]),correct(rhs[row['id']])))
    b=sum(bool(a) and not bool(c) for _,_,a,c in pairs);c=sum(bool(d) and not bool(a) for _,_,a,d in pairs)
    n=b+c
    exact=min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.
    groups={}
    for _,group,a,bval in pairs:
        groups.setdefault(group,[]).append(a-bval)
    rng=random.Random(20260928);keys=sorted(groups)
    boot=[]
    for _ in range(2000 if len(keys)>=20 else 0):
        sample=[v for k in rng.choices(keys,k=len(keys)) for v in groups[k]]
        boot.append(sum(sample)/len(sample))
    boot.sort()
    return {'n':len(pairs),'groups':len(keys),'left_only_correct':b,'right_only_correct':c,
            'accuracy_delta':sum(a-bval for _,_,a,bval in pairs)/len(pairs),
            'exact_mcnemar_p_unadjusted_iid_only':exact,
            'cluster_bootstrap_95_percentile':[boot[49],boot[1949]] if boot else None,
            'note':'Binary/joint correctness, not field F1. McNemar assumes independent rows. Bootstrap connects every shared group; suppressed below 20 components.'}
