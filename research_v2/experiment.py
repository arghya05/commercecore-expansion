"""Fail-closed suite verification and development-only selection locks.

Locks are tamper-evident workflow controls, not a security boundary against the owner.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from research_v2.core import canonical, digest, file_sha, validate_contract, group_overlap


def read_rows(path):
    # str.splitlines also splits valid JSON string characters such as U+2028.
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_suite(folder, require_ready=True):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    if require_ready and manifest['status']!='MACHINE_GATES_PASS':
        raise ValueError('Data machine gates failed: '+str(manifest['blocking_issues']))
    rows={s:read_rows(folder/(s+'.jsonl' if s!='test' else 'private/test.gold.jsonl')) for s in ['train','dev','test']}
    for split,rr in rows.items():
        validate_contract(rr)
        if any(r['split']!=split for r in rr):raise ValueError('Partition label mismatch')
        if digest(rr)!=manifest['partition_contract_hashes'][split]:raise ValueError('Modified '+split+' contract')
    if group_overlap(rows):raise ValueError('Cross-partition group overlap')
    test_inputs=read_rows(folder/'test.inputs.jsonl')
    if test_inputs!=[{k:v for k,v in r.items() if k!='gold'} for r in rows['test']]:
        raise ValueError('Test input contract mismatch')
    return manifest,rows


def training_gate(folder, review_file):
    """A human review is required in addition to machine checks; no training is launched."""
    manifest,rows=verify_suite(folder)
    review=json.loads(Path(review_file).read_text())
    if review.get('manifest_sha256')!=file_sha(Path(folder)/'manifest.json'):
        raise ValueError('Quality review covers a different suite')
    if review.get('decision')!='APPROVE' or not review.get('reviewers') or review.get('open_critical_issues')!=0:
        raise ValueError('Human quality gate is not approved')
    for task in sorted({r['task'] for r in rows['train']}):
        item=review.get('tasks',{}).get(task,{})
        # Thresholds are declared requirements, not invented measurements.
        if item.get('sample_size',0)<100 or item.get('independent_raters',0)<2:
            raise ValueError('Insufficient independent annotation: '+task)
        if item.get('adjudicated_label_support_rate',0)<.98 or item.get('naturalness_pass_rate',0)<.95:
            raise ValueError('Annotation quality below declared threshold: '+task)
        report=Path(review_file).parent/item.get('annotation_file','')
        if not report.is_file() or file_sha(report)!=item.get('annotation_sha256'):
            raise ValueError('Missing or modified annotation evidence: '+task)
    return {'status':'APPROVED_FOR_DECLARED_TASKS','tasks':sorted(review['tasks']),
            'manifest_sha256':file_sha(Path(folder)/'manifest.json'),'review_sha256':file_sha(review_file)}


def freeze(folder, cohort_path):
    folder=Path(folder);manifest,_=verify_suite(folder)
    if (folder/'test_access.json').exists():raise ValueError('Test already opened; create a new experiment version')
    cohort=json.loads(Path(cohort_path).read_text())
    candidates=cohort.get('candidates',[])
    if len(candidates)<2 or len({c['name'] for c in candidates})!=len(candidates):
        raise ValueError('Need at least two uniquely named candidates')
    for c in candidates:
        for key in ['model_identity','configuration_sha256','selection_dev_report_sha256','prompt_sha256','decode']:
            if not c.get(key):raise ValueError('Unfrozen candidate '+c['name']+': '+key)
        if c['decode'].get('temperature')!=0 or c['decode'].get('max_new_tokens',0)<=0:
            raise ValueError('Declare deterministic decoding and a positive output cap')
    if cohort.get('selection_policy')!='development_only_then_one_test_pass':
        raise ValueError('Test-adaptive selection is forbidden')
    lock={'created_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':file_sha(folder/'manifest.json'),
          'test_contract_sha256':manifest['partition_contract_hashes']['test'],'cohort':cohort}
    with (folder/'selection_lock.json').open('x') as stream:stream.write(json.dumps(lock,indent=2)+'\n')
    return lock


def open_test(folder, cohort_hash):
    """Record access BEFORE returning gold; exactly one access per frozen experiment."""
    folder=Path(folder);manifest,_=verify_suite(folder)
    lock=json.loads((folder/'selection_lock.json').read_text())
    if lock['manifest_sha256']!=file_sha(folder/'manifest.json') or digest(lock['cohort'])!=cohort_hash:
        raise ValueError('Changed suite or candidate cohort')
    event={'opened_utc':datetime.now(timezone.utc).isoformat(),'cohort_sha256':cohort_hash,
           'test_contract_sha256':manifest['partition_contract_hashes']['test'],
           'policy':'No selection or retraining from these test results; record failures and deviations.'}
    with (folder/'test_access.json').open('x') as stream:stream.write(json.dumps(event,indent=2)+'\n')
    return read_rows(folder/'private/test.gold.jsonl')


def main():
    from research_v2.execution import require_runpod
    require_runpod()
    p=argparse.ArgumentParser();p.add_argument('command',choices=['verify','training-gate','freeze'])
    p.add_argument('--suite',default='research_v2/work/suite_v2');p.add_argument('--review');p.add_argument('--cohort')
    a=p.parse_args()
    if a.command=='verify':
        manifest,_=verify_suite(a.suite);print(json.dumps({'status':manifest['status'],'counts':manifest['counts']},indent=2))
    elif a.command=='training-gate':print(json.dumps(training_gate(a.suite,a.review),indent=2))
    else:print(json.dumps(freeze(a.suite,a.cohort),indent=2))


if __name__=='__main__':main()
