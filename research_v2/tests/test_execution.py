import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_v2.core import canonical, digest
from research_v2.execution import require_runpod
from research_v2.experiment import read_rows, verify_suite, freeze, open_test
from research_v2.run_local import select_dev


def row(split):
    return {'id':split,'task':'identity','source':'fixture','split':split,
            'groups':['group:'+split],'input':{'listing_a':'A','listing_b':'B'},'gold':'same'}


def fixture(folder):
    (folder/'private').mkdir()
    partitions={s:[row(s)] for s in ['train','dev','test']}
    for split,rows in partitions.items():
        path=folder/(split+'.jsonl' if split!='test' else 'private/test.gold.jsonl')
        path.write_text(''.join(canonical(r)+'\n' for r in rows))
    inputs=[{k:v for k,v in r.items() if k!='gold'} for r in partitions['test']]
    (folder/'test.inputs.jsonl').write_text(''.join(canonical(r)+'\n' for r in inputs))
    manifest={'status':'MACHINE_GATES_PASS','blocking_issues':[],
              'partition_contract_hashes':{s:digest(rr) for s,rr in partitions.items()}}
    (folder/'manifest.json').write_text(json.dumps(manifest))


class ExecutionTests(unittest.TestCase):
    def test_laptop_is_rejected_even_if_pod_environment_is_spoofed(self):
        with patch('research_v2.execution.platform.system',return_value='Darwin'),patch.dict('os.environ',{'RUNPOD_POD_ID':'fixture'}):
            with self.assertRaises(RuntimeError):require_runpod()

    def test_linux_without_pod_is_rejected(self):
        with patch('research_v2.execution.platform.system',return_value='Linux'),patch.dict('os.environ',{},clear=True):
            with self.assertRaises(RuntimeError):require_runpod()

    def test_json_embedded_unicode_line_separator_is_not_a_record_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'rows.jsonl';value={'text':'A\u2028B\u0085C'}
            p.write_text(canonical(value)+'\n')
            self.assertEqual(read_rows(p),[value])

    def test_development_runner_rejects_train_and_test_rows(self):
        for split in ['train','test']:
            with self.assertRaises(ValueError):select_dev([row(split)],1)
        self.assertEqual(select_dev([row('dev')],1),[row('dev')])

    def test_modified_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);fixture(folder);verify_suite(folder)
            changed=row('dev');changed['gold']='distinct'
            (folder/'dev.jsonl').write_text(canonical(changed)+'\n')
            with self.assertRaises(ValueError):verify_suite(folder)

    def test_test_access_requires_exact_frozen_cohort_and_is_one_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);fixture(folder)
            cohort={'selection_policy':'development_only_then_one_test_pass','candidates':[
                {'name':name,'model_identity':'fixture','configuration_sha256':'a'*64,
                 'selection_dev_report_sha256':'b'*64,'prompt_sha256':'c'*64,
                 'decode':{'temperature':0,'max_new_tokens':8}} for name in ['a','b']]}
            cp=folder/'cohort.json';cp.write_text(json.dumps(cohort));freeze(folder,cp)
            with self.assertRaises(ValueError):open_test(folder,'wrong-cohort')
            self.assertFalse((folder/'test_access.json').exists())
            self.assertEqual(open_test(folder,digest(cohort)),[row('test')])
            with self.assertRaises(FileExistsError):open_test(folder,digest(cohort))
            with self.assertRaises(ValueError):freeze(folder,cp)


if __name__=='__main__':unittest.main()
