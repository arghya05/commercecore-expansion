import json
import tempfile
import unittest
from pathlib import Path
from research_v2.core import (parse_label, parse_fields, field_counts, score_rows,
                              validate_contract, group_overlap, paired_correctness, prompt)
from research_v2.build_suite import Components, take_groups


def example(id='1', gold='same', split='dev', groups=None):
    return {'id':id,'task':'identity','input':{'listing_a':'A','listing_b':'B'},'gold':gold,
            'groups':groups or ['entity:1'],'split':split,'source':'test'}


def output(id='1', raw='same',status='ok'):
    return {'id':id,'raw_output':raw,'status':status}


class ProtocolTests(unittest.TestCase):
    def test_opposite_and_negated_labels_are_not_substring_matches(self):
        self.assertEqual(parse_label('incompatible','compatibility'),'incompatible')
        self.assertIsNone(parse_label('not same','identity'))
        self.assertIsNone(parse_label('same or distinct','identity'))
        self.assertIsNone(parse_label('a: b','identity'))
        self.assertEqual(parse_label(' SAME. ','identity'),'same')

    def test_extraction_rejects_duplicate_keys_extra_fields_and_invalid_types(self):
        for raw in ['{"brand":"x","brand":"y","color":null}',
                    '{"brand":"x","color":[],"reason":"x"}',
                    '{"brand":"x","color":false}',
                    '{"brand":"x","color":""}', '```json\n{}\n```']:
            self.assertIsNone(parse_fields(raw))
        self.assertEqual(parse_fields('{"brand":null,"color":null}'),{'brand':None,'color':None})

    def test_field_exactness_and_null_denominators(self):
        wrong=field_counts(['black suede'],['black'])
        self.assertEqual((wrong['tp'],wrong['fp'],wrong['fn']),(0,1,1))
        result=field_counts([None,None,'Blue',' Red '],[None,'x',None,'red'])
        self.assertEqual((result['tp'],result['fp'],result['fn'],result['tn']),(1,1,1,1))
        self.assertEqual(field_counts(['Red'],['red'],False)['tp'],0)

    def test_failures_are_retained_in_denominator(self):
        score=score_rows([example('1'),example('2')],[output('1'),output('2',None,'timeout')])['identity']
        self.assertEqual(score['n'],2)
        self.assertEqual(score['accuracy'],.5)
        self.assertEqual(score['invalid_rate'],.5)

    def test_join_rejects_missing_duplicate_and_unexpected_rows(self):
        for outputs in [[],[output('2')],[output(),output()]]:
            with self.assertRaises(ValueError):score_rows([example()],outputs)

    def test_contract_rejects_empty_gold_outside_ontology_and_duplicate_ids(self):
        for records in [[],[example(gold='unknown')],[example(),example()]]:
            with self.assertRaises(ValueError):validate_contract(records)
        validate_contract([example()])

    def test_shared_entity_detects_overlap_despite_distinct_examples(self):
        conflicts=group_overlap({'train':[example('1',split='train',groups=['a','b'])],
                                 'test':[example('2',split='test',groups=['b','c'])]})
        self.assertEqual(conflicts[0]['group'],'b')

    def test_transitive_family_components_and_whole_group_sampling(self):
        c=Components();c.join(['id:a','image:1']);c.join(['image:1','model:m']);c.join(['id:b','model:m'])
        self.assertEqual(c.root('id:a'),c.root('id:b'))
        records=[{'sampling_group':'a','id':str(n)} for n in range(3)]+[{'sampling_group':'b','id':'4'}]
        selected=take_groups(records,2,'seed')
        self.assertEqual([r['id'] for r in selected],['4'])

    def test_pairing_connects_every_shared_group(self):
        rows=[example('1',groups=['a','b']),example('2',groups=['b','c'])]
        r=paired_correctness(rows,[output('1'),output('2')],[output('1','distinct'),output('2','distinct')])
        self.assertEqual(r['groups'],1)
        self.assertEqual(r['accuracy_delta'],1)
        self.assertIsNone(r['cluster_bootstrap_95_percentile'])

    def test_prompt_contains_no_gold_or_split(self):
        a=example();b=example(gold='distinct',split='test')
        self.assertEqual(prompt(a),prompt(b))


if __name__=='__main__':unittest.main()
