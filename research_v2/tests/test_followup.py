import unittest
from research_v2.run_local import legacy_prompt
from research_v2.core import parse_label,score_rows

class FollowupTests(unittest.TestCase):
    def test_legacy_prompt_preserves_training_interface(self):
        row={'task':'identity','input':{'listing_a':'A','listing_b':'B'}}
        self.assertEqual(legacy_prompt(row),'Are these listings the same purchasable item or distinct?\nListing A: A\nListing B: B\nAnswer:')

    def test_opposite_labels_are_not_substrings(self):
        self.assertEqual(parse_label('incompatible','compatibility'),'incompatible')
        self.assertIsNone(parse_label('not same','identity'))
        self.assertIsNone(parse_label('same or distinct','identity'))

    def test_failed_rows_stay_in_denominator(self):
        rows=[{'id':'a','task':'identity','gold':'same'}, {'id':'b','task':'identity','gold':'same'}]
        outputs=[{'id':'a','status':'ok','raw_output':'same'}, {'id':'b','status':'error','raw_output':None}]
        self.assertEqual(score_rows(rows,outputs)['identity']['accuracy'],.5)
        with self.assertRaises(ValueError):score_rows(rows,outputs[:1])

if __name__=='__main__':unittest.main()
