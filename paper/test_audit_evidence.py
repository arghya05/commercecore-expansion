"""Offline scientific checks: independent metric identities and real scorer fixtures."""
import ast
import itertools
import math
import unittest
from pathlib import Path

from audit_evidence import aggregate_pairing_bounds, classification, exact_mcnemar, fingerprint

ROOT = Path(__file__).resolve().parents[1]


def pure_function(path, name):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class EvidenceTests(unittest.TestCase):
    def test_confusion_reconstruction_with_invalid_output(self):
        got = classification({'a->a': 3, 'a->b': 1, 'b->a': 1, 'b->invalid': 1}, ['a', 'b'])
        self.assertEqual(got['n'], 6)
        self.assertEqual(got['accuracy'], .5)
        self.assertEqual(got['per_class']['b']['recall'], 0)
        self.assertEqual(got['per_class']['a']['f1'], .75)
        self.assertEqual(got['macro_f1'], .375)
        self.assertEqual(got['balanced_accuracy'], .375)

    def test_pairing_bounds_exhaustively_on_small_universes(self):
        for n in range(1, 6):
            vectors = list(itertools.product((0, 1), repeat=n))
            for a in range(n+1):
                for b in range(n+1):
                    observed = set()
                    for left in vectors:
                        if sum(left) != a:
                            continue
                        for right in vectors:
                            if sum(right) == b:
                                observed.add(sum(x*y for x, y in zip(left, right)))
                    got = aggregate_pairing_bounds(n, a, b)
                    self.assertEqual({r['both_correct'] for r in got['possible_pairings']}, observed)
                    for r in got['possible_pairings']:
                        self.assertGreaterEqual(n-a-b+r['both_correct'], 0)

    def test_exact_pairing_endpoints(self):
        got = aggregate_pairing_bounds(200, 191, 184)
        self.assertEqual(len(got['possible_pairings']), 10)
        self.assertEqual(got['p_min'], .015625)
        self.assertAlmostEqual(got['p_max'], .2295229434967041)
        self.assertEqual(exact_mcnemar(0, 0), 1)
        self.assertEqual(exact_mcnemar(7, 0), exact_mcnemar(0, 7))

    def test_invalid_margins_rejected(self):
        for values in ((4, 5, 2), (-1, 0, 0), (4, 2.5, 2), (4, -1, 2)):
            with self.assertRaises(ValueError):
                aggregate_pairing_bounds(*values)

    def test_real_extraction_scorer_accepts_partial_and_overlong_fields(self):
        score = pure_function('expansion/eval/understand_locked_eval.py', 'score_field')
        self.assertEqual(score(['black'], ['black suede'])['tp'], 1)
        self.assertEqual(score(['black suede'], ['black'])['tp'], 1)
        self.assertEqual(score([None], ['black'])['fn'], 1)
        got = score(['red'], ['black'])
        self.assertEqual((got['tp'], got['fp'], got['fn']), (0, 1, 1))

    def test_real_shared_scorer_opposite_label_counterexamples(self):
        path = ROOT / 'expansion/train/train_shared_adapter.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score_dev_accuracy')
        assignment = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'correct' for t in n.targets))
        expression = compile(ast.Expression(assignment.value), str(path), 'eval')
        for gold, prediction in [('compatible', 'incompatible'), ('same', 'not same')]:
            self.assertTrue(eval(expression, {'ex': {'target': gold}, 'pred_text': prediction}))
            self.assertNotEqual(gold, prediction)

    def test_scenario_key_order_does_not_change_group(self):
        self.assertEqual(fingerprint({'a': 1, 'b': 2}), fingerprint({'b': 2, 'a': 1}))
        self.assertNotEqual(fingerprint({'a': 1, 'b': 2}), fingerprint({'a': 1, 'b': 3}))


if __name__ == '__main__':
    unittest.main()
