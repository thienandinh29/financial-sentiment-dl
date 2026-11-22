import unittest
from scripts.make_leakage import make_manifest


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.split = {'train': ['t'], 'validation': ['a', 'b', 'c', 'd'], 'test': ['x']}
        self.pairs = [('a', 'b'), ('c', 'a'), ('a', 'x'), ('a', 't')]

    def test_disjoint_and_direct_pairs(self):
        result = make_manifest(self.split, self.pairs, ['a'], 2)
        self.assertEqual(result['train_add_validation_ids'], ['a'])
        self.assertEqual(result['validation_evaluation_ids'], ['b', 'c', 'd'])
        self.assertEqual(result['validation_exposed_ids'], ['b', 'c'])
        self.assertEqual(result['fraction_of_frozen_validation'], .5)
        self.assertAlmostEqual(result['fraction_of_evaluation'], 2 / 3)
        self.assertEqual(self.split['test'], ['x'])

    def test_reject_invalid_donors_and_unreachable_target(self):
        for donors in ([], ['x'], ['t'], ['a', 'a']):
            with self.subTest(donors=donors), self.assertRaises(ValueError):
                make_manifest(self.split, self.pairs, donors, 2)
        with self.assertRaises(ValueError):
            make_manifest(self.split, self.pairs, ['a'], 3)

    def test_donor_not_counted_as_exposed(self):
        result = make_manifest(self.split, self.pairs, ['a', 'b'], 1)
        self.assertEqual(result['validation_exposed_ids'], ['c'])
        self.assertTrue(set(result['train_add_validation_ids']).isdisjoint(result['validation_evaluation_ids']))


if __name__ == '__main__':
    unittest.main()
