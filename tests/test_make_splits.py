import unittest
import tempfile
from pathlib import Path
from scripts.make_splits import assign_groups, describe, verify_distribution, verify_pair_clusters


class SplitTests(unittest.TestCase):
    def test_reject_size_or_label_imbalance(self):
        balanced = {'train': {'size': 700}, 'validation': {'size': 150}, 'test': {'size': 150}}
        verify_distribution('example', balanced, 2.0)
        with self.assertRaisesRegex(ValueError, 'label drift'):
            verify_distribution('example', balanced, 2.01)
        unbalanced = {'train': {'size': 750}, 'validation': {'size': 100}, 'test': {'size': 150}}
        with self.assertRaisesRegex(ValueError, 'size drift'):
            verify_distribution('example', unbalanced, 0.0)

    def test_pair_artifacts_must_match_clusters(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory) / 'pairs.csv'
            pairs.write_text('id_a,id_b\na,b\n', encoding='utf-8')
            verify_pair_clusters(pairs, {'a': 'one', 'b': 'one'})
            with self.assertRaisesRegex(ValueError, 'spans clusters'):
                verify_pair_clusters(pairs, {'a': 'one', 'b': 'two'})
            with self.assertRaisesRegex(ValueError, 'absent from clusters'):
                verify_pair_clusters(pairs, {'a': 'one'})

    def test_clusters_determinism_and_label_balance(self):
        groups = [[f"{label}-{group}-{n}" for n in range(2)]
                  for label in range(3) for group in range(100)]
        labels = {item: int(item.split('-')[0]) for group in groups for item in group}
        first = assign_groups(groups, labels, 1)
        self.assertEqual(first, assign_groups(groups, labels, 1))
        self.assertEqual(set(first), set(labels))
        for group in groups:
            self.assertEqual(len({first[item] for item in group}), 1)
        report, drift = describe(first, labels)
        self.assertLessEqual(drift, 2)
        for split, ratio in (("train", .7), ("validation", .15), ("test", .15)):
            self.assertAlmostEqual(report[split]["size"] / len(labels), ratio, delta=.02)


if __name__ == '__main__':
    unittest.main()
