import unittest
from scripts.make_splits import assign_groups, describe


class SplitTests(unittest.TestCase):
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
