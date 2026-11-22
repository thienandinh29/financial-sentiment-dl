import collections
import unittest

from scripts.pick300 import build_assignment


class Pick300Tests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'id': f'id{i:04}', 'url': f'https://example.org/{i}',
                      'pool': 'test_candidate' if i < 350 else 'train_unlabeled'}
                     for i in range(400)]
        self.excluded = {f'id{i:04}' for i in range(30)}

    def test_selection_and_balanced_assignments(self):
        selected, assignment = build_assignment(self.rows, self.excluded)
        self.assertEqual((selected, assignment), build_assignment(reversed(self.rows), self.excluded))
        self.assertEqual(len(selected), 300)
        self.assertTrue(self.excluded.isdisjoint(identifier for identifier, _ in selected))
        counts = collections.Counter(identifier for identifier, _, _ in assignment)
        self.assertEqual(collections.Counter(counts.values()), {1: 200, 2: 100})
        groups = collections.defaultdict(list)
        for identifier, annotator, round_no in assignment:
            groups[identifier].append((annotator, round_no))
        pairs = collections.Counter(tuple(sorted(a for a, _ in group))
                                    for group in groups.values() if len(group) == 2)
        self.assertEqual(sorted(pairs.values()), [33, 33, 34])
        self.assertTrue(all(len({r for _, r in group}) == 1 for group in groups.values()))
        counts = collections.Counter((a, r) for _, a, r in assignment)
        self.assertEqual([counts[(f'tv{i}', 1)] for i in (1, 2, 3)], [69, 68, 68])
        self.assertEqual([counts[(f'tv{i}', 2)] for i in (1, 2, 3)], [65, 66, 64])

    def test_too_few_or_duplicate(self):
        with self.assertRaises(ValueError):
            build_assignment(self.rows[:300], self.excluded)
        with self.assertRaises(ValueError):
            build_assignment(self.rows + [self.rows[0]], self.excluded)


if __name__ == '__main__':
    unittest.main()
