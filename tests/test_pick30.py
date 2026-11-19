import unittest
from scripts.pick30 import select_candidates


class TrialSampleTests(unittest.TestCase):
    def test_reproducible_and_candidates_only(self):
        rows = [{'id': str(i), 'url': f'https://example.com/{i}', 'pool': 'test_candidate', 'text': 'private'} for i in range(40)]
        rows.append({'id': 'other', 'url': 'https://example.com/other', 'pool': 'other'})
        sample = select_candidates(rows)
        self.assertEqual(sample, select_candidates(reversed(rows)))
        self.assertEqual(len(set(i for i, _ in sample)), 30)
        self.assertNotIn('other', {i for i, _ in sample})
        with self.assertRaises(ValueError):
            select_candidates(rows + [rows[0]])


if __name__ == '__main__':
    unittest.main()
