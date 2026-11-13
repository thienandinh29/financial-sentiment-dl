import unittest
from scripts.summarize_manual_check import summarize


class ManualReviewTests(unittest.TestCase):
    def test_complete_counts_and_incomplete_rejected(self):
        rows = [{'id_a': f'{kind}-{i}', 'id_b': f'{kind}-other-{i}', 'type': kind,
                 'is_duplicate': 'yes' if i < 40 else 'no'}
                for kind in ('lexical', 'semantic_only') for i in range(50)]
        self.assertEqual(summarize(rows)['semantic_only']['precision'], .8)
        rows[0]['is_duplicate'] = ''
        with self.assertRaises(ValueError):
            summarize(rows)
        rows[0]['is_duplicate'] = 'yes'
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]])


if __name__ == '__main__':
    unittest.main()
