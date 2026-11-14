import unittest
import tempfile
from pathlib import Path
from scripts.summarize_manual_check import read_review, summarize


class ManualReviewTests(unittest.TestCase):
    def test_read_both_delimiters_and_quoted_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'review.csv'
            for delimiter in (',', ';'):
                path.write_text(f'id_a{delimiter}note\na{delimiter}"one{delimiter}two"\n', encoding='utf-8')
                self.assertEqual(read_review(path), [{'id_a': 'a', 'note': f'one{delimiter}two'}])

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
