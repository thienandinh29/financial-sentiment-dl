import csv
import tempfile
import unittest
from pathlib import Path
from scripts.make_splits import verify_manual_review


class ReviewGateTests(unittest.TestCase):
    def test_completed_matching_sample_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs, review = root / 'pairs.csv', root / 'review.csv'
            with pairs.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id_a', 'id_b', 'jaccard'])
                writer.writerows((f'{kind}-{i}', f'{kind}-other-{i}', '0.9' if kind == 'lexical' else '')
                                 for kind in ('lexical', 'semantic_only') for i in range(50))
            rows = [(f'{kind}-{i}', f'{kind}-other-{i}', kind, 'yes')
                    for kind in ('lexical', 'semantic_only') for i in range(50)]
            def write_review():
                with review.open('w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id_a', 'id_b', 'type', 'is_duplicate'])
                    writer.writerows(rows)
            write_review()
            self.assertEqual(verify_manual_review(review, pairs)['lexical']['yes'], 50)
            rows[0] = (*rows[0][:3], '')
            write_review()
            with self.assertRaises(ValueError):
                verify_manual_review(review, pairs)
            rows[0] = ('unrelated', 'other', 'lexical', 'yes')
            write_review()
            with self.assertRaises(ValueError):
                verify_manual_review(review, pairs)


if __name__ == '__main__':
    unittest.main()
