import json
import tempfile
import unittest
from pathlib import Path

from models.tfidf_lr import run
from train.data import write_split


class BaselineTests(unittest.TestCase):
    def test_validation_only_and_hash_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, assignments = [], []
            for index, label in enumerate(('bearish', 'neutral', 'bullish')):
                for n in range(4):
                    identifier = f'{index}-{n}'
                    rows.append({'id': identifier, 'text': f'{label} market {label}', 'label': label})
                    assignments.append((identifier, 'train' if n < 2 else 'validation' if n == 2 else 'test'))
            raw = root / 'raw.jsonl'
            raw.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            digest = write_split(root / 'split.jsonl', assignments)
            cfg = {'experiment': 'E0', 'data': {'split_file': 'split.jsonl', 'split_sha256': 'bad',
                   'raw_files': ['raw.jsonl']}, 'model': {'ngram_range': [1, 2], 'min_df': 1, 'C': 1.0}, 'seeds': [1]}
            with self.assertRaises(ValueError):
                run(cfg, root)
            cfg['data']['split_sha256'] = digest
            self.assertAlmostEqual(run(cfg, root), 1.0)
            out = root / 'results/E0/seed1'
            predictions = [json.loads(line) for line in (out / 'preds_validation.jsonl').read_text().splitlines()]
            self.assertEqual({p['id'] for p in predictions}, {'0-2', '1-2', '2-2'})
            for prediction in predictions:
                self.assertEqual(len(prediction['probs']), 3)
                self.assertAlmostEqual(sum(prediction['probs']), 1, places=5)
            self.assertFalse((out / 'preds_test.jsonl').exists())
            metadata = json.loads((out / 'run.json').read_text())
            self.assertEqual(metadata['seed'], 1)
            self.assertIn('commit', metadata)
            self.assertIn('elapsed_seconds', metadata)


if __name__ == '__main__':
    unittest.main()
