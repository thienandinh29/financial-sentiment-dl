import csv
import tempfile
import unittest
from pathlib import Path
from scripts.trial_agreement import analyze


def write(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)


class TrialAgreementTests(unittest.TestCase):
    def test_pairwise_kappa_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = root / 'trial.csv'
            write(trial, ['id', 'url'], [(str(i), f'https://example.com/{i}') for i in range(30)])
            paths = [root / f'{name}.csv' for name in ('a', 'b', 'c')]
            for n, path in enumerate(paths):
                write(path, ['id', 'label', 'reason'], [(str(i), 'bullish' if (i < 15 if n < 2 else i < 10) else 'bearish', 'reason') for i in range(30)])
            result = analyze(trial, paths)
            self.assertEqual(result['pairs'][0]['agreement'], 1)
            self.assertEqual(result['pairs'][0]['kappa'], 1)
            self.assertAlmostEqual(result['pairs'][1]['agreement'], 25 / 30)
            self.assertAlmostEqual(result['pairs'][1]['kappa'], 2 / 3)
            self.assertEqual(set(result['disagreement_ids']), {str(i) for i in range(10, 15)})
            write(paths[0], ['id', 'label', 'reason'], [('0', 'bullish', 'reason')])
            with self.assertRaises(ValueError):
                analyze(trial, paths)


if __name__ == '__main__':
    unittest.main()
