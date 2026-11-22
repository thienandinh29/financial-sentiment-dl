import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from train.data import sha256_file, write_split


class LeakageCliTests(unittest.TestCase):
    def test_hash_checks_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / 'split.jsonl'
            digest = write_split(split, [('t', 'train'), ('a', 'validation'),
                                         ('b', 'validation'), ('x', 'test')])
            pairs = root / 'pairs.csv'
            with pairs.open('w', encoding='utf-8', newline='') as f:
                w = csv.writer(f); w.writerow(['id_a', 'id_b']); w.writerow(['a', 'b'])
            donors = root / 'donors.json'
            donors.write_text(json.dumps(['a']), encoding='utf-8')
            out = root / 'out.json'
            args = [sys.executable, '-m', 'scripts.make_leakage', '--split', str(split),
                    '--split-sha256', digest, '--pairs', str(pairs), '--pairs-sha256',
                    sha256_file(pairs), '--donors', str(donors), '--requested-count', '1',
                    '--output', str(out)]
            bad = args.copy(); bad[bad.index('--pairs-sha256') + 1] = 'bad'
            self.assertNotEqual(subprocess.run(bad, capture_output=True).returncode, 0)
            self.assertFalse(out.exists())
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 0)
            original = out.read_bytes()
            self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)
            self.assertEqual(out.read_bytes(), original)
            self.assertEqual(sha256_file(split), digest)


if __name__ == '__main__':
    unittest.main()
