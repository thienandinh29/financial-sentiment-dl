import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from train.data import sha256_file, write_split

class LeakageCliTests(unittest.TestCase):
    def test_hash_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            split = root / "split.jsonl"
            ids = [f"v{i}" for i in range(20)]
            digest = write_split(split, [("t", "train")] + [(i, "validation") for i in ids] + [("x", "test")])
            pairs = root / "pairs.csv"
            pairs.write_text("id_a,id_b,cosine\n", encoding="utf-8")
            clusters = root / "clusters.csv"
            with clusters.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["id", "cluster_id"]); w.writerows((i, i) for i in ["t", *ids, "x"])
            args = [sys.executable, "-m", "scripts.make_leakage", "--split", str(split), "--split-sha256", digest, "--pairs", str(pairs), "--pairs-sha256", sha256_file(pairs), "--clusters", str(clusters), "--clusters-sha256", sha256_file(clusters), "--output-dir"]
            bad = args.copy(); bad[bad.index("--pairs-sha256") + 1] = "bad"
            self.assertNotEqual(subprocess.run(bad + [str(root / "bad")], capture_output=True).returncode, 0)
            for name in ("one", "two"):
                self.assertEqual(subprocess.run(args + [str(root / name)], capture_output=True).returncode, 0)
            for name in ("leakage_5.json", "leakage_10.json", "leakage_20.json", "SHA256SUMS"):
                self.assertEqual((root / "one" / name).read_bytes(), (root / "two" / name).read_bytes())
            self.assertNotEqual(subprocess.run(args + [str(root / "one")], capture_output=True).returncode, 0)
            self.assertEqual(sha256_file(split), digest)
