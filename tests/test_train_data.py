"""Tests for train/data.py on small synthetic files."""
import json
import tempfile
import unittest
from pathlib import Path

from train.data import LABELS, load_split, load_split_data, sha256_file, write_split

RAW = [("a", "Apple rises", "bullish"), ("b", "Oil falls", "bearish"), ("c", "Fed meets", "neutral"),
       ("d", "Gold flat", "neutral")]


class DataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.raw = root / "raw.jsonl"
        self.raw.write_text("".join(json.dumps({"id": i, "text": t, "label": l}) + "\n" for i, t, l in RAW),
                            encoding="utf-8")
        self.split = root / "split.jsonl"
        self.sha = write_split(self.split, [("a", "train"), ("b", "train"), ("c", "validation"), ("d", "test")])

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_label_order(self):
        self.assertEqual(LABELS, ("bearish", "neutral", "bullish"))

    def test_loads_train_and_validation_with_canonical_labels(self):
        data = load_split_data(self.split, self.sha, [self.raw])
        self.assertEqual(data["train"], [("a", "Apple rises", 2), ("b", "Oil falls", 0)])
        self.assertEqual(data["validation"], [("c", "Fed meets", 1)])
        self.assertNotIn("test", data)

    def test_refuses_test_split(self):
        with self.assertRaises(ValueError):
            load_split_data(self.split, self.sha, [self.raw], splits=("train", "test"))

    def test_refuses_missing_or_wrong_hash(self):
        for sha in (None, "", "0" * 64):
            with self.assertRaises(ValueError):
                load_split(self.split, sha)

    def test_refuses_text_in_split_and_duplicate_ids(self):
        bad = Path(self.tmp.name) / "bad.jsonl"
        for rows in ([{"id": "a", "split": "train", "text": "x"}],
                     [{"id": "a", "split": "train"}, {"id": "a", "split": "validation"}],
                     [{"id": "a", "split": "dev"}]):
            bad.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_split(bad, sha256_file(bad))

    def test_ids_missing_from_raw_raise(self):
        split = Path(self.tmp.name) / "s2.jsonl"
        sha = write_split(split, [("a", "train"), ("zzz", "validation")])
        with self.assertRaises(ValueError):
            load_split_data(split, sha, [self.raw])


if __name__ == "__main__":
    unittest.main()
