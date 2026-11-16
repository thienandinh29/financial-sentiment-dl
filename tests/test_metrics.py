"""Tests for evaluate/metrics.py with a hand-computed example."""
import json
import tempfile
import unittest
from pathlib import Path

from evaluate.metrics import read_preds, score, validation_gold
from train.data import write_split

B, N, U = [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]   # predicts bearish / neutral / bullish


class MetricsTest(unittest.TestCase):
    def test_hand_computed_example(self):
        # gold:  a=bearish b=bearish c=neutral d=bullish
        # pred:  a=bearish b=neutral c=neutral d=bullish
        # bearish F1 = 2*1*0.5/1.5 = 2/3; neutral P=0.5 R=1 -> 2/3; bullish = 1
        result = score({"a": B, "b": N, "c": N, "d": U}, {"a": 0, "b": 0, "c": 1, "d": 2})
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["f1"], {"bearish": 0.6667, "neutral": 0.6667, "bullish": 1.0})
        self.assertEqual(result["macro_f1"], round((2 / 3 + 2 / 3 + 1) / 3, 4))
        self.assertEqual(result["macro_f1"], 0.7778)
        self.assertEqual(result["confusion"], [[1, 1, 0], [0, 1, 0], [0, 0, 1]])

    def test_ids_must_match(self):
        with self.assertRaises(ValueError):
            score({"a": B}, {"a": 0, "b": 1})
        with self.assertRaises(ValueError):
            score({"a": B, "b": N}, {"a": 0})

    def test_read_preds_rejects_bad_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            bad_cases = (
                [{"id": "a", "probs": [0.5, 0.5]}],
                [{"id": "a", "probs": [0.9, 0.9, 0.9]}],
                [{"id": "a", "probs": [-0.1, 0.6, 0.5]}],
                [{"id": "a", "probs": B}, {"id": "a", "probs": N}],
            )
            for rows in bad_cases:
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_preds(path)

    def test_read_preds_and_validation_gold_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "raw.jsonl"
            raw_rows = [
                {"id": "id-1", "text": "falls", "label": "bearish"},
                {"id": "id-2", "text": "flat", "label": "neutral"},
                {"id": "id-3", "text": "rises", "label": "bullish"},
            ]
            raw_path.write_text("".join(json.dumps(r) + "\n" for r in raw_rows), encoding="utf-8")
            split_path = root / "split.jsonl"
            sha = write_split(split_path, [("id-1", "train"), ("id-2", "validation"), ("id-3", "validation")])
            cfg = {"data": {"split_file": "split.jsonl", "split_sha256": sha, "raw_files": ["raw.jsonl"]}}
            gold = validation_gold(cfg, root=root)
            self.assertEqual(gold, {"id-2": 1, "id-3": 2})

            preds_path = root / "preds_validation.jsonl"
            preds_rows = [{"id": "id-2", "probs": N}, {"id": "id-3", "probs": U}]
            preds_path.write_text("".join(json.dumps(r) + "\n" for r in preds_rows), encoding="utf-8")
            scored = score(read_preds(preds_path), gold)
            self.assertEqual(scored["n"], 2)
            self.assertEqual(scored["f1"]["neutral"], 1.0)
            self.assertEqual(scored["f1"]["bullish"], 1.0)


if __name__ == "__main__":
    unittest.main()
