"""Synthetic fixtures only: no project test set or network is accessed."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from data.download import fns_rows, phrasebank_rows, write_jsonl
from scripts.eda import describe, read_rows
from scripts.run_metadata import sha256
from scripts.feasibility import leakage_bound


class DataPreparationTests(unittest.TestCase):
    def test_leakage_bound_counts_extra_cluster_members(self):
        bound = leakage_bound(["a", "a", "b", "c", "c", "c"], 0.5)
        self.assertEqual(bound["max_leakable_items_loose"], 3)
        self.assertEqual(bound["approximate_test_size"], 3)
        self.assertEqual(bound["loose_upper_fraction"], 1)

    def test_fns_mapping_ids_and_deterministic_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text('text,label\n"Café, falls",0\nRises,1\nMeeting,2\n',
                              encoding="utf-8")
            rows = list(fns_rows(source, "train"))
            self.assertEqual([r["label"] for r in rows], ["bearish", "bullish", "neutral"])
            self.assertEqual(rows[0]["id"], "fns-train-00000")
            first, second = root / "first.jsonl", root / "second.jsonl"
            write_jsonl(rows, first)
            write_jsonl(rows, second)
            self.assertEqual(sha256(first), sha256(second))
            self.assertEqual(read_rows([first]), rows)
            self.assertNotIn(b"\r\n", first.read_bytes())
            with self.assertRaises(ValueError):
                read_rows([first, second])

    def test_phrasebank_reads_allagree_and_splits_last_at(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pb.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
                                 "Café @ company@positive\nDrop@negative\nInfo@neutral\n".encode("latin-1"))
                archive.writestr("Sentences_50Agree.txt", "Must not read@negative")
            rows = list(phrasebank_rows(path))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], {"id": "pb-00000", "text": "Café @ company", "label": "bullish"})
            self.assertEqual(rows[1]["label"], "bearish")

    def test_unknown_fns_label_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("text,label\nExample,3\n", encoding="utf-8")
            with self.assertRaises(KeyError):
                list(fns_rows(path, "train"))

    def test_eda_lengths_and_markers(self):
        rows = [{"id": "a", "text": "$AAPL rises https://example.org", "label": "bullish"},
                {"id": "b", "text": "AI IT US", "label": "neutral"},
                {"id": "c", "text": "www.example.org falls", "label": "bearish"}]
        summary, lengths = describe(rows, str.split)
        self.assertEqual(lengths, [3, 3, 2])
        self.assertEqual(summary["raw_markers"]["cashtag"]["count"], 1)
        self.assertEqual(summary["raw_markers"]["url"]["count"], 2)
        self.assertEqual(summary["length"]["fraction_at_most_64"], 1)
        self.assertEqual(sum(v["count"] for v in summary["classes"].values()), 3)


if __name__ == "__main__":
    unittest.main()
