"""Contract checks for human gold label files in labeling/gold/ (named <annotator>_round<k>.csv)."""
import csv
import re
import unittest
from pathlib import Path

GOLD_DIR = Path("labeling/gold")
ASSIGNMENT = Path("labeling/assignment.csv")
LABELS = {"bearish", "neutral", "bullish"}


def read_gold(path):
    """Read one gold file; return (fieldnames, list of row dicts)."""
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def assigned_ids(annotator, round_):
    """Return the ordered IDs assigned to one annotator in one round."""
    with open(ASSIGNMENT, encoding="utf-8-sig", newline="") as stream:
        return [r["id"] for r in csv.DictReader(stream)
                if r["annotator"] == annotator and r["round"] == str(round_)]


class GoldLabelTest(unittest.TestCase):
    def test_gold_files_match_assignment(self):
        files = sorted(GOLD_DIR.glob("*.csv"))
        self.assertTrue(files, "No gold files found")
        for path in files:
            with self.subTest(file=path.name):
                m = re.fullmatch(r"(tv[123])_round([12])\.csv", path.name)
                self.assertTrue(m, f"Unexpected gold file name {path.name}")
                fields, rows = read_gold(path)
                self.assertEqual(fields, ["id", "label", "reason"])
                self.assertEqual([r["id"] for r in rows], assigned_ids(m[1], m[2]))
                for r in rows:
                    self.assertIn(r["label"], LABELS)
                    self.assertTrue(r["reason"].strip(), f"Empty reason for {r['id']}")


if __name__ == "__main__":
    unittest.main()
