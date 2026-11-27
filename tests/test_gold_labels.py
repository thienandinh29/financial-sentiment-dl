"""Contract checks for human gold label files in labeling/gold/ (named <annotator>_round<k>.csv)."""
import csv
import re
import tempfile
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


def double_assigned_ids(path=ASSIGNMENT):
    """Return the original 100 double-assigned IDs in assignment order, without labels."""
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts = {}
    for row in rows:
        counts.setdefault(row["id"], []).append(row["annotator"])
    ids = [identifier for identifier, annotators in counts.items()
           if len(annotators) == 2 and len(set(annotators)) == 2]
    if len(ids) != 100:
        raise ValueError("Expected exactly 100 double-assigned IDs")
    return ids


class GoldLabelTest(unittest.TestCase):
    def test_gold_files_match_assignment(self):
        files = sorted(GOLD_DIR.glob("*.csv"))
        self.assertTrue(files, "No gold files found")
        for path in files:
            with self.subTest(file=path.name):
                m = re.fullmatch(r"(tv[123])_round([12])\.csv", path.name)
                self.assertTrue(m or path.name == "tv1_double100.csv", f"Unexpected gold file name {path.name}")
                fields, rows = read_gold(path)
                self.assertEqual(fields, ["id", "label", "reason"])
                expected = double_assigned_ids() if path.name == "tv1_double100.csv" else assigned_ids(m[1], m[2])
                self.assertEqual([r["id"] for r in rows], expected)
                for r in rows:
                    self.assertIn(r["label"], LABELS)
                    self.assertTrue(r["reason"].strip(), f"Empty reason for {r['id']}")

    def test_double_assigned_ids_contract(self):
        ids = double_assigned_ids()
        self.assertEqual(len(ids), 100)
        self.assertEqual(len(set(ids)), 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignment.csv"
            path.write_text("id,annotator,round\na,tv1,1\na,tv2,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 100"):
                double_assigned_ids(path)


if __name__ == "__main__":
    unittest.main()
