"""Tests for scripts/review_pairs.py: row specs, resumable review, merge checks, FNS file writing."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_pairs as rp  # noqa: E402

SHEET = [{"id_a": f"a{i}", "id_b": f"b{i}", "type": "lexical" if i <= 3 else "semantic_only",
          "jaccard": "0.9" if i <= 3 else "", "cosine": "0.95", "is_duplicate": "", "note": ""}
         for i in range(1, 7)]
TEXTS = {f"{p}{i}": f"secret text {p}{i}" for i in range(1, 7) for p in "ab"}


def scripted(answers):
    """Return an ask() that gives the scripted answers in order and fails if asked for more."""
    it = iter(answers)
    return lambda prompt: next(it)


def part(rows, answer="yes", note=""):
    """Build a {row: part row} dict for the given sheet rows."""
    return {r: {"row": r, "id_a": SHEET[r - 1]["id_a"], "id_b": SHEET[r - 1]["id_b"],
                "type": SHEET[r - 1]["type"], "is_duplicate": answer, "note": note} for r in rows}


class RowSpecTest(unittest.TestCase):
    def test_parse_and_format_rows(self):
        self.assertEqual(rp.parse_rows("1-3,5", 6), [1, 2, 3, 5])
        self.assertEqual(rp.parse_rows("1-3,2-4", 6), [1, 2, 3, 4])
        self.assertEqual(rp.ranges([1, 2, 3, 5, 7, 8]), "1-3,5,7-8")
        self.assertEqual(rp.parse_rows(rp.ranges([2, 3, 6]), 6), [2, 3, 6])

    def test_bad_specs_raise(self):
        for spec in ("", "0", "7", "3-1", "a", "1-", "1,,2"):
            with self.assertRaises(ValueError, msg=spec):
                rp.parse_rows(spec, 6)


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "parts" / "me.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_saves_each_answer_resumes_and_never_writes_text(self):
        shown = []
        # row 1: y + note; row 2: invalid "x" is asked again, then n and no note; row 3: quit
        done = rp.review(SHEET, TEXTS, [1, 2, 3], self.out, ask=scripted(["y", "same news", "x", "n", "", "q"]),
                         show=shown.append)
        self.assertEqual(done, 2)
        self.assertTrue(any("secret text a1" in s for s in shown))          # texts are shown on screen
        saved = rp.read_answers(self.out)
        self.assertEqual((saved[1]["is_duplicate"], saved[1]["note"]), ("yes", "same news"))
        self.assertEqual(saved[2]["is_duplicate"], "no")
        self.assertNotIn("secret text", self.out.read_text(encoding="utf-8"))   # ...but never saved
        # resume: rows 1 and 2 are not asked again; skip row 3 (nothing saved), then answer it
        self.assertEqual(rp.review(SHEET, TEXTS, [1, 2, 3], self.out, ask=scripted(["s"]), show=lambda s: None), 0)
        self.assertNotIn(3, rp.read_answers(self.out))
        self.assertEqual(rp.review(SHEET, TEXTS, [1, 2, 3], self.out, ask=scripted(["no", ""]), show=lambda s: None), 1)
        self.assertEqual(rp.read_answers(self.out)[3]["is_duplicate"], "no")

    def test_end_of_input_keeps_progress(self):
        def ask(prompt):
            if ask.calls >= 2:
                raise EOFError
            ask.calls += 1
            return ["y", ""][ask.calls - 1]
        ask.calls = 0
        self.assertEqual(rp.review(SHEET, TEXTS, [1, 2], self.out, ask=ask, show=lambda s: None), 1)
        self.assertIn(1, rp.read_answers(self.out))

    def test_refuses_wrong_sheet_and_missing_text(self):
        rp.write_answers(self.out, {1: {**part([1])[1], "id_a": "other"}})
        with self.assertRaises(ValueError):
            rp.review(SHEET, TEXTS, [1], self.out, ask=scripted([]), show=lambda s: None)
        with self.assertRaises(ValueError):
            rp.review(SHEET, {}, [2], Path(self.tmp.name) / "new.csv", ask=scripted([]), show=lambda s: None)


class MergeTest(unittest.TestCase):
    def test_fills_rows_reports_missing_and_summarizes(self):
        merged, unanswered = rp.merge(SHEET, [("a", part([1, 2], "yes", "n1")), ("b", part([4], "no"))])
        self.assertEqual([r["is_duplicate"] for r in merged], ["yes", "yes", "", "no", "", ""])
        self.assertEqual(merged[0]["note"], "n1")
        self.assertEqual(unanswered, [3, 5, 6])
        self.assertEqual(SHEET[0]["is_duplicate"], "")                      # the input sheet is not changed
        stats = rp.summarize(merged)
        self.assertEqual(stats["answered"], 3)
        self.assertEqual(stats["by_type"], {"lexical": {"yes": 2, "no": 0}, "semantic_only": {"yes": 0, "no": 1}})

    def test_rejects_overlap_mismatch_bad_answer_and_double_fill(self):
        with self.assertRaises(ValueError):
            rp.merge(SHEET, [("a", part([1, 2])), ("b", part([2, 3]))])      # row 2 answered twice
        with self.assertRaises(ValueError):
            rp.merge(SHEET, [("a", {1: {**part([1])[1], "id_b": "zzz"}})])   # different pair than the sheet
        with self.assertRaises(ValueError):
            rp.merge(SHEET, [("a", part([1], "maybe"))])                     # not yes/no
        with self.assertRaises(ValueError):
            rp.merge(SHEET, [("a", {7: {**part([1])[1], "row": 7}})])        # outside the sheet
        filled = [dict(r) for r in SHEET]
        filled[0]["is_duplicate"] = "no"
        with self.assertRaises(ValueError):
            rp.merge(filled, [("a", part([1]))])                             # sheet already has an answer


class FnsFileTest(unittest.TestCase):
    def test_writes_download_format_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            rp.write_fns_raw({"train": [("Café, falls", 0), ("Rises", 1)], "validation": [("Meeting", 2)]}, tmp)
            lines = (Path(tmp) / "train.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0]), {"id": "fns-train-00000", "text": "Café, falls", "label": "bearish"})
            self.assertEqual(json.loads(lines[1])["label"], "bullish")
            self.assertEqual(json.loads((Path(tmp) / "validation.jsonl").read_text(encoding="utf-8"))["id"],
                             "fns-validation-00000")
            good = {s: rp.sha256_file(Path(tmp) / f"{s}.jsonl") for s in ("train", "validation")}
            rp.verify_raw(tmp, good)
            with self.assertRaises(ValueError):
                rp.verify_raw(tmp, {**good, "train": "0" * 64})


if __name__ == "__main__":
    unittest.main()
