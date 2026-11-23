"""Unit tests for models/llm_soft_labels.py (task 3.11: P2 soft labels on train_unlabeled pool)."""
import json
import tempfile
import unittest
from pathlib import Path

from evaluate.metrics import read_preds
from models.llm_soft_labels import (
    DEFAULT_HEADLINES,
    EXPECTED_HEADLINES_SHA256,
    generate_soft_labels,
    load_excluded_ids,
    load_p2_prompt_template,
    load_unlabeled_2026_pool,
    validate_soft_label_row,
)


class LlmSoftLabelsTests(unittest.TestCase):
    def test_loads_winning_p2_prompt_and_real_trial30_exclusions(self):
        template, cfg = load_p2_prompt_template()
        self.assertIn("bearish =", template)
        self.assertIn("bullish =", template)
        self.assertIn("neutral =", template)
        self.assertEqual(cfg["model"]["name"], "Qwen/Qwen2.5-7B-Instruct")

        excluded = load_excluded_ids()
        # 30 trial IDs from labeling/trial30_ids.csv + 300 disjoint test IDs from labeling/assignment.csv
        self.assertEqual(len(excluded), 330)

        if DEFAULT_HEADLINES.is_file():
            eligible, stats = load_unlabeled_2026_pool(
                DEFAULT_HEADLINES,
                excluded_ids=excluded,
                expected_sha256=EXPECTED_HEADLINES_SHA256,
            )
            self.assertEqual(stats["total_rows"], 2168)
            self.assertEqual(stats["pool_counts"], {"test_candidate": 759, "train_unlabeled": 1409})
            self.assertEqual(stats["excluded_overlap_train_unlabeled"], 0)
            self.assertEqual(stats["excluded_overlap_test_candidate"], 330)
            self.assertEqual(len(eligible), 1409)
            self.assertTrue(excluded.isdisjoint({hid for hid, _ in eligible}))

    def test_pool_filter_excludes_test_candidate_and_trial_or_test_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial_csv = root / "trial30_ids.csv"
            test_csv = root / "test300_ids.csv"
            trial_csv.write_text("id,url\ntrial-id-1,https://example.com/1\n", encoding="utf-8")
            test_csv.write_text("id,url\ntest-id-1,https://example.com/2\n", encoding="utf-8")

            headlines_path = root / "headlines.jsonl"
            rows = [
                {"id": "u-1", "title": "Company reports strong quarterly earnings growth", "pool": "train_unlabeled"},
                {"id": "u-2", "title": "Central bank holds interest rates steady today", "pool": "train_unlabeled"},
                {"id": "rss-1", "title": "RSS test candidate headline must be skipped", "pool": "test_candidate"},
                {"id": "trial-id-1", "title": "Trial headline even if marked train_unlabeled", "pool": "train_unlabeled"},
                {"id": "test-id-1", "title": "Test headline even if marked train_unlabeled", "pool": "train_unlabeled"},
            ]
            headlines_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

            excluded = load_excluded_ids([trial_csv, test_csv])
            self.assertEqual(excluded, {"trial-id-1", "test-id-1"})

            eligible, stats = load_unlabeled_2026_pool(headlines_path, excluded_ids=excluded)
            self.assertEqual([i for i, _ in eligible], ["u-1", "u-2"])
            self.assertEqual(stats["total_rows"], 5)
            self.assertEqual(stats["eligible_train_unlabeled"], 2)
            self.assertEqual(stats["skipped_non_train_pool"], 1)
            self.assertEqual(stats["skipped_excluded_trial_or_test_ids"], 2)

            with self.assertRaises(ValueError):
                load_unlabeled_2026_pool(
                    headlines_path,
                    excluded_ids=excluded,
                    expected_sha256="0" * 64,
                )

    def test_generate_soft_labels_writes_text_free_jsonl_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            headlines_path = root / "headlines.jsonl"
            rows = [
                {"id": "u-1", "title": "Secret headline one that must not appear in output", "pool": "train_unlabeled"},
                {"id": "u-2", "title": "Secret headline two that must not appear in output", "pool": "train_unlabeled"},
                {"id": "rss-skip", "title": "RSS candidate headline", "pool": "test_candidate"},
            ]
            headlines_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            out_dir = root / "soft_out"

            calls = []

            def fake_scorer(prompt):
                calls.append(prompt)
                return ({"bearish": 0.1, "neutral": 0.2, "bullish": 0.7}, 0.995, ["C", "B", "A"], [-2.3, -1.6, -0.35])

            # First run with limit=1
            summary1 = generate_soft_labels(
                headlines_path=headlines_path,
                out_dir=out_dir,
                exclusion_files=(),
                scorer_fn=fake_scorer,
                limit=1,
            )
            self.assertEqual(summary1["n_scored_total"], 1)
            self.assertEqual(summary1["n_newly_scored"], 1)
            self.assertEqual(len(calls), 1)

            # Second run without limit: should resume and only score u-2
            summary2 = generate_soft_labels(
                headlines_path=headlines_path,
                out_dir=out_dir,
                exclusion_files=(),
                scorer_fn=fake_scorer,
            )
            self.assertEqual(summary2["n_scored_total"], 2)
            self.assertEqual(summary2["n_newly_scored"], 1)
            self.assertEqual(len(calls), 2)

            out_jsonl = out_dir / "p2_soft_labels.jsonl"
            raw_output = out_jsonl.read_text(encoding="utf-8")
            self.assertNotIn("Secret headline", raw_output)
            preds = read_preds(out_jsonl)
            self.assertEqual(set(preds.keys()), {"u-1", "u-2"})

    def test_validate_soft_label_row_rejects_text_fields_and_bad_probs(self):
        with self.assertRaises(ValueError):
            validate_soft_label_row({"id": "u-1", "probs": [0.2, 0.3, 0.5], "title": "leaked text"})
        with self.assertRaises(ValueError):
            validate_soft_label_row({"id": "u-1", "probs": [0.5, 0.6, 0.1]})

    def test_integrated_p2_soft_labels_artifacts(self):
        out_dir = Path("results/llm_soft_labels")
        out_jsonl = out_dir / "p2_soft_labels.jsonl"
        out_run = out_dir / "run.json"
        self.assertTrue(out_jsonl.is_file(), f"Missing {out_jsonl}")
        self.assertTrue(out_run.is_file(), f"Missing {out_run}")

        run_meta = json.loads(out_run.read_text(encoding="utf-8"))
        self.assertEqual(run_meta["prompt"], "P2")
        self.assertEqual(run_meta["model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(run_meta["quantization"], "nf4-4bit")
        self.assertEqual(run_meta["label_order"], ["bearish", "neutral", "bullish"])
        self.assertEqual(run_meta["n_scored_total"], 1409)
        self.assertEqual(run_meta["n_newly_scored"], 1409)
        self.assertEqual(
            run_meta["pool_stats"]["headlines_sha256"],
            EXPECTED_HEADLINES_SHA256,
        )

        excluded = load_excluded_ids()
        self.assertEqual(len(excluded), 330)

        preds = read_preds(out_jsonl)
        self.assertEqual(len(preds), 1409)
        self.assertTrue(excluded.isdisjoint(preds.keys()))

        rows = [
            json.loads(line)
            for line in out_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1409)
        masses = []
        top1_flags = []
        for row in rows:
            validate_soft_label_row(row)
            self.assertEqual(
                set(row.keys()),
                {"id", "probs", "abc_mass", "top1_is_letter", "letter_logp"},
            )
            self.assertEqual(len(row["letter_logp"]), 3)
            masses.append(float(row["abc_mass"]))
            top1_flags.append(bool(row["top1_is_letter"]))

        self.assertEqual(round(min(masses), 4), run_meta["min_abc_mass"])
        self.assertEqual(round(sum(masses) / len(masses), 4), run_meta["mean_abc_mass"])
        self.assertEqual(round(sum(top1_flags) / len(top1_flags), 4), run_meta["share_top1_is_letter"])

        if DEFAULT_HEADLINES.is_file():
            eligible, _ = load_unlabeled_2026_pool(
                DEFAULT_HEADLINES,
                excluded_ids=excluded,
                expected_sha256=EXPECTED_HEADLINES_SHA256,
            )
            self.assertEqual([r["id"] for r in rows], [hid for hid, _ in eligible])

    def test_tv3_round1_llm_annotation_csv_contract(self):
        import csv

        round1_path = Path("labeling/llm_annotations/round1/tv3.csv")
        assign_path = Path("labeling/assignment.csv")
        self.assertTrue(round1_path.is_file(), f"Missing {round1_path}")
        self.assertTrue(assign_path.is_file(), f"Missing {assign_path}")

        with open(round1_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ["id", "label", "reason"])
            rows = list(reader)

        self.assertEqual(len(rows), 68)
        ids = []
        for r in rows:
            hid = (r.get("id") or "").strip()
            lbl = (r.get("label") or "").strip()
            reason = (r.get("reason") or "").strip()
            self.assertTrue(hid, "Empty ID in labeling/llm_annotations/round1/tv3.csv")
            self.assertIn(lbl, {"bearish", "neutral", "bullish"})
            self.assertTrue(reason, f"Empty reason for ID {hid}")
            ids.append(hid)

        self.assertEqual(len(set(ids)), 68)

        with open(assign_path, "r", encoding="utf-8-sig", newline="") as f:
            expected_ids = [
                r["id"].strip()
                for r in csv.DictReader(f)
                if r["annotator"].strip() == "tv3" and str(r["round"]).strip() == "1"
            ]
        self.assertEqual(ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
