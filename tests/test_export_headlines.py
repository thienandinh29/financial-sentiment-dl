"""Tests for scripts/export_headlines.py on a small fake lakehouse."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_headlines import export_headlines, junk_reason, read_headlines  # noqa: E402

ROWS = [
    # version_id, article_id, source, title, source_time, observed_at, available_at
    ("r1v1", "r1", "RSS", "Apple beats estimates", "2026-03-01 10:00:00+00", "2026-03-01 10:01:00+00", "2026-03-01 10:02:00+00"),
    ("r1v2", "r1", "RSS", "  Apple   beats estimates again ", "2026-03-01 10:00:00+00", "2026-03-01 11:01:00+00", "2026-03-01 11:02:00+00"),
    ("r2", "r2", "RSS", "Oil falls", "2026-01-01 00:00:00+00", "2026-01-01 00:01:00+00", "2026-01-01 00:02:00+00"),
    ("old", "old", "RSS", "Old news", "2025-12-31 23:59:59+00", "2026-01-01 00:00:00+00", "2026-01-01 00:01:00+00"),
    # 2026-01-01 06:00 in Bangkok is still 2025 in UTC
    ("tz", "tz", "RSS", "Timezone edge", "2026-01-01 06:00:00+07", "2026-01-01 00:00:00+00", "2026-01-01 00:01:00+00"),
    ("blank", "blank", "RSS", "   ", "2026-05-01 00:00:00+00", "2026-05-01 00:00:00+00", "2026-05-01 00:01:00+00"),
    ("g1", "g1", "GDELT", "Acme widget maker lifts yearly profit outlook after strong orders", None, "2026-04-01 00:00:00+00", "2026-04-01 00:01:00+00"),
    ("g2", "g2", "GDELT", "Bike lanes", None, "2026-04-02 00:00:00+00", "2026-04-02 00:01:00+00"),
    ("g3", "g3", "GDELT", "48151623.story behind acme consulting group partners", None, "2026-04-03 00:00:00+00", "2026-04-03 00:01:00+00"),
    ("s1", "s1", "SEC", "Form 8-K filed by some company today", None, "2026-04-04 00:00:00+00", "2026-04-04 00:01:00+00"),
]


def make_lakehouse(path):
    """Create a DuckDB file with the research_observations columns the export reads."""
    conn = duckdb.connect(str(path))
    conn.execute("""CREATE TABLE research_observations (
        version_id VARCHAR, article_id VARCHAR, source VARCHAR, title VARCHAR, url VARCHAR,
        source_time TIMESTAMPTZ, observed_at TIMESTAMPTZ, available_at TIMESTAMPTZ,
        source_time_kind VARCHAR, title_provenance VARCHAR, nlp_eligible BOOLEAN)""")
    conn.executemany(
        "INSERT INTO research_observations VALUES (?, ?, ?, ?, 'https://x/' || ?, ?, ?, ?, 'k', 'p', ? = 'RSS')",
        [(v, a, s, t, v, st, ob, av, s) for v, a, s, t, st, ob, av in ROWS])
    conn.close()


class JunkFilterTest(unittest.TestCase):
    def test_rejects_short_ids_and_symbols_keeps_headlines(self):
        self.assertIsNone(junk_reason("Acme widget maker lifts yearly profit outlook after strong orders"))
        self.assertEqual(junk_reason("Bike lanes"), "too_short")
        self.assertEqual(junk_reason("48151623.cms acme shares rise today"), "url_id")
        self.assertEqual(junk_reason("q3 - - 2026 $ % results"), "non_alpha")


class ExportHeadlinesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "lakehouse.duckdb"
        make_lakehouse(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_latest_rss_version_utc_2026_and_filtered_gdelt(self):
        rows, rejected = read_headlines(self.db, 2026)
        self.assertEqual([r["id"] for r in rows], ["r2", "r1v2", "g1"])
        self.assertEqual([r["pool"] for r in rows], ["test_candidate", "test_candidate", "train_unlabeled"])
        self.assertEqual(rows[1]["title"], "Apple beats estimates again")
        self.assertEqual(rows[0]["published_utc"], "2026-01-01T00:00:00Z")
        self.assertEqual(rows[2]["published_utc"], "2026-04-01T00:00:00Z")   # no source time: observed
        self.assertEqual(rejected, {"too_short": 1, "url_id": 1})

    def test_writes_jsonl_and_manifest_without_changing_the_db(self):
        before = self.db.read_bytes()
        out = Path(self.tmp.name) / "data" / "raw" / "headlines_2026" / "headlines.jsonl"
        manifest = export_headlines(self.db, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["source"] for line in lines], ["RSS", "RSS", "GDELT"])
        self.assertEqual(manifest["pools"], {"test_candidate": 2, "train_unlabeled": 1})
        saved = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["output_sha256"], manifest["output_sha256"])
        self.assertNotIn("Oil falls", json.dumps(saved))
        self.assertEqual(self.db.read_bytes(), before)

    def test_refuses_output_outside_raw(self):
        with self.assertRaises(ValueError):
            export_headlines(self.db, Path(self.tmp.name) / "results" / "headlines.jsonl")


if __name__ == "__main__":
    unittest.main()
