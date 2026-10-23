"""Tests for scripts/export_headlines.py on a small fake lakehouse."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_headlines import export_headlines, read_headlines  # noqa: E402

ROWS = [
    # id, source, title, url, published_time, is_near_duplicate
    ("b", "RSS", "  Apple   beats estimates ", "https://x/b", "2026-03-01 10:00:00+00", False),
    ("a", "GDELT", "Oil falls", "https://x/a", "2026-01-01 00:00:00+00", True),
    ("old", "RSS", "Old news", "https://x/old", "2025-12-31 23:59:59+00", False),
    ("next", "RSS", "Next year", "https://x/next", "2027-01-01 00:00:00+00", False),
    ("blank", "RSS", "   ", "https://x/blank", "2026-05-01 00:00:00+00", False),
    # 2026-01-01 06:00 in Bangkok is still 2025 in UTC
    ("tz", "RSS", "Timezone edge", "https://x/tz", "2026-01-01 06:00:00+07", False),
]


def make_lakehouse(path):
    """Create a DuckDB file with the silver_financial_news columns the export reads."""
    conn = duckdb.connect(str(path))
    conn.execute("""CREATE TABLE silver_financial_news (
        id VARCHAR PRIMARY KEY, source VARCHAR, title VARCHAR, url VARCHAR,
        published_time TIMESTAMPTZ, is_near_duplicate BOOLEAN, canonical_cluster_id VARCHAR)""")
    conn.executemany("INSERT INTO silver_financial_news VALUES (?, ?, ?, ?, ?, ?, NULL)", ROWS)
    conn.close()


class ExportHeadlinesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "lakehouse.duckdb"
        make_lakehouse(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_keeps_only_utc_2026_with_text_sorted_by_time(self):
        rows = read_headlines(self.db, 2026)
        self.assertEqual([r["id"] for r in rows], ["a", "b"])
        self.assertEqual(rows[0]["published_utc"], "2026-01-01T00:00:00Z")
        self.assertEqual(rows[1]["title"], "Apple beats estimates")
        self.assertTrue(rows[0]["pipeline_near_duplicate"])

    def test_writes_jsonl_and_manifest_without_changing_the_db(self):
        before = self.db.read_bytes()
        out = Path(self.tmp.name) / "data" / "raw" / "headlines_2026" / "headlines.jsonl"
        manifest = export_headlines(self.db, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["id"] for line in lines], ["a", "b"])
        self.assertEqual(manifest["rows"], 2)
        self.assertEqual(manifest["sources"], {"GDELT": 1, "RSS": 1})
        saved = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["output_sha256"], manifest["output_sha256"])
        self.assertNotIn("Oil falls", json.dumps(saved))
        self.assertEqual(self.db.read_bytes(), before)

    def test_refuses_output_outside_raw(self):
        with self.assertRaises(ValueError):
            export_headlines(self.db, Path(self.tmp.name) / "results" / "headlines.jsonl")


if __name__ == "__main__":
    unittest.main()
