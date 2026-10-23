"""Export 2026 headlines from a copy of the Paper 1 lakehouse into data/raw/headlines_2026/.

Reads the DuckDB file read-only and never writes to the Paper 1 project. Run it on a copy of
lakehouse.duckdb, not on the live Docker volume. The output holds publisher text, so it must
stay under data/raw/ (gitignored) and is never committed.

Usage:
    python scripts/export_headlines.py --db path/to/lakehouse-copy.duckdb
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DEFAULT_OUT = Path("data/raw/headlines_2026/headlines.jsonl")

QUERY = """
SELECT id, source, title, url, published_time, is_near_duplicate, canonical_cluster_id
FROM silver_financial_news
WHERE published_time >= make_timestamptz(?, 1, 1, 0, 0, 0.0, 'UTC')
  AND published_time <  make_timestamptz(? + 1, 1, 1, 0, 0, 0.0, 'UTC')
  AND trim(coalesce(title, '')) <> ''
ORDER BY published_time, id
"""


def sha256_file(path):
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_headlines(db_path, year=2026):
    """Read one year's headlines from a lakehouse file, read-only.

    Input: path to a DuckDB lakehouse, publication year (UTC).
    Output: list of dicts sorted by (published_utc, id); titles are whitespace-stripped.
    Near-duplicate flags are kept, not filtered: deduplication is done in this project.
    """
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        conn.execute("SET TimeZone = 'UTC'")
        rows = conn.execute(QUERY, [year, year]).fetchall()
    finally:
        conn.close()
    return [{
        "id": row[0],
        "source": row[1],
        "title": " ".join(row[2].split()),
        "url": row[3],
        "published_utc": row[4].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pipeline_near_duplicate": bool(row[5]),
        "pipeline_cluster_id": row[6],
    } for row in rows]


def export_headlines(db_path, out_path=DEFAULT_OUT, year=2026):
    """Write headlines as JSONL plus a manifest next to it.

    Input: lakehouse path, output .jsonl path (must be inside a data/raw/ folder), year.
    Output: the manifest dict (counts and SHA-256 of input and output, no headline text).
    """
    out_path = Path(out_path)
    if "raw" not in out_path.resolve().parent.parts:
        raise ValueError("Output must be under data/raw/: headline text is never committed")
    rows = read_headlines(db_path, year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "year": year,
        "rows": len(rows),
        "pipeline_near_duplicates": sum(r["pipeline_near_duplicate"] for r in rows),
        "sources": {s: sum(r["source"] == s for r in rows) for s in sorted({r["source"] for r in rows})},
        "first_published_utc": rows[0]["published_utc"] if rows else None,
        "last_published_utc": rows[-1]["published_utc"] if rows else None,
        "input_db_sha256": sha256_file(db_path),
        "output_sha256": sha256_file(out_path),
        "exported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, help="copy of the Paper 1 lakehouse.duckdb")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()
    manifest = export_headlines(args.db, args.out, args.year)
    print(json.dumps({k: v for k, v in manifest.items() if k != "exported_at_utc"}, indent=2))


if __name__ == "__main__":
    main()
