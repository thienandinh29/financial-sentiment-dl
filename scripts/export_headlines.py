"""Export 2026 headlines from a copy of the Paper 1 lakehouse into data/raw/headlines_2026/.

Reads research_observations read-only and never writes to the Paper 1 project. Run it on a
copy of lakehouse.duckdb, not on the live Docker volume. The output holds publisher text, so
it must stay under data/raw/ (gitignored) and is never committed.

Two pools (see context/decisions.md, 2026-10-26):
  - test_candidate: RSS, latest version per article (real feed headlines);
  - train_unlabeled: GDELT titles rebuilt from URL slugs, after a junk filter; training only.

Usage:
    python scripts/export_headlines.py --db path/to/lakehouse-copy.duckdb
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DEFAULT_OUT = Path("data/raw/headlines_2026/headlines.jsonl")
POOLS = {"RSS": "test_candidate", "GDELT": "train_unlabeled"}
MIN_WORDS = 5
MIN_ALPHA_SHARE = 0.6
ID_TOKEN = re.compile(r"\d{5,}|\.(cms|html?|story|php|aspx?)$", re.IGNORECASE)

# Latest version per article: the last one to become available.
QUERY = """
SELECT version_id, article_id, source, title, url, coalesce(source_time, observed_at) AS published,
       source_time_kind, title_provenance, nlp_eligible
FROM research_observations
WHERE source IN ('RSS', 'GDELT')
  AND coalesce(source_time, observed_at) >= make_timestamptz(?, 1, 1, 0, 0, 0.0, 'UTC')
  AND coalesce(source_time, observed_at) <  make_timestamptz(? + 1, 1, 1, 0, 0, 0.0, 'UTC')
  AND trim(coalesce(title, '')) <> ''
QUALIFY row_number() OVER (PARTITION BY source, article_id
                           ORDER BY available_at DESC, observed_at DESC, version_id DESC) = 1
ORDER BY published, version_id
"""


def sha256_file(path):
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def junk_reason(title):
    """Return why a slug-derived title is junk, or None if it looks like a headline.

    Rejects titles with fewer than MIN_WORDS words, URL ids such as '134531861.cms', or
    fewer than MIN_ALPHA_SHARE purely alphabetic words.
    """
    words = title.split()
    if len(words) < MIN_WORDS:
        return "too_short"
    if any(ID_TOKEN.search(w) for w in words):
        return "url_id"
    if sum(w.isalpha() for w in words) / len(words) < MIN_ALPHA_SHARE:
        return "non_alpha"
    return None


def read_headlines(db_path, year=2026):
    """Read one year's headlines from a lakehouse file, read-only.

    Input: path to a DuckDB lakehouse, publication year (UTC; observed time when the source
    gave none). Output: (rows, rejected) where rows are dicts sorted by (published_utc, id)
    with a 'pool' field, and rejected counts GDELT junk by reason.
    """
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        conn.execute("SET TimeZone = 'UTC'")
        records = conn.execute(QUERY, [year, year]).fetchall()
    finally:
        conn.close()
    rows, rejected = [], Counter()
    for r in records:
        title = " ".join(r[3].split())
        if r[2] == "GDELT" and (reason := junk_reason(title)):
            rejected[reason] += 1
            continue
        rows.append({
            "id": r[0], "article_id": r[1], "source": r[2], "pool": POOLS[r[2]], "title": title,
            "url": r[4],
            "published_utc": r[5].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_time_kind": r[6], "title_provenance": r[7], "nlp_eligible": bool(r[8]),
        })
    return rows, dict(rejected)


def export_headlines(db_path, out_path=DEFAULT_OUT, year=2026):
    """Write headlines as JSONL plus a manifest next to it.

    Input: lakehouse path, output .jsonl path (must be inside a data/raw/ folder), year.
    Output: the manifest dict (counts and SHA-256 of input and output, no headline text).
    """
    out_path = Path(out_path)
    if "raw" not in out_path.resolve().parent.parts:
        raise ValueError("Output must be under data/raw/: headline text is never committed")
    rows, rejected = read_headlines(db_path, year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "year": year,
        "rows": len(rows),
        "pools": dict(sorted(Counter(r["pool"] for r in rows).items())),
        "gdelt_rejected": rejected,
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
