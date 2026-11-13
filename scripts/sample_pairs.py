"""Pick 50 lexical + 50 semantic-only near-duplicate pairs (seeded) for a manual check.

Writes results/feasibility/manual_check_100.csv with empty 'is_duplicate' and 'note' columns to
fill by hand, and prints both texts of each pair on screen only (texts are never saved).
"""
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.data import DEFAULT_RAW, load_raw  # noqa: E402

PAIRS = ROOT / "results/feasibility/fns/pairs.csv"
OUT = ROOT / "results/feasibility/manual_check_100.csv"


def main(seed=1, per_type=50):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    with open(PAIRS, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lexical = [r for r in rows if r["jaccard"]]
    semantic_only = [r for r in rows if not r["jaccard"]]
    rng = random.Random(seed)
    picked = ([dict(r, type="lexical") for r in rng.sample(lexical, per_type)]
              + [dict(r, type="semantic_only") for r in rng.sample(semantic_only, per_type)])
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: keep your filled-in answers, do not overwrite")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id_a", "id_b", "type", "jaccard", "cosine", "is_duplicate", "note"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows({**{k: r[k] for k in ("id_a", "id_b", "type", "jaccard", "cosine")},
                          "is_duplicate": "", "note": ""} for r in picked)
    text = {i: t for i, (t, _) in load_raw([ROOT / p for p in DEFAULT_RAW]).items()}
    for k, r in enumerate(picked, 1):
        print(f"\n[{k}] {r['type']}  jaccard={r['jaccard'] or '-'}  cosine={r['cosine'] or '-'}")
        print("  A:", text[r["id_a"]])
        print("  B:", text[r["id_b"]])


if __name__ == "__main__":
    main()
