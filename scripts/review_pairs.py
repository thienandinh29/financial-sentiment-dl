"""Review near-duplicate pairs by hand, split across people, then merge the answers.

The pair sheet (results/feasibility/manual_check_100.csv) holds ids and scores only. The tweets are read
from data/raw/fns/ on your own machine and are never written to any output file. Steps:

    python scripts/review_pairs.py fetch                           # once: download FNS text, check SHA-256
    python scripts/review_pairs.py review --name tv2 --rows 1-15,51-65 [--sheet PATH]
    python scripts/review_pairs.py merge --parts a.csv b.csv c.csv --out merged.csv [--sheet PATH]

`review` saves each answer at once, so you can quit (q) and continue later. Answer "y" when both texts
report the same news or event; the same topic alone is "n". Write notes in your own words.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "results/feasibility/manual_check_100.csv"
PARTS_DIR = ROOT / "results/feasibility/manual_check_parts"
RAW_DIR = ROOT / "data/raw/fns"
PART_COLUMNS = ["row", "id_a", "id_b", "type", "is_duplicate", "note"]
ANSWERS = ("yes", "no")
# SHA-256 of the FNS files TV1 sampled the pairs from (results/TV1_WEEK1.md): a mismatch means other texts.
EXPECTED_SHA256 = {
    "train": "52158609a11e71afd3dab4cd292773bd3397a9e7890c9354fb5b16cebf7acde0",
    "validation": "a8cb54b82b1a01c767cdc9b63c697483360fe43ab37cc90c68cb90169af4484a",
}
FNS_LABELS = {0: "bearish", 1: "bullish", 2: "neutral"}      # source order, as in data/download.py


def parse_rows(spec, total):
    """Turn '1-15,51-65' into a sorted list of 1-based row numbers, each within 1..total."""
    rows = set()
    for part in spec.split(","):
        lo, sep, hi = part.strip().partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()):
            raise ValueError(f"Bad row spec '{part.strip()}': use numbers like 1-15,51-65")
        lo, hi = int(lo), int(hi) if sep else int(lo)
        if not 1 <= lo <= hi <= total:
            raise ValueError(f"Rows must satisfy 1 <= start <= end <= {total}: '{part.strip()}'")
        rows.update(range(lo, hi + 1))
    return sorted(rows)


def ranges(numbers):
    """Format sorted numbers as '1-3,5' (the inverse of parse_rows)."""
    out, start, prev = [], None, None
    for n in list(numbers) + [None]:
        if start is None:
            start = prev = n
        elif n is not None and n == prev + 1:
            prev = n
        else:
            out.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = n
    return ",".join(out)


def read_sheet(path):
    """Read the pair sheet as a list of dicts (utf-8, with or without a BOM)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_answers(path):
    """Read a part file into {row number: row dict}; empty if the file does not exist yet."""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {int(r["row"]): r for r in csv.DictReader(f)}


def write_answers(path, answers):
    """Write a part file in row order, through a temporary file so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PART_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(answers[row] for row in sorted(answers))
    tmp.replace(path)


def load_texts(raw_dir):
    """Read data/raw/fns/{train,validation}.jsonl into {id: text}; say how to get them if missing."""
    files = [Path(raw_dir) / f"{name}.jsonl" for name in ("train", "validation")]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"FNS text not found ({', '.join(missing)}). "
                                "Run once: python scripts/review_pairs.py fetch")
    texts = {}
    for f in files:
        with open(f, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    texts[row["id"]] = row["text"]
    return texts


def review(sheet, texts, rows, out_path, ask=input, show=print):
    """Show each pair in `rows` and record yes/no answers, resuming from out_path.

    Input: sheet rows (list of dicts), {id: text}, 1-based row numbers, the part-file path,
    ask(prompt) -> str and show(text). Every answer is saved at once. 's' skips a pair, 'q' stops.
    Output: how many pairs were answered in this run. Texts are shown but never written to a file.
    """
    answers = read_answers(out_path)
    for r, a in answers.items():
        pair = sheet[r - 1] if 1 <= r <= len(sheet) else None
        if pair is None or (a["id_a"], a["id_b"], a["type"]) != (pair["id_a"], pair["id_b"], pair["type"]):
            raise ValueError(f"{out_path} does not match this sheet at row {r}: wrong sheet or old file?")
    todo = [r for r in rows if r not in answers]
    missing = sorted({i for r in todo for i in (sheet[r - 1]["id_a"], sheet[r - 1]["id_b"]) if i not in texts})
    if missing:
        raise ValueError(f"{len(missing)} pair ids have no text, e.g. {missing[:3]}: wrong FNS files?")
    show(f"{len(rows) - len(todo)} of your {len(rows)} pairs are already answered; {len(todo)} to go.")
    done = 0
    try:
        for n, r in enumerate(todo, 1):
            pair = sheet[r - 1]
            show(f"\n[{r}/{len(sheet)}]  {len(todo) - n + 1} left  {pair['type']}  "
                 f"jaccard={pair['jaccard'] or '-'}  cosine={pair['cosine'] or '-'}")
            show(f"  A: {texts[pair['id_a']]}")
            show(f"  B: {texts[pair['id_b']]}")
            while True:
                choice = ask("Same news or event?  y = yes, n = no, s = skip, q = quit: ").strip().lower()
                if choice in ("y", "yes", "n", "no", "s", "q"):
                    break
                show("Please type y, n, s or q.")
            if choice == "q":
                break
            if choice == "s":
                continue
            note = ask("Note (optional, your own words, do not copy the tweets): ").strip()
            answers[r] = {"row": r, "id_a": pair["id_a"], "id_b": pair["id_b"], "type": pair["type"],
                          "is_duplicate": "yes" if choice.startswith("y") else "no", "note": note}
            write_answers(out_path, answers)
            done += 1
    except EOFError:
        show("\nInput ended; progress is saved.")
    show(f"\nSaved {done} new answer(s) in {out_path}. {len([r for r in rows if r not in answers])} of your pairs still open.")
    return done


def merge(sheet, parts):
    """Fill is_duplicate and note in the sheet from several part files.

    Input: sheet rows and a list of (name, {row: part row}). Output: (merged rows, rows still unanswered).
    Raises if a part row does not match the sheet's pair, an answer is not yes/no, or a row is answered
    twice (in two parts, or in a part and already in the sheet).
    """
    merged = [dict(r) for r in sheet]
    seen = {}
    for name, answers in parts:
        for r, a in answers.items():
            if not 1 <= r <= len(sheet):
                raise ValueError(f"{name}: row {r} is outside the sheet (1-{len(sheet)})")
            pair = sheet[r - 1]
            if (a["id_a"], a["id_b"], a["type"]) != (pair["id_a"], pair["id_b"], pair["type"]):
                raise ValueError(f"{name}: row {r} is a different pair than in the sheet")
            answer = a["is_duplicate"].strip().lower()
            if answer not in ANSWERS:
                raise ValueError(f"{name}: row {r} needs yes or no, got '{a['is_duplicate']}'")
            if r in seen or pair["is_duplicate"].strip():
                raise ValueError(f"Row {r} is answered twice ({seen.get(r, 'sheet')} and {name})")
            seen[r] = name
            merged[r - 1]["is_duplicate"] = answer
            merged[r - 1]["note"] = a.get("note", "")
    unanswered = [i for i, r in enumerate(merged, 1) if r["is_duplicate"].strip().lower() not in ANSWERS]
    return merged, unanswered


def summarize(rows):
    """Count answers per pair type: {'answered', 'total', 'by_type': {type: {'yes': y, 'no': n}}}."""
    by_type = {}
    for r in rows:
        answer = r["is_duplicate"].strip().lower()
        if answer in ANSWERS:
            by_type.setdefault(r["type"], {"yes": 0, "no": 0})[answer] += 1
    return {"answered": sum(v["yes"] + v["no"] for v in by_type.values()), "total": len(rows), "by_type": by_type}


def sha256_file(path):
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fns_raw(rows_by_split, out_dir):
    """Write {split: [(text, label int), ...]} as JSONL with ids fns-<split>-00000, as data/download.py does."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for k, (text, label) in enumerate(rows):
                row = {"id": f"fns-{split}-{k:05d}", "text": text, "label": FNS_LABELS[label]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_raw(out_dir, expected=EXPECTED_SHA256):
    """Raise if a FNS file's SHA-256 differs from the one TV1 sampled from."""
    for split, sha in expected.items():
        actual = sha256_file(Path(out_dir) / f"{split}.jsonl")
        if actual != sha:
            raise ValueError(f"{split}.jsonl has SHA-256 {actual[:12]}..., expected {sha[:12]}...: "
                             "these are not the texts TV1 sampled from")


def fetch(out_dir=RAW_DIR):
    """Download the Twitter FNS text once, write it to data/raw/fns/ and check its SHA-256.

    On a mismatch the written files are deleted, so a wrong copy is never used.
    """
    from datasets import load_dataset
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    write_fns_raw({s: [(r["text"], r["label"]) for r in ds[s]] for s in ("train", "validation")}, out_dir)
    try:
        verify_raw(out_dir)
    except ValueError:
        for split in EXPECTED_SHA256:
            (Path(out_dir) / f"{split}.jsonl").unlink(missing_ok=True)
        raise


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the FNS text once and check its SHA-256")
    p = sub.add_parser("review", help="read pairs and answer yes/no")
    p.add_argument("--name", required=True, help="your short name; answers go to manual_check_parts/<name>.csv")
    p.add_argument("--rows", required=True, help="sheet rows to review, e.g. 1-15,51-65")
    p.add_argument("--sheet", default=str(SHEET))
    p.add_argument("--out", help="part file to write (default: results/feasibility/manual_check_parts/<name>.csv)")
    m = sub.add_parser("merge", help="combine part files into one filled sheet")
    m.add_argument("--parts", nargs="+", required=True)
    m.add_argument("--sheet", default=str(SHEET))
    m.add_argument("--out", required=True)
    m.add_argument("--force", action="store_true", help="overwrite --out even if it already holds answers")
    args = parser.parse_args()

    if args.command == "fetch":
        fetch()
        print(f"FNS text saved in {RAW_DIR} and its SHA-256 matches the one TV1 sampled from.")
    elif args.command == "review":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
            parser.error("--name may only contain letters, digits, - and _")
        sheet = read_sheet(args.sheet)
        out = Path(args.out) if args.out else PARTS_DIR / f"{args.name}.csv"
        review(sheet, load_texts(RAW_DIR), parse_rows(args.rows, len(sheet)), out)
    else:
        sheet = read_sheet(args.sheet)
        out = Path(args.out)
        if out.exists() and not args.force and any(r["is_duplicate"].strip() for r in read_sheet(out)):
            parser.error(f"{out} already holds answers; use --force to overwrite it")
        merged, unanswered = merge(sheet, [(Path(p).stem, read_answers(p)) for p in args.parts])
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sheet[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(merged)
        stats = summarize(merged)
        print(f"Wrote {out}: {stats['answered']} of {stats['total']} pairs answered.")
        for kind, c in sorted(stats["by_type"].items()):
            print(f"  {kind}: {c['yes']} yes, {c['no']} no")
        if unanswered:
            print(f"Still unanswered rows: {ranges(unanswered)}")


if __name__ == "__main__":
    main()
