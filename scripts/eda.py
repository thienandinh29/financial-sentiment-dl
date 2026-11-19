"""Describe the full, unsplit FNS corpus; PhraseBank gets counts only."""

import argparse
from collections import Counter
import importlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metadata import run_metadata, sha256

LABELS = ("bearish", "bullish", "neutral")
FNS_FILES = (ROOT / "data/raw/fns/train.jsonl", ROOT / "data/raw/fns/validation.jsonl")


def read_rows(paths):
    """Read local JSONL files; reject duplicate IDs, blank text or unknown labels."""
    rows, seen = [], set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in seen or row["label"] not in LABELS or not row["text"].strip():
                    raise ValueError("Duplicate ID, unknown label or blank text")
                seen.add(row["id"])
                rows.append(row)
    return rows


def describe(rows, tokenize):
    """Return FNS counts and length/marker summaries for a supplied tokenizer."""
    import numpy as np

    lengths = [len(tokenize(row["text"])) for row in rows]
    if not lengths or min(lengths) < 1:
        raise ValueError("EDA requires nonempty token sequences")
    counts = Counter(row["label"] for row in rows)
    total = len(rows)
    summary = {
        "count": total,
        "classes": {label: {"count": counts[label], "fraction": counts[label] / total}
                    for label in LABELS},
        "minority_class": min(LABELS, key=lambda label: counts[label]),
        "length": {"unit": "tokens before padding/truncation",
                   "median": float(np.median(lengths)),
                   "p95": float(np.percentile(lengths, 95)),
                   "p99": float(np.percentile(lengths, 99)), "max": max(lengths),
                   "fraction_at_most_64": sum(n <= 64 for n in lengths) / total},
        "raw_markers": {
            "definition": "Regex on original FNS; cashtags only, not bare ticker whitelist",
            "cashtag_pattern": r"\$[A-Za-z][A-Za-z0-9]*(?:[.-][A-Za-z]+)?\b",
            "url_pattern": r"https?://\S+|www\.\S+",
        },
    }
    for marker, pattern in (("cashtag", summary["raw_markers"]["cashtag_pattern"]),
                            ("url", summary["raw_markers"]["url_pattern"])):
        count = sum(bool(re.search(pattern, row["text"], re.IGNORECASE)) for row in rows)
        summary["raw_markers"][marker] = {"count": count, "fraction": count / total}
    return summary, lengths


def save_figures(summary, lengths, output):
    """Save two text-free aggregate PNG figures to the given output directory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    counts = [summary["classes"][label]["count"] for label in LABELS]
    bars = axis.bar(LABELS, counts, color=("#bf4b45", "#29856c", "#60758d"))
    axis.bar_label(bars)
    axis.set(title="Twitter FNS: class distribution (all source rows)", ylabel="Rows")
    axis.set_ylim(0, max(counts) * 1.15)
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    axis.hist(lengths, bins=range(0, max(max(lengths), 64) + 3, 2), color="#60758d")
    axis.axvline(64, color="#bf4b45", linestyle="--", label="max_len = 64")
    axis.set(title="Twitter FNS: " + summary["tokenizer"] + " lengths",
             xlabel="Tokens before truncation", ylabel="Rows")
    axis.legend()
    fig.savefig(output / "length_distribution.png", dpi=160)
    plt.close(fig)


def main():
    """Run descriptive EDA with an explicit tokenizer; never fit or evaluate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", choices=("team", "whitespace"), default="team")
    parser.add_argument("--tokenizer-path", type=Path,
                        help="Use TV2 tokenizer from a separate checkout; record its SHA-256")
    args = parser.parse_args()
    start = time.perf_counter()
    if args.tokenizer == "team":
        try:
            if args.tokenizer_path:
                spec = importlib.util.spec_from_file_location("team_tokenizer", args.tokenizer_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                tokenize = module.tokenize
            else:
                tokenize = importlib.import_module("models.tokenizer").tokenize
        except ModuleNotFoundError as error:
            parser.error(f"TV2 tokenizer is unavailable ({error}). Explicitly use "
                         "--tokenizer whitespace for provisional statistics only.")
    else:
        tokenize = str.split
    rows = read_rows(FNS_FILES)
    summary, lengths = describe(rows, tokenize)
    summary["tokenizer"] = args.tokenizer
    summary["max_len_status"] = (
        "pending TV2 tokenizer; whitespace lengths cannot validate model max_len"
        if args.tokenizer == "whitespace" else "measured with team tokenizer; no config changed"
    )
    if args.tokenizer == "team":
        tokens = [tokenize(row["text"]) for row in rows]
        summary["normalized_markers"] = {
            marker: {"count": sum(marker in seq for seq in tokens),
                     "fraction": sum(marker in seq for seq in tokens) / len(rows)}
            for marker in ("<ticker>", "<url>")
        }
    inputs = list(FNS_FILES)
    phrasebank = ROOT / "data/raw/phrasebank/allagree.jsonl"
    if phrasebank.exists():
        # Counts only; no PhraseBank text/length/marker statistics or model calls.
        counts = Counter(row["label"] for row in read_rows([phrasebank]))
        summary["phrasebank_counts_only"] = {"count": sum(counts.values()),
                                              "classes": dict(sorted(counts.items()))}
        inputs.append(phrasebank)
    if args.tokenizer == "team":
        inputs.append((args.tokenizer_path or ROOT / "models/tokenizer.py").resolve())
    summary["provenance"] = run_metadata(start, {
        "tokenizer": args.tokenizer, "max_len_reference": 64,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs},
        "script_sha256": sha256(Path(__file__)),
    }, ["numpy", "matplotlib"])
    output = ROOT / "results/eda" / args.tokenizer
    output.mkdir(parents=True, exist_ok=True)
    report = output / "summary.json"
    if report.exists():
        parser.error(f"Report already exists: {report}; preserve it before rerunning")
    save_figures(summary, lengths, output)
    summary["provenance"]["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
