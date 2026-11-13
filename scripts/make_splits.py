"""Make the frozen FNS splits: cluster split (main) and random split (E9), 70/15/15, label-balanced.

Reads data/raw/fns/*.jsonl and the dedup clusters, writes ID-only split files, SHA256SUMS and a
text-free summary. Every near-duplicate cluster lands whole in one split (cluster split).
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.data import DEFAULT_RAW, LABELS, load_raw, write_split  # noqa: E402
from scripts.summarize_manual_check import summarize  # noqa: E402

RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def verify_manual_review(path, pairs_path):
    """Require the completed seed-1 sample from the current pair file."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reviewed = list(csv.DictReader(f))
    result = summarize(reviewed)
    with open(pairs_path, encoding="utf-8-sig", newline="") as f:
        pairs = list(csv.DictReader(f))
    rng = random.Random(1)
    lexical = [r for r in pairs if r["jaccard"]]
    semantic = [r for r in pairs if not r["jaccard"]]
    expected = {(r["id_a"], r["id_b"], kind)
                for kind, sample in (("lexical", rng.sample(lexical, 50)),
                                     ("semantic_only", rng.sample(semantic, 50)))
                for r in sample}
    actual = {(r["id_a"], r["id_b"], r["type"]) for r in reviewed}
    if actual != expected:
        raise ValueError("Manual review does not match the seed-1 sample from the current pair file")
    return result


def assign_groups(groups, labels_of, seed):
    """Put each group (list of ids) whole into one split, keeping each label near 70/15/15.

    Biggest groups first (ties in seeded random order); each goes to the split whose counts
    for the group's labels are furthest below target. Output: dict id -> split name.
    """
    rng = random.Random(seed)
    groups = [sorted(g) for g in groups]
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)                     # stable: equal sizes keep the shuffle
    total = Counter(labels_of[i] for g in groups for i in g)
    target = {s: {lab: r * total[lab] for lab in total} for s, r in RATIOS.items()}
    have = {s: Counter() for s in RATIOS}
    out = {}
    for group in groups:
        need = Counter(labels_of[i] for i in group)
        best = max(RATIOS, key=lambda s: sum((target[s][lab] - have[s][lab]) / target[s][lab] * n
                                             for lab, n in need.items()))
        have[best].update(need)
        out.update((i, best) for i in group)
    return out


def describe(assignment, labels_of):
    """Return per-split size, share and label shares (percent), plus the worst label drift."""
    total = Counter(labels_of.values())
    report, worst = {}, 0.0
    for split in RATIOS:
        ids = [i for i, s in assignment.items() if s == split]
        counts = Counter(labels_of[i] for i in ids)
        shares = {LABELS[lab]: round(100 * counts[lab] / len(ids), 2) for lab in sorted(total)}
        for lab in total:
            worst = max(worst, abs(100 * counts[lab] / len(ids) - 100 * total[lab] / len(labels_of)))
        report[split] = {"size": len(ids), "share": round(100 * len(ids) / len(labels_of), 2), "labels": shares}
    return report, worst


def crossing_pairs(pairs_csv, assignment):
    """Count near-duplicate pairs whose two ids sit in different splits, by split pair."""
    crossing = Counter()
    with open(pairs_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            a, b = assignment[row["id_a"]], assignment[row["id_b"]]
            if a != b:
                crossing["-".join(sorted((a, b)))] += 1
    return dict(sorted(crossing.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clusters", default="results/feasibility/fns/clusters.csv")
    parser.add_argument("--pairs", default="results/feasibility/fns/pairs.csv")
    parser.add_argument("--review", default="results/feasibility/manual_check_100.csv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="data/splits")
    args = parser.parse_args()
    verify_manual_review(ROOT / args.review, ROOT / args.pairs)
    raw = load_raw([ROOT / p for p in DEFAULT_RAW])
    labels_of = {i: label for i, (_, label) in raw.items()}
    clusters = defaultdict(list)
    with open(ROOT / args.clusters, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            clusters[row["cluster_id"]].append(row["id"])
    if sorted(i for g in clusters.values() for i in g) != sorted(labels_of):
        raise ValueError("Cluster file and raw files do not hold the same ids")
    out, summary, sums = ROOT / args.out, {"seed": args.seed}, []
    if any((out / name).exists() for name in ("cluster_split.jsonl", "random_split.jsonl", "SHA256SUMS")):
        raise FileExistsError("Split output already exists; refusing to overwrite frozen splits")
    assignments = {}
    for name, groups in (("cluster_split", list(clusters.values())), ("random_split", [[i] for i in labels_of])):
        assignment = assign_groups(groups, labels_of, args.seed)
        report, worst = describe(assignment, labels_of)
        if worst > 2:
            raise ValueError(f"{name}: label drift {worst} exceeds 2 percentage points")
        assignments[name] = assignment
        summary[name] = {"splits": report, "max_label_drift_points": worst,
                         "near_duplicate_pairs_across_splits": crossing_pairs(ROOT / args.pairs, assignment)}
    for name, assignment in assignments.items():
        sha = write_split(out / f"{name}.jsonl", sorted(assignment.items()))
        summary[name]["sha256"] = sha
        sums.append(f"{sha}  {name}.jsonl")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
