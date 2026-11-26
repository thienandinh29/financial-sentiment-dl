"""Create nested, ID-only E10 overlays on the frozen validation split."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from train.data import load_split, sha256_file


def build_manifests(split, clusters, pairs, targets, seed=1):
    """Select seeded donors and return exact exposure partitions for each target."""
    validation = list(split["validation"])
    valid = set(validation)
    if len(valid) != len(validation) or not valid or set(clusters) != set(sum((list(split[k]) for k in ("train", "validation", "test")), [])):
        raise ValueError("Invalid split or cluster coverage")
    counts = Counter(clusters[i] for i in validation)
    neighbours = {i: set() for i in validation}
    for a, b in pairs:
        if a in valid and b in valid:
            neighbours[a].add(b)
            neighbours[b].add(a)
        elif (a in valid and b in split["train"]) or (b in valid and a in split["train"]):
            raise ValueError("Validation already exposed to original train")
    order = sorted(validation)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: counts[clusters[i]] < 2)
    donors, exposed, output = set(), set(), []
    for target in targets:
        if not isinstance(target, int) or target <= len(exposed) or target > len(valid):
            raise ValueError("Targets must increase and fit validation")
        while len(exposed) < target:
            remaining = target - len(exposed)
            candidate = next((i for i in order if i not in donors and 0 < len(({i} | neighbours[i]) - exposed) <= remaining), None)
            if candidate is None:
                raise ValueError(f"Cannot reach target {target}; reached {len(exposed)}")
            donors.add(candidate)
            exposed |= {candidate} | neighbours[candidate]
        exact = donors
        near = exposed - exact
        output.append({"seed": seed, "train_add_validation_ids": sorted(donors),
                       "validation_evaluation_ids": validation,
                       "validation_exact_ids": sorted(exact),
                       "validation_near_only_ids": sorted(near),
                       "validation_exposed_ids": sorted(exposed),
                       "validation_clean_ids": sorted(valid - exposed),
                       "n_exposed": len(exposed)})
    return output


def main():
    """Verify frozen input hashes and write three new deterministic manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split", "split-sha256", "pairs", "pairs-sha256", "clusters", "clusters-sha256", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    split = load_split(args.split, args.split_sha256)
    for path, digest in ((args.pairs, args.pairs_sha256), (args.clusters, args.clusters_sha256)):
        if sha256_file(path) != digest:
            raise ValueError(f"SHA-256 mismatch: {path}")
    with open(args.clusters, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    clusters = {r["id"]: r["cluster_id"] for r in rows}
    if len(clusters) != len(rows):
        raise ValueError("Duplicate cluster ID")
    with open(args.pairs, encoding="utf-8-sig", newline="") as f:
        pairs = [(r["id_a"], r["id_b"]) for r in csv.DictReader(f)
                 if r["cosine"] and float(r["cosine"]) >= .92]
    targets = [(len(split["validation"]) * p + 99) // 100 for p in (5, 10, 20)]
    manifests = build_manifests(split, clusters, pairs, targets, args.seed)
    out = Path(args.output_dir)
    paths = [out / f"leakage_{p}.json" for p in (5, 10, 20)]
    if any(p.exists() for p in paths):
        raise FileExistsError("Refusing to overwrite manifests")
    out.mkdir(parents=True, exist_ok=True)
    for path, manifest in zip(paths, manifests):
        manifest["input_sha256"] = {"split": args.split_sha256, "pairs": args.pairs_sha256, "clusters": args.clusters_sha256}
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in paths), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
