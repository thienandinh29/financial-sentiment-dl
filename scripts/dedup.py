"""Find near-duplicate texts and group them into clusters (RQ1 leakage, cluster split).

Two edge types, thresholds from SPEC.md:
  - lexical: exact Jaccard of word unigram+bigram shingles >= 0.75, candidates from MinHash LSH;
  - semantic: cosine of L2-normalized all-MiniLM-L6-v2 embeddings >= 0.85.
Clusters are connected components over both edge types. Outputs hold IDs and scores only,
never text, so they can be shared.

Usage:
    python scripts/dedup.py --input texts.jsonl --out results/feasibility/fns
    python scripts/dedup.py --input fns.csv --text-field text --no-semantic
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH

JACCARD_THRESHOLD = 0.75
COSINE_THRESHOLD = 0.85
NUM_PERM = 128
LSH_THRESHOLD = 0.5   # candidate net, set below JACCARD_THRESHOLD for recall; exact Jaccard decides
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 1


def shingles(text):
    """Return the set of lowercase word unigrams and bigrams; punctuation is dropped."""
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return set(tokens) | {f"{a} {b}" for a, b in zip(tokens, tokens[1:])}


def jaccard(left, right):
    """Exact Jaccard of two sets; two empty sets score 0 (no evidence of duplication)."""
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def lexical_pairs(texts, threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM, lsh_threshold=LSH_THRESHOLD):
    """Find pairs with exact shingle Jaccard >= threshold.

    Input: list of texts. Output: dict {(i, j): jaccard} with i < j (list indices).
    MinHash LSH only proposes candidates; every reported pair is verified exactly.
    """
    sets = [shingles(t) for t in texts]
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    pairs = {}
    for i, s in enumerate(sets):
        if not s:
            continue
        m = MinHash(num_perm=num_perm, seed=SEED)
        for token in s:
            m.update(token.encode("utf-8"))
        for j in lsh.query(m):
            score = jaccard(sets[j], s)
            if score >= threshold:
                pairs[(j, i)] = score
        lsh.insert(i, m)
    return pairs


def embed(texts, model_name=SEMANTIC_MODEL, batch_size=256):
    """Return L2-normalized float32 embeddings, shape (n, d), from a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def semantic_pairs(embeddings, threshold=COSINE_THRESHOLD, block=2048):
    """Find pairs with cosine >= threshold by exact blockwise search.

    Input: (n, d) array of L2-normalized rows. Output: dict {(i, j): cosine} with i < j.
    """
    pairs = {}
    n = len(embeddings)
    for start in range(0, n, block):
        sims = embeddings[start:start + block] @ embeddings.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if i < c:
                pairs[(i, c)] = float(sims[r, c])
    return pairs


def clusters(n, edges):
    """Group n items into connected components over the given (i, j) edges.

    Output: list of cluster ids, one per item; a cluster id is the smallest member index.
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(n)]


def read_records(path, id_field="id", text_field="text"):
    """Read (id, text) from .jsonl or .csv; a missing id column falls back to the row number."""
    path = Path(path)
    with open(path, encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f) if path.suffix == ".csv" else (json.loads(line) for line in f if line.strip())
        records = [(str(r.get(id_field, k)), r[text_field]) for k, r in enumerate(rows)]
    ids = [r[0] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate values in id field '{id_field}'")
    return ids, [r[1] for r in records]


def run(ids, texts, semantic=True, embeddings=None):
    """Compute pairs, clusters and a summary for one corpus.

    Input: ids and texts (same length); semantic toggles the MiniLM step; embeddings can be
    passed precomputed. Output: (pairs list, cluster id per item, summary dict).
    """
    lex = lexical_pairs(texts)
    sem = semantic_pairs(embed(texts) if embeddings is None else embeddings) if semantic else {}
    keys = sorted(set(lex) | set(sem))
    pairs = [{"id_a": ids[i], "id_b": ids[j],
              "jaccard": round(lex[(i, j)], 4) if (i, j) in lex else None,
              "cosine": round(sem[(i, j)], 4) if (i, j) in sem else None} for i, j in keys]
    cluster_of = clusters(len(texts), keys)
    sizes = Counter(cluster_of)
    in_dup = sum(1 for c in cluster_of if sizes[c] > 1)
    summary = {
        "items": len(texts),
        "pairs_lexical": len(lex),
        "pairs_semantic": len(sem) if semantic else None,
        "pairs_total": len(keys),
        "clusters": len(sizes),
        "clusters_size_gt1": sum(1 for s in sizes.values() if s > 1),
        "largest_cluster": max(sizes.values(), default=0),
        "items_in_dup_clusters": in_dup,
        "share_items_in_dup_clusters": round(in_dup / len(texts), 4) if texts else 0.0,
        "thresholds": {"jaccard": JACCARD_THRESHOLD, "cosine": COSINE_THRESHOLD if semantic else None},
        "semantic_model": SEMANTIC_MODEL if semantic else None,
    }
    return pairs, [ids[c] for c in cluster_of], summary


def write_outputs(out_dir, ids, pairs, cluster_ids, summary):
    """Write pairs.csv, clusters.csv (id, cluster_id) and summary.json into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id_a", "id_b", "jaccard", "cosine"])
        writer.writeheader()
        writer.writerows(pairs)
    with open(out / "clusters.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "cluster_id"])
        writer.writerows(zip(ids, cluster_ids))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help=".jsonl or .csv with id and text columns")
    parser.add_argument("--out", required=True, help="output folder for pairs/clusters/summary")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--no-semantic", action="store_true", help="skip the MiniLM step")
    args = parser.parse_args()
    ids, texts = read_records(args.input, args.id_field, args.text_field)
    pairs, cluster_ids, summary = run(ids, texts, semantic=not args.no_semantic)
    write_outputs(args.out, ids, pairs, cluster_ids, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
