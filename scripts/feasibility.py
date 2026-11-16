"""Run TV2's dedup implementation on unsplit FNS, with reproducible provenance."""

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eda import FNS_FILES, read_rows
from scripts.run_metadata import run_metadata, sha256


def leakage_bound(cluster_ids, test_fraction=0.15):
    """Return a loose cluster-based upper bound, not a feasible split guarantee."""
    sizes = Counter(cluster_ids)
    capacity = sum(size - 1 for size in sizes.values())
    test_size = math.ceil(len(cluster_ids) * test_fraction)
    return {"assumed_test_fraction": test_fraction, "approximate_test_size": test_size,
            "max_leakable_items_loose": capacity,
            "loose_upper_fraction": min(1, capacity / test_size) if test_size else 0,
            "caveat": "Loose bound only: cluster allocation, class balance and retained train "
                      "size may lower feasibility. No split constructed; E10 unchanged."}


def main():
    """Load the supplied TV2 script, run both edge types, save ID-only outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedup-script", required=True, type=Path)
    parser.add_argument("--model-revision", required=True,
                        help="Immutable HF commit for all-MiniLM-L6-v2")
    args = parser.parse_args()
    output = ROOT / "results/feasibility"
    if (output / "run.json").exists():
        parser.error("Previous feasibility run exists; preserve it before rerunning")
    start = time.perf_counter()
    spec = importlib.util.spec_from_file_location("tv2_dedup", args.dedup_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    rows = read_rows(FNS_FILES)
    ids = [row["id"] for row in rows]
    texts = [row["text"] for row in rows]
    model_path = snapshot_download(module.SEMANTIC_MODEL, revision=args.model_revision,
        cache_dir=ROOT / "data/cache/huggingface", token=False,
        allow_patterns=["config.json", "config_sentence_transformers.json", "modules.json",
                        "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json",
                        "tokenizer_config.json", "vocab.txt", "model.safetensors",
                        "1_Pooling/config.json"])
    model = SentenceTransformer(model_path, device="cpu")
    embeddings = model.encode(texts, batch_size=256, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False).astype("float32")
    for name, semantic in (("fns", True), ("fns-lexical", False)):
        pairs, clusters, summary = module.run(ids, texts, semantic=semantic, embeddings=embeddings)
        summary["leakage_bound"] = leakage_bound(clusters)
        summary["pair_fraction_all_possible"] = len(pairs) / (len(ids) * (len(ids) - 1) / 2)
        summary["candidate_recall_note"] = "Lexical LSH proposes candidates; recall is not guaranteed"
        module.write_outputs(output / name, ids, pairs, clusters, summary)
        print(json.dumps(summary, indent=2), flush=True)
    metadata = run_metadata(start, {
        "dedup_script": args.dedup_script.relative_to(ROOT).as_posix()
            if args.dedup_script.is_absolute() else args.dedup_script.as_posix(),
        "dedup_script_sha256": sha256(args.dedup_script),
        "script_sha256": sha256(Path(__file__)),
        "model": module.SEMANTIC_MODEL, "model_revision": args.model_revision,
        "jaccard": module.JACCARD_THRESHOLD, "cosine": module.COSINE_THRESHOLD,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in FNS_FILES},
    }, ["numpy", "torch", "sentence-transformers", "transformers", "datasketch", "huggingface_hub"])
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
