"""E0 baseline: TF-IDF + logistic regression, fit on train only, scored on validation only.

Usage:
    python -m models.tfidf_lr --config configs/E0.yaml
Writes results/E0/seed1/preds_validation.jsonl ({"id", "probs"} in LABELS order) and run.json.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from train.data import DEFAULT_RAW, LABELS, load_split_data
from train.runlog import run_metadata, write_json


def run(cfg, root="."):
    """Fit on train, predict validation, write the contract files; return validation Macro-F1."""
    started = datetime.now(timezone.utc)
    root, dcfg, mcfg = Path(root), cfg["data"], cfg["model"]
    data = load_split_data(root / dcfg["split_file"], dcfg["split_sha256"],
                           [root / p for p in dcfg.get("raw_files", DEFAULT_RAW)])
    train, val = data["train"], data["validation"]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=tuple(mcfg["ngram_range"]),
                                 min_df=mcfg["min_df"], sublinear_tf=True)
    x_train = vectorizer.fit_transform([t for _, t, _ in train])           # fit on train only
    clf = LogisticRegression(C=mcfg["C"], class_weight="balanced", max_iter=2000, random_state=cfg["seeds"][0])
    clf.fit(x_train, [y for _, _, y in train])
    assert list(clf.classes_) == [0, 1, 2], "classes must follow LABELS order"
    probs = clf.predict_proba(vectorizer.transform([t for _, t, _ in val]))
    f1 = f1_score([y for _, _, y in val], probs.argmax(1), average="macro")
    out = root / "results" / cfg["experiment"] / f"seed{cfg['seeds'][0]}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preds_validation.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for (i, _, _), p in zip(val, probs):
            f.write(json.dumps({"id": i, "probs": [round(float(x), 6) for x in p]}) + "\n")
    run_info = run_metadata(cfg, cfg["seeds"][0], started, cwd=root)
    run_info.update({"label_order": list(LABELS), "val_macro_f1": float(f1),
                     "vocab_size": len(vectorizer.vocabulary_)})
    write_json(out / "run.json", run_info)
    return f1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(f"E0 validation Macro-F1: {run(cfg):.4f}")


if __name__ == "__main__":
    main()
