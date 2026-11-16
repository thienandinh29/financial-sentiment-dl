"""Shared evaluation for every model: preds_<split>.jsonl + gold labels -> Macro-F1, per-class F1, confusion.

Usage:
    python -m evaluate.metrics --preds results/E0/seed1/preds_validation.jsonl --config configs/E0.yaml
Only validation is scored before week 6 (AGENTS.md rule 1).
"""
import argparse
import json
import math
from pathlib import Path

import yaml
from sklearn.metrics import confusion_matrix, f1_score

from train.data import DEFAULT_RAW, LABELS, load_split_data


def read_preds(path):
    """Read {"id", "probs"} lines into dict id -> probs; each probs has 3 values in [0, 1] summing to ~1."""
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "probs" not in row:
                raise ValueError(f"Prediction row must contain 'id' and 'probs': {row}")
            probs = row["probs"]
            if (
                not isinstance(probs, list)
                or len(probs) != len(LABELS)
                or not all(isinstance(p, (int, float)) and math.isfinite(float(p)) and 0.0 <= float(p) <= 1.0 for p in probs)
                or abs(sum(probs) - 1.0) > 1e-3
            ):
                raise ValueError(f"Bad probs for id {row['id']}: {probs}")
            if row["id"] in preds:
                raise ValueError(f"Duplicate id in predictions: {row['id']}")
            preds[row["id"]] = [float(p) for p in probs]
    return preds


def score(preds, gold):
    """Score predictions against gold labels (dict id -> label index); ids must match exactly.

    Output: {"n", "macro_f1", "f1": {label: value}, "confusion": rows = gold, cols = predicted}.
    """
    if set(preds) != set(gold):
        raise ValueError(
            f"Ids differ: {len(set(gold) - set(preds))} missing, {len(set(preds) - set(gold))} extra"
        )
    ids = sorted(gold)
    y_true = [gold[i] for i in ids]
    y_pred = [max(range(len(LABELS)), key=lambda k: preds[i][k]) for i in ids]
    labels = list(range(len(LABELS)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": len(ids),
        "macro_f1": round(float(per_class.mean()), 4),
        "f1": {LABELS[k]: round(float(v), 4) for k, v in enumerate(per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def validation_gold(cfg, root="."):
    """Return dict id -> label index for the validation split named in a config."""
    root, dcfg = Path(root), cfg["data"]
    data = load_split_data(
        root / dcfg["split_file"],
        dcfg["split_sha256"],
        [root / p for p in dcfg.get("raw_files", DEFAULT_RAW)],
        splits=("validation",),
    )
    return {i: y for i, _, y in data["validation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preds", required=True)
    parser.add_argument("--config", required=True, help="config whose split gives the gold labels")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(score(read_preds(args.preds), validation_gold(cfg)), indent=2))


if __name__ == "__main__":
    main()
