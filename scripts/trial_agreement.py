"""Compute pairwise trial agreement after three independent human label files exist."""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

LABELS = ('bearish', 'neutral', 'bullish')


def read_rows(path, fields):
    """Read a CSV and enforce the exact contract, including unique nonempty IDs."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f'{path}: expected columns {fields}')
        rows = list(reader)
    ids = [row['id'].strip() for row in rows]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise ValueError(f'{path}: empty or duplicate ID')
    return rows


def analyze(trial_path, annotation_paths):
    """Return pairwise agreement, kappa and disagreement IDs for three complete reviews."""
    paths = [Path(path) for path in annotation_paths]
    if len(paths) != 3 or len({path.resolve() for path in paths}) != 3 or len({p.stem for p in paths}) != 3:
        raise ValueError('Provide exactly three distinct annotator files with different names')
    trial = read_rows(trial_path, ['id', 'url'])
    ids = {row['id'].strip() for row in trial}
    if len(ids) != 30 or any(not row['url'].strip() for row in trial):
        raise ValueError('Trial list must contain exactly 30 IDs and URLs')
    annotations = {}
    for path in paths:
        rows = read_rows(path, ['id', 'label', 'reason'])
        labels = {row['id'].strip(): row['label'].strip().lower() for row in rows}
        if set(labels) != ids or any(label not in LABELS for label in labels.values()):
            raise ValueError(f'{path}: IDs or labels do not match the trial contract')
        if any(not row['reason'].strip() for row in rows):
            raise ValueError(f'{path}: every label requires a reason')
        annotations[path.stem] = labels
    ordered_ids = sorted(ids)
    pairs = []
    for a, b in combinations(sorted(annotations), 2):
        x = [annotations[a][i] for i in ordered_ids]
        y = [annotations[b][i] for i in ordered_ids]
        # Kappa is undefined when both annotators use only the same single label.
        kappa = None if len(set(x + y)) == 1 else float(cohen_kappa_score(x, y, labels=LABELS))
        pairs.append({'annotators': [a, b], 'agreement': sum(p == q for p, q in zip(x, y)) / len(ids),
                      'kappa': kappa})
    disagreements = [i for i in ordered_ids if len({labels[i] for labels in annotations.values()}) > 1]
    return {'items': len(ids), 'pairs': pairs, 'disagreement_ids': disagreements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', nargs=3, help='three human annotation CSV files')
    parser.add_argument('--trial', default='labeling/trial30_ids.csv')
    args = parser.parse_args()
    print(json.dumps(analyze(args.trial, args.annotations), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
