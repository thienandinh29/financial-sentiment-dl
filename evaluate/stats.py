"""Paired validation comparison: percentile bootstrap and exact McNemar test."""
import argparse
import json

import numpy as np
from scipy.stats import binomtest
import yaml

from evaluate.metrics import read_preds, validation_gold


def compare(pred_a, pred_b, gold, repetitions=1000, seed=1):
    """Compare aligned ID->probabilities maps against ID->class; return B minus A and CI."""
    if not gold or set(pred_a) != set(gold) or set(pred_b) != set(gold):
        raise ValueError("Gold and both prediction files must have the same nonempty ID set")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    ids = sorted(gold)
    y = np.asarray([gold[i] for i in ids])
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Gold labels must be 0, 1 or 2")
    a = np.asarray([np.argmax(pred_a[i]) for i in ids])
    b = np.asarray([np.argmax(pred_b[i]) for i in ids])

    def macro_f1(truth, predicted):
        scores = []
        for label in range(3):
            tp = np.count_nonzero((truth == label) & (predicted == label))
            denominator = np.count_nonzero(truth == label) + np.count_nonzero(predicted == label)
            scores.append(2 * tp / denominator if denominator else 0.0)
        return float(np.mean(scores))

    difference = macro_f1(y, b) - macro_f1(y, a)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        ix = rng.integers(0, len(ids), size=len(ids))
        samples.append(macro_f1(y[ix], b[ix]) - macro_f1(y[ix], a[ix]))
    only_a = int(np.count_nonzero((a == y) & (b != y)))
    only_b = int(np.count_nonzero((a != y) & (b == y)))
    discordant = only_a + only_b
    return {"n": len(ids), "difference_b_minus_a": difference,
            "bootstrap": {"repetitions": repetitions, "seed": seed,
                          "ci95_percentile": np.percentile(samples, [2.5, 97.5]).tolist()},
            "mcnemar_exact": {"only_a_correct": only_a, "only_b_correct": only_b,
                               "p_value": float(binomtest(only_a, discordant, 0.5).pvalue) if discordant else 1.0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    print(json.dumps(compare(read_preds(args.a), read_preds(args.b), validation_gold(cfg),
                             args.repetitions, args.seed), indent=2))


if __name__ == "__main__":
    main()
