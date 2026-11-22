"""Synthetic paired-bootstrap detection simulation; never reads project test data."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scripts.run_metadata import run_metadata, sha256


def macro(y, p, axis=-1):
    """Macro-F1 for arrays of true/predicted labels, including absent classes as zero."""
    values = []
    for k in range(3):
        tp = np.sum((y == k) & (p == k), axis=axis)
        den = np.sum(y == k, axis=axis) + np.sum(p == k, axis=axis)
        values.append(np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0))
    return np.mean(values, axis=0)


def simulate(n, correction, rng, trials=100, bootstraps=1000):
    """Simulate paired classifiers, returning mean observed difference and rejection rate."""
    deltas, rejections = [], 0
    for _ in range(trials):
        y = rng.choice(3, n, p=[0.25, 0.50, 0.25])
        wrong = rng.random(n) < 0.30
        a = np.where(wrong, (y + rng.integers(1, 3, n)) % 3, y)
        b = np.where(wrong & (rng.random(n) < correction), y, a)
        deltas.append(float(macro(y, b) - macro(y, a)))
        # Batch resamples to keep memory bounded.
        diffs = []
        for start in range(0, bootstraps, 100):
            ix = rng.integers(0, n, size=(min(100, bootstraps - start), n))
            diffs.extend(macro(y[ix], b[ix]) - macro(y[ix], a[ix]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rejections += bool(lo > 0 or hi < 0)
    return float(np.mean(deltas)), rejections / trials


def main():
    """Write simulation table and provenance to results/power/."""
    start = time.perf_counter()
    settings = {"seed": 1, "sample_sizes": [1791, 300], "correction_rates": [0, .01, .02, .04, .06, .08, .12, .16, .24],
                "trials": 100, "bootstrap_repetitions": 1000, "prevalence": [.25, .5, .25], "baseline_accuracy": .7}
    rng = np.random.default_rng(settings["seed"])
    out = Path("results/power")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in settings["sample_sizes"]:
        for correction in settings["correction_rates"]:
            delta, power = simulate(n, correction, rng, settings["trials"], settings["bootstrap_repetitions"])
            rows.append({"n": n, "correction_rate": correction, "mean_delta_macro_f1": delta, "fraction_ci_excludes_zero": power})
    with (out / "simulation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    meta = run_metadata(start, settings, ["numpy"])
    meta["script_sha256"] = sha256(__file__)
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
