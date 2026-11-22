import unittest

from evaluate.stats import compare


def probs(label):
    return [float(i == label) for i in range(3)]


class StatsTests(unittest.TestCase):
    def test_hand_computed_and_reproducible(self):
        gold = {str(i): i % 3 for i in range(6)}
        a = {i: probs(y) for i, y in gold.items()}
        b = {i: probs((y + 1) % 3 if i in ("0", "1", "2") else y) for i, y in gold.items()}
        result = compare(a, b, gold, seed=42)
        self.assertAlmostEqual(result["difference_b_minus_a"], -0.5)
        self.assertEqual(result["mcnemar_exact"], {"only_a_correct": 3, "only_b_correct": 0, "p_value": 0.25})
        self.assertEqual(result, compare(a, b, gold, seed=42))
        self.assertLessEqual(result["bootstrap"]["ci95_percentile"][1], 0)

    def test_identical_and_id_mismatch(self):
        gold = {"x": 0, "y": 1, "z": 2}
        pred = {i: probs(y) for i, y in gold.items()}
        result = compare(pred, pred, gold)
        self.assertEqual(result["bootstrap"]["ci95_percentile"], [0, 0])
        self.assertEqual(result["mcnemar_exact"]["p_value"], 1)
        with self.assertRaises(ValueError):
            compare(pred, {"x": probs(0)}, gold)


if __name__ == "__main__":
    unittest.main()
