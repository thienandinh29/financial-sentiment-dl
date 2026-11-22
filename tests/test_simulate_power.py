import unittest
import numpy as np
from scripts.simulate_power import macro, simulate


class SimulationTests(unittest.TestCase):
    def test_macro_and_identical_models(self):
        y = np.array([0, 1, 2, 0])
        p = np.array([0, 1, 1, 2])
        self.assertAlmostEqual(macro(y, p), (2 / 3 + 2 / 3 + 0) / 3)
        delta, rate = simulate(30, 0, np.random.default_rng(1), trials=3, bootstraps=20)
        self.assertEqual((delta, rate), (0, 0))


if __name__ == "__main__":
    unittest.main()
