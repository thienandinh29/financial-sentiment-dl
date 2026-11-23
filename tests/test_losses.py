"""Tests for train/losses.py with values computed by hand."""
import math
import unittest

import torch
from torch import nn

from train.losses import FocalLoss, build_loss, class_weights


class LossTest(unittest.TestCase):
    def test_balanced_class_weights_from_train_counts(self):
        # counts 1, 2, 3 out of 6 -> 6 / (3 * c) = 2, 1, 2/3
        w = class_weights([0, 1, 1, 2, 2, 2])
        torch.testing.assert_close(w, torch.tensor([2.0, 1.0, 2 / 3]))

    def test_missing_class_raises(self):
        with self.assertRaises(ValueError):
            class_weights([0, 0, 2])

    def test_focal_hand_value(self):
        # logits [0, 0, 0] -> p_y = 1/3; loss = (2/3)^2 * log 3
        loss = FocalLoss(gamma=2)(torch.zeros(1, 3), torch.tensor([1]))
        self.assertAlmostEqual(loss.item(), (2 / 3) ** 2 * math.log(3), places=6)

    def test_focal_gamma_zero_equals_weighted_ce(self):
        torch.manual_seed(1)
        logits, target = torch.randn(8, 3), torch.tensor([0, 1, 2, 2, 1, 0, 2, 2])
        weight = torch.tensor([2.0, 1.0, 0.5])
        torch.testing.assert_close(FocalLoss(gamma=0, weight=weight)(logits, target),
                                   nn.CrossEntropyLoss(weight=weight)(logits, target))

    def test_focal_down_weights_easy_examples(self):
        easy = torch.tensor([[5.0, 0.0, 0.0]])
        ce = nn.CrossEntropyLoss()(easy, torch.tensor([0]))
        self.assertLess(FocalLoss(gamma=2)(easy, torch.tensor([0])).item(), ce.item() / 100)

    def test_weighted_focal_hand_value(self):
        # Two samples, logits [0, 0, 0] -> p_y = 1/3, per-sample loss L = (2/3)^2 * log 3.
        # Weights w_0 = 2, w_2 = 0.5: (2 L + 0.5 L) / (2 + 0.5) = L.
        # Sample 2 has logits [0, 0, log 2] for target 2 -> p_y = 1/2, loss (1/2)^2 * log 2.
        weight = torch.tensor([2.0, 1.0, 0.5])
        logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, math.log(2)]])
        loss = FocalLoss(gamma=2, weight=weight)(logits, torch.tensor([0, 2]))
        expected = (2 * (2 / 3) ** 2 * math.log(3) + 0.5 * 0.25 * math.log(2)) / 2.5
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_build_loss_passes_gamma(self):
        loss = build_loss("focal", [0, 1, 2, 2], gamma=3.0)
        self.assertEqual(loss.gamma, 3.0)
        torch.testing.assert_close(loss.weight, class_weights([0, 1, 2, 2]))

    def test_build_loss_names(self):
        labels = [0, 1, 2, 2]
        self.assertIsNone(build_loss("ce", labels).weight)
        torch.testing.assert_close(build_loss("weighted_ce", labels).weight, class_weights(labels))
        self.assertIsInstance(build_loss("focal", labels), FocalLoss)
        with self.assertRaises(ValueError):
            build_loss("hinge", labels)


if __name__ == "__main__":
    unittest.main()
