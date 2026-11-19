"""Losses for the classifiers: plain CE, weighted CE (E2 reference) and focal loss (E13).

Class weights are computed from train labels only (AGENTS.md rule 3).
"""
import torch
from torch import nn
from torch.nn import functional as F


def class_weights(train_labels, num_classes=3):
    """Return 'balanced' weights n / (k * count_c) as a float32 tensor.

    Input: train label indices. Output: tensor of shape (num_classes,). A class absent from
    train raises, because its weight would be undefined.
    """
    counts = torch.bincount(torch.as_tensor(list(train_labels)), minlength=num_classes).float()
    if (counts == 0).any():
        raise ValueError(f"Class missing from train labels: counts {counts.tolist()}")
    return counts.sum() / (num_classes * counts)


class FocalLoss(nn.Module):
    """Focal loss -w_y * (1 - p_y)^gamma * log p_y, averaged like weighted CE.

    gamma = 0 gives (weighted) cross-entropy. The mean divides by the sum of the sample
    weights, the same convention as nn.CrossEntropyLoss(weight=...).
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp.exp()) ** self.gamma) * logp
        if self.weight is None:
            return loss.mean()
        w = self.weight[target]
        return (w * loss).sum() / w.sum()


def build_loss(name, train_labels, num_classes=3, gamma=2.0):
    """Return the loss named in the config: 'ce', 'weighted_ce' or 'focal' (weighted, gamma)."""
    if name == "ce":
        return nn.CrossEntropyLoss()
    weight = class_weights(train_labels, num_classes)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weight)
    if name == "focal":
        return FocalLoss(gamma=gamma, weight=weight)
    raise ValueError(f"Unknown loss '{name}'")
