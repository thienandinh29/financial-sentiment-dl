"""TextCNN (Kim, 2014) for E1: parallel 1-D convolutions over word embeddings + max-over-time pooling."""
import torch
from torch import nn
from torch.nn import functional as F

from models.embeddings import PAD_ID, embedding_layer
from models.registry import register_model

NUM_CLASSES = 3


class TextCNN(nn.Module):
    """Input: tokens (batch, time) padded with PAD_ID, lengths (batch,). Output: logits (batch, 3).

    Each filter size k gives num_filters feature maps (ReLU). Only windows that start inside the
    real text are max-pooled, so extra padding never changes the output; a text shorter than k
    keeps its single window (real tokens + zero padding vectors).
    """

    def __init__(self, matrix, filter_sizes=(3, 4, 5), num_filters=100, dropout=0.3, trainable=True):
        super().__init__()
        self.embedding = embedding_layer(matrix, trainable=trainable)
        self.filter_sizes = tuple(filter_sizes)
        self.convs = nn.ModuleList(nn.Conv1d(self.embedding.embedding_dim, num_filters, k)
                                   for k in self.filter_sizes)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(num_filters * len(self.filter_sizes), NUM_CLASSES)

    def forward(self, tokens, lengths):
        lengths = lengths.to(tokens.device)
        if tokens.size(1) < max(self.filter_sizes):          # every filter needs at least one window
            tokens = F.pad(tokens, (0, max(self.filter_sizes) - tokens.size(1)), value=PAD_ID)
        x = self.dropout(self.embedding(tokens)).transpose(1, 2)    # (batch, dim, time)
        pooled = []
        for k, conv in zip(self.filter_sizes, self.convs):
            h = F.relu(conv(x))                                       # (batch, filters, time - k + 1)
            starts = torch.arange(h.size(2), device=h.device)
            valid = starts[None, :] < torch.clamp(lengths - k + 1, min=1)[:, None]
            pooled.append(h.masked_fill(~valid[:, None, :], float("-inf")).max(dim=2).values)
        return self.out(self.dropout(torch.cat(pooled, dim=1)))


@register_model("textcnn")
def build(model_cfg, matrix):
    """Build TextCNN from the `model` section of a config."""
    return TextCNN(matrix, model_cfg.get("filter_sizes", (3, 4, 5)), model_cfg.get("num_filters", 100),
                   model_cfg.get("dropout", 0.3), model_cfg.get("embedding_trainable", True))
