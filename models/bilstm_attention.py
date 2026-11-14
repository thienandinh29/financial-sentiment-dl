"""BiLSTM with additive self-attention pooling (E2); pooling='max' gives the E3 ablation."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.embeddings import embedding_layer
from models.registry import register_model

NUM_CLASSES = 3


class BiLSTMAttention(nn.Module):
    """Input: tokens (batch, time), lengths (batch,). Output: logits (batch, 3).

    The LSTM reads packed sequences, so padding never enters the recurrence. Attention scores
    a_t = v . tanh(W h_t) are masked to the real tokens and softmax-normalised; the sentence vector
    is sum_t a_t h_t. With pooling='max' it is the max of h_t over the real tokens instead.
    The last attention weights (batch, time) stay in `last_attention` for visualisation (week 7).
    """

    def __init__(self, matrix, hidden_size=128, num_layers=2, bidirectional=True, pooling="attention",
                 dropout=0.3, trainable=True):
        super().__init__()
        if pooling not in ("attention", "max"):
            raise ValueError(f"pooling must be 'attention' or 'max', got '{pooling}'")
        self.embedding = embedding_layer(matrix, trainable=trainable)
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.pooling = pooling
        self.attn = nn.Linear(out_dim, out_dim)
        self.attn_v = nn.Linear(out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(out_dim, NUM_CLASSES)
        self.last_attention = None

    def forward(self, tokens, lengths):
        emb = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=tokens.size(1))
        mask = torch.arange(h.size(1), device=h.device)[None, :] < lengths.to(h.device)[:, None]
        if self.pooling == "attention":
            scores = self.attn_v(torch.tanh(self.attn(h))).squeeze(-1).masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            self.last_attention = weights.detach()
            sentence = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        else:
            sentence = h.masked_fill(~mask[..., None], float("-inf")).max(dim=1).values
        return self.out(self.dropout(sentence))


@register_model("bilstm_attention")
def build(model_cfg, matrix):
    """Build BiLSTM-Attention (or its max-pooling ablation) from the `model` section of a config."""
    return BiLSTMAttention(matrix, model_cfg.get("hidden_size", 128), model_cfg.get("num_layers", 2),
                           model_cfg.get("bidirectional", True), model_cfg.get("pooling", "attention"),
                           model_cfg.get("dropout", 0.3), model_cfg.get("embedding_trainable", True))
