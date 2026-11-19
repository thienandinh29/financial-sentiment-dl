"""Vocabulary and embedding matrices (GloVe or random) for the RNN/CNN models.

The vocabulary is fit on train texts only (AGENTS.md rule 3); validation and test texts are
only encoded with it. Index 0 is <pad> (zero vector), index 1 is <unk>; the tokenizer's
placeholders (<url>, <ticker>, <num>) are always in the vocabulary.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.tokenizer import SPECIAL_TOKENS, tokenize

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


class Vocab:
    """Token <-> id mapping with train token counts (used for the token-level OOV rate)."""

    def __init__(self, itos, counts=None):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.counts = Counter(counts or {})
        if self.itos[:2] != [PAD, UNK] or len(self.stoi) != len(self.itos):
            raise ValueError("Vocab must start with <pad>, <unk> and have unique tokens")

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len=None):
        """Map one text to ids (unknown tokens -> UNK_ID), truncated to max_len; no padding."""
        ids = [self.stoi.get(t, UNK_ID) for t in tokenize(text)]
        return ids[:max_len] if max_len else ids

    def save(self, path):
        """Write the vocabulary and train counts to JSON (stored with each run)."""
        Path(path).write_text(json.dumps({"itos": self.itos, "counts": self.counts}, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Read a vocabulary written by save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["itos"], data["counts"])


def build_vocab(train_texts, min_freq=1, max_size=None):
    """Fit a vocabulary on train texts only.

    Input: train texts; tokens seen fewer than min_freq times are dropped; max_size caps the
    total size including specials. Output: Vocab ordered <pad>, <unk>, placeholders, then by
    descending train frequency (ties alphabetical, so the order is deterministic).
    """
    counts = Counter(t for text in train_texts for t in tokenize(text))
    specials = [PAD, UNK, *SPECIAL_TOKENS]
    words = sorted((t for t, c in counts.items() if c >= min_freq and t not in specials),
                   key=lambda t: (-counts[t], t))
    itos = specials + words
    return Vocab(itos[:max_size] if max_size else itos, counts)


def read_glove(path, vocab):
    """Read GloVe vectors for the vocabulary's tokens only.

    Input: a GloVe .txt file ("word v1 ... vd" per line), a Vocab. Output: dict token -> float32
    vector; tokens missing from GloVe are absent. Lines with a different dimension raise.
    """
    wanted, found, dim = set(vocab.itos), {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, *values = line.rstrip("\n").split(" ")
            if dim is None:
                dim = len(values)
            elif len(values) != dim:
                raise ValueError(f"Inconsistent GloVe dimension at '{word}'")
            if word in wanted and word not in found:
                found[word] = np.asarray(values, dtype=np.float32)
    return found


def random_embeddings(vocab, dim, seed, std=None):
    """Return a (len(vocab), dim) float32 matrix ~ N(0, std^2), seeded; <pad> row is zero.

    Used for E4 (random embedding) and to fill rows that GloVe lacks.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, std if std is not None else 1.0 / np.sqrt(dim),
                        size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD_ID] = 0.0
    return matrix


def glove_embeddings(path, vocab, seed):
    """Build the embedding matrix from GloVe.

    Input: GloVe .txt path, Vocab, seed. Output: (matrix, stats). Rows found in GloVe copy the
    vector; <unk>, placeholders and missing tokens are random with GloVe's own std (seeded);
    <pad> is zero. stats has the type and train-token OOV rates.
    """
    found = read_glove(path, vocab)
    if not found:
        raise ValueError("No vocabulary token found in the GloVe file")
    vectors = np.stack(list(found.values()))
    matrix = random_embeddings(vocab, vectors.shape[1], seed, std=float(vectors.std()))
    for token, vector in found.items():
        matrix[vocab.stoi[token]] = vector
    matrix[PAD_ID] = 0.0
    words = [t for t in vocab.itos if t not in (PAD, UNK, *SPECIAL_TOKENS)]
    missing = [t for t in words if t not in found]
    total = sum(vocab.counts[t] for t in words)
    stats = {
        "vocab_size": len(vocab),
        "dim": int(vectors.shape[1]),
        "found_in_glove": len(found),
        "type_oov_rate": round(len(missing) / len(words), 4) if words else 0.0,
        "train_token_oov_rate": round(sum(vocab.counts[t] for t in missing) / total, 4) if total else 0.0,
        "top_oov": sorted(missing, key=lambda t: (-vocab.counts[t], t))[:20],
    }
    return matrix, stats


def unk_rate(vocab, texts):
    """Share of tokens in texts that map to <unk> (e.g. on validation; nothing is fit)."""
    ids = [i for text in texts for i in vocab.encode(text)]
    return round(sum(i == UNK_ID for i in ids) / len(ids), 4) if ids else 0.0


def embedding_layer(matrix, trainable=True):
    """Wrap a numpy matrix as nn.Embedding with padding_idx=PAD_ID (pad gets no gradient)."""
    return nn.Embedding.from_pretrained(torch.from_numpy(matrix), freeze=not trainable, padding_idx=PAD_ID)
