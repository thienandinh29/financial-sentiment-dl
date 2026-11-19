"""Tests for models/textcnn.py and models/bilstm_attention.py: shapes, padding, attention, registry."""
import unittest

import torch

from models.bilstm_attention import BiLSTMAttention
from models.embeddings import PAD_ID, Vocab, random_embeddings
from models.textcnn import TextCNN
from train.run import MODELS

TOKENS = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 0, 0, 0], [11, 0, 0, 0, 0, 0]])
LENGTHS = torch.tensor([6, 3, 1])


def matrix(size=20, dim=8):
    """A seeded random embedding matrix for a toy vocabulary (<pad> row is zero)."""
    return random_embeddings(Vocab(["<pad>", "<unk>"] + [f"w{i}" for i in range(size - 2)]), dim, seed=1)


def models():
    torch.manual_seed(1)
    m = matrix()
    return {"textcnn": TextCNN(m, num_filters=4), "attention": BiLSTMAttention(m, hidden_size=5),
            "max": BiLSTMAttention(m, hidden_size=5, pooling="max")}


class ModelTest(unittest.TestCase):
    def test_output_shape_and_gradients_flow(self):
        for name, model in models().items():
            logits = model(TOKENS, LENGTHS)
            self.assertEqual(tuple(logits.shape), (3, 3), name)
            logits.sum().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), name)

    def test_extra_padding_does_not_change_the_output(self):
        for name, model in models().items():
            model.eval()
            padded = torch.cat([TOKENS, torch.full((3, 7), PAD_ID)], dim=1)
            torch.testing.assert_close(model(TOKENS, LENGTHS), model(padded, LENGTHS), msg=name)

    def test_attention_sums_to_one_over_real_tokens_only(self):
        model = models()["attention"].eval()
        model(TOKENS, LENGTHS)
        weights = model.last_attention
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
        self.assertTrue(torch.all(weights[1, 3:] == 0) and torch.all(weights[2, 1:] == 0))

    def test_textcnn_handles_text_shorter_than_its_filters(self):
        out = models()["textcnn"].eval()(torch.tensor([[5]]), torch.tensor([1]))
        self.assertEqual(tuple(out.shape), (1, 3))
        self.assertTrue(torch.isfinite(out).all())

    def test_built_from_config_through_the_registry(self):
        m = matrix()
        self.assertIsInstance(MODELS["textcnn"]({"filter_sizes": [2, 3], "num_filters": 4}, m), TextCNN)
        bilstm = MODELS["bilstm_attention"]({"hidden_size": 4, "pooling": "max", "embedding_trainable": False}, m)
        self.assertEqual(bilstm.pooling, "max")
        self.assertFalse(bilstm.embedding.weight.requires_grad)
        with self.assertRaises(ValueError):
            BiLSTMAttention(m, pooling="mean")


if __name__ == "__main__":
    unittest.main()
