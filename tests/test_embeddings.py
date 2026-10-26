"""Tests for models/embeddings.py with a tiny fake GloVe file."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.embeddings import (PAD_ID, UNK_ID, Vocab, build_vocab, embedding_layer,
                               glove_embeddings, random_embeddings, unk_rate)

TRAIN = ["Apple shares rise", "Apple shares fall on $TSLA news", "Oil rises 5%"]
GLOVE = {"apple": [1, 0, 0], "shares": [0, 1, 0], "rise": [0, 0, 1], "oil": [1, 1, 0],
         "fall": [0, 1, 1], "unused": [9, 9, 9]}


class VocabTest(unittest.TestCase):
    def test_order_specials_then_frequency_then_alphabetical(self):
        vocab = build_vocab(TRAIN)
        self.assertEqual(vocab.itos[:5], ["<pad>", "<unk>", "<url>", "<ticker>", "<num>"])
        self.assertEqual(vocab.itos[5:7], ["apple", "shares"])          # seen twice
        self.assertEqual(vocab.itos[7:], sorted(vocab.itos[7:]))         # seen once, A-Z

    def test_fit_on_train_only_unseen_words_are_unk(self):
        vocab = build_vocab(TRAIN)
        self.assertNotIn("gold", vocab.stoi)
        self.assertEqual(vocab.encode("Gold shares rise"), [UNK_ID, vocab.stoi["shares"], vocab.stoi["rise"]])
        self.assertEqual(unk_rate(vocab, ["Gold shares rise"]), round(1 / 3, 4))

    def test_min_freq_max_len_and_save_load(self):
        vocab = build_vocab(TRAIN, min_freq=2)
        self.assertEqual(vocab.itos[5:], ["apple", "shares"])
        self.assertEqual(len(vocab.encode("Apple shares rise", max_len=2)), 2)
        with tempfile.TemporaryDirectory() as tmp:
            vocab.save(Path(tmp) / "vocab.json")
            loaded = Vocab.load(Path(tmp) / "vocab.json")
        self.assertEqual(loaded.itos, vocab.itos)
        self.assertEqual(loaded.counts, vocab.counts)


class EmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.glove = Path(self.tmp.name) / "glove.txt"
        self.glove.write_text("".join(f"{w} {' '.join(map(str, v))}\n" for w, v in GLOVE.items()),
                              encoding="utf-8")
        self.vocab = build_vocab(TRAIN)

    def tearDown(self):
        self.tmp.cleanup()

    def test_glove_rows_copied_pad_zero_missing_random_and_seeded(self):
        matrix, stats = glove_embeddings(self.glove, self.vocab, seed=1)
        self.assertEqual(matrix.shape, (len(self.vocab), 3))
        np.testing.assert_array_equal(matrix[self.vocab.stoi["apple"]], [1, 0, 0])
        np.testing.assert_array_equal(matrix[PAD_ID], [0, 0, 0])
        self.assertTrue(np.any(matrix[UNK_ID] != 0))
        again, _ = glove_embeddings(self.glove, self.vocab, seed=1)
        np.testing.assert_array_equal(matrix, again)
        self.assertEqual(stats["found_in_glove"], 5)
        self.assertEqual(set(stats["top_oov"]), {"news", "on", "rises", "%"})
        self.assertEqual(stats["type_oov_rate"], round(4 / 9, 4))
        self.assertEqual(stats["train_token_oov_rate"], round(4 / 11, 4))

    def test_wrong_dimension_raises(self):
        with open(self.glove, "a", encoding="utf-8") as f:
            f.write("bad 1 2\n")
        with self.assertRaises(ValueError):
            glove_embeddings(self.glove, self.vocab, seed=1)

    def test_random_embeddings_for_e4(self):
        a = random_embeddings(self.vocab, 4, seed=7)
        np.testing.assert_array_equal(a, random_embeddings(self.vocab, 4, seed=7))
        self.assertFalse(np.array_equal(a, random_embeddings(self.vocab, 4, seed=8)))
        np.testing.assert_array_equal(a[PAD_ID], 0)

    def test_layer_freezes_on_request_and_pad_gets_no_gradient(self):
        matrix = random_embeddings(self.vocab, 4, seed=1)
        self.assertFalse(embedding_layer(matrix, trainable=False).weight.requires_grad)
        layer = embedding_layer(matrix, trainable=True)
        layer(torch.tensor([[PAD_ID, UNK_ID]])).sum().backward()
        self.assertTrue(torch.all(layer.weight.grad[PAD_ID] == 0))
        self.assertTrue(torch.any(layer.weight.grad[UNK_ID] != 0))


if __name__ == "__main__":
    unittest.main()
