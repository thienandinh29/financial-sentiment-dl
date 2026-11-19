"""End-to-end check of train.run on CPU: tiny model, toy data, 2 epochs (task 2.10)."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from models.embeddings import embedding_layer
from train.data import write_split
from train.run import MODELS, register_model, run_experiment

WORDS = {"bullish": ["rises", "gains", "beats", "jumps"], "bearish": ["falls", "drops", "misses", "slumps"],
         "neutral": ["meets", "holds", "says", "files"]}


@register_model("mean_pool_test")
def mean_pool(model_cfg, matrix):
    """Mean of embeddings over real tokens -> linear layer; enough to learn the toy data."""
    class MeanPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = embedding_layer(matrix, trainable=True)
            self.out = nn.Linear(matrix.shape[1], 3)

        def forward(self, tokens, lengths):
            mask = (tokens != 0).unsqueeze(-1).float()
            return self.out((self.emb(tokens) * mask).sum(1) / lengths.clamp(min=1).unsqueeze(1).float())
    return MeanPool()


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = random.Random(0)
        rows = [{"id": f"x{k:03d}", "text": f"Company {rng.choice(WORDS[label])} today", "label": label}
                for k, label in enumerate(rng.choice(list(WORDS)) for _ in range(90))]
        raw = self.root / "raw.jsonl"
        raw.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        assignment = [(r["id"], "train" if k < 60 else "validation" if k < 80 else "test")
                      for k, r in enumerate(rows)]
        self.n_val = 20
        sha = write_split(self.root / "split.jsonl", assignment)
        self.cfg = {
            "experiment": "E_test",
            "data": {"split_file": "split.jsonl", "split_sha256": sha, "raw_files": ["raw.jsonl"], "max_len": 16},
            "model": {"name": "mean_pool_test", "embedding": "random", "embedding_dim": 16},
            "train": {"loss": "weighted_ce", "optimizer": "adam", "lr": 0.05, "batch_size": 8, "max_epochs": 2,
                      "early_stopping": {"metric": "val_macro_f1", "patience": 3}, "mixed_precision": True},
            "seeds": [1],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def run_once(self, tag):
        return run_experiment(self.cfg, root=self.root, results_dir=self.root / tag / "results",
                              checkpoint_dir=self.root / tag / "ckpt", device="cpu")

    def test_trains_and_writes_contract_files(self):
        self.run_once("a")
        out = self.root / "a/results/E_test/seed1"
        run = json.loads((out / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(len(run["history"]), 2)
        self.assertLess(run["history"][1]["train_loss"], run["history"][0]["train_loss"])
        for key in ("commit", "config", "seed", "libraries", "gpu", "elapsed_seconds", "best_epoch", "val_macro_f1"):
            self.assertIn(key, run)
        self.assertEqual(run["label_order"], ["bearish", "neutral", "bullish"])
        preds = [json.loads(l) for l in (out / "preds_validation.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(preds), self.n_val)
        self.assertTrue(all(abs(sum(p["probs"]) - 1) < 1e-4 and len(p["probs"]) == 3 for p in preds))
        self.assertFalse((out / "preds_test.jsonl").exists())
        self.assertTrue((self.root / "a/ckpt/E_test/seed1/best.pt").exists())
        self.assertTrue((out / "vocab.json").exists())

    def test_same_seed_gives_same_predictions(self):
        self.run_once("a")
        self.run_once("b")
        read = lambda tag: (self.root / tag / "results/E_test/seed1/preds_validation.jsonl").read_text(encoding="utf-8")
        self.assertEqual(read("a"), read("b"))

    def test_unknown_model_is_rejected(self):
        self.cfg["model"]["name"] = "bilstm_attention_not_yet"
        with self.assertRaises(ValueError):
            self.run_once("c")

    def test_registry_keeps_test_model(self):
        self.assertIn("mean_pool_test", MODELS)


if __name__ == "__main__":
    unittest.main()
