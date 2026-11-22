"""Unit tests for models/transformer.py and configs/E5.yaml, configs/E7.yaml (tasks 3.9 & 3.10)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from evaluate.metrics import read_preds, score
from models.transformer import (
    configure_encoder_freezing,
    is_classifier_head_param,
    linear_warmup_lr_multiplier,
    macro_f1_np,
    validate_transformer_config,
    write_preds,
)

try:
    import torch
except ImportError:
    torch = None


class FakeParam:
    """Lightweight parameter stand-in for testing encoder freezing without torch installed."""

    def __init__(self, count: int):
        self._count = count
        self.requires_grad = True

    def numel(self) -> int:
        return self._count


class FakeDistilBertModel:
    """Mock DistilBERT sequence classifier with encoder and classification head parameters."""

    def __init__(self):
        self._params = [
            ("distilbert.embeddings.word_embeddings.weight", FakeParam(23440896)),
            ("distilbert.transformer.layer.0.attention.q_lin.weight", FakeParam(589824)),
            ("distilbert.transformer.layer.5.ffn.lin2.weight", FakeParam(2359296)),
            ("pre_classifier.weight", FakeParam(589824)),
            ("pre_classifier.bias", FakeParam(768)),
            ("classifier.weight", FakeParam(2304)),
            ("classifier.bias", FakeParam(3)),
        ]

    def named_parameters(self):
        return list(self._params)

    def parameters(self):
        return [p for _, p in self._params]


class TransformerConfigAndFreezingTests(unittest.TestCase):
    def setUp(self):
        self.e5 = yaml.safe_load(Path("configs/E5.yaml").read_text(encoding="utf-8"))
        self.e7 = yaml.safe_load(Path("configs/E7.yaml").read_text(encoding="utf-8"))

    def test_e5_and_e7_configs_are_valid_and_differ_by_one_factor(self):
        validate_transformer_config(self.e5)
        validate_transformer_config(self.e7)
        self.assertEqual(self.e5["experiment"], "E5")
        self.assertEqual(self.e7["experiment"], "E7")
        self.assertFalse(self.e5["model"]["freeze_encoder"])
        self.assertTrue(self.e7["model"]["freeze_encoder"])

        # AGENTS.md Rule 4: E7 must copy E5 and change only freeze_encoder
        e5_normalized = copy.deepcopy(self.e5)
        e7_normalized = copy.deepcopy(self.e7)
        e5_normalized["experiment"] = "E_cmp"
        e7_normalized["experiment"] = "E_cmp"
        e5_normalized["model"]["freeze_encoder"] = True
        self.assertEqual(e5_normalized, e7_normalized)

    def test_validate_transformer_config_rejects_invalid_configs(self):
        bad_no_sha = copy.deepcopy(self.e5)
        bad_no_sha["data"]["split_sha256"] = None
        with self.assertRaises(ValueError):
            validate_transformer_config(bad_no_sha)

        bad_metric = copy.deepcopy(self.e5)
        bad_metric["train"]["early_stopping"]["metric"] = "test_macro_f1"
        with self.assertRaises(ValueError):
            validate_transformer_config(bad_metric)

        bad_labels = copy.deepcopy(self.e5)
        bad_labels["model"]["num_labels"] = 2
        with self.assertRaises(ValueError):
            validate_transformer_config(bad_labels)

    def test_encoder_freezing_e5_vs_e7(self):
        model_e5 = FakeDistilBertModel()
        stats_e5 = configure_encoder_freezing(model_e5, freeze_encoder=False)
        self.assertFalse(stats_e5["freeze_encoder"])
        self.assertEqual(stats_e5["trainable_params"], stats_e5["total_params"])
        self.assertEqual(stats_e5["frozen_params"], 0)

        model_e7 = FakeDistilBertModel()
        stats_e7 = configure_encoder_freezing(model_e7, freeze_encoder=True)
        self.assertTrue(stats_e7["freeze_encoder"])
        expected_head_params = 589824 + 768 + 2304 + 3
        self.assertEqual(stats_e7["trainable_params"], expected_head_params)
        self.assertEqual(stats_e7["frozen_params"], stats_e7["total_params"] - expected_head_params)
        for name, param in model_e7.named_parameters():
            self.assertEqual(param.requires_grad, is_classifier_head_param(name))

    def test_linear_warmup_lr_multiplier_schedule(self):
        self.assertAlmostEqual(linear_warmup_lr_multiplier(0, 100, warmup_ratio=0.1), 0.0)
        self.assertAlmostEqual(linear_warmup_lr_multiplier(5, 100, warmup_ratio=0.1), 0.5)
        self.assertAlmostEqual(linear_warmup_lr_multiplier(10, 100, warmup_ratio=0.1), 1.0)
        self.assertAlmostEqual(linear_warmup_lr_multiplier(55, 100, warmup_ratio=0.1), 0.5)
        self.assertAlmostEqual(linear_warmup_lr_multiplier(100, 100, warmup_ratio=0.1), 0.0)
        with self.assertRaises(ValueError):
            linear_warmup_lr_multiplier(0, 0, warmup_ratio=0.1)

    def test_write_preds_matches_shared_evaluation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preds_validation.jsonl"
            write_preds(p, ["id-1", "id-2"], [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
            preds = read_preds(p)
            scored = score(preds, {"id-1": 0, "id-2": 2})
            self.assertEqual(scored["macro_f1"], round((1.0 + 0.0 + 1.0) / 3, 4))
            self.assertAlmostEqual(macro_f1_np([0, 2], [0, 2]), 2.0 / 3.0, places=4)

    def test_integrated_e5_and_e7_results_artifacts(self):
        expected_macros = {
            ("E5", 1): 0.7968,
            ("E5", 2): 0.8053,
            ("E5", 3): 0.8068,
            ("E7", 1): 0.5939,
            ("E7", 2): 0.5898,
            ("E7", 3): 0.5901,
        }
        reference_ids = None
        for (exp, seed), exp_f1 in expected_macros.items():
            seed_dir = Path("results") / exp / f"seed{seed}"
            run_path = seed_dir / "run.json"
            pred_path = seed_dir / "preds_validation.jsonl"
            self.assertTrue(run_path.is_file(), f"Missing {run_path}")
            self.assertTrue(pred_path.is_file(), f"Missing {pred_path}")

            run_data = json.loads(run_path.read_text(encoding="utf-8"))
            self.assertEqual(run_data["config"]["experiment"], exp)
            self.assertEqual(run_data["seed"], seed)
            self.assertEqual(
                run_data["config"]["data"]["split_sha256"],
                "8c2c35ea789acf5b2bd9c74c5e59fa0df325034422be42ae7b47f2e8c9c46aaa",
            )
            self.assertEqual(run_data["model_stats"]["total_params"], 66955779)
            if exp == "E5":
                self.assertEqual(run_data["model_stats"]["trainable_params"], 66955779)
                self.assertEqual(run_data["model_stats"]["frozen_params"], 0)
            else:
                self.assertEqual(run_data["model_stats"]["trainable_params"], 592899)
                self.assertEqual(run_data["model_stats"]["frozen_params"], 66362880)
            self.assertEqual(round(float(run_data["val_macro_f1"]), 4), exp_f1)

            preds = read_preds(pred_path)
            self.assertEqual(len(preds), 1791)
            if reference_ids is None:
                reference_ids = set(preds.keys())
            else:
                self.assertEqual(set(preds.keys()), reference_ids)


if __name__ == "__main__":
    unittest.main()
