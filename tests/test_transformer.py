"""Unit tests for models/transformer.py and configs/E5.yaml, E6.yaml, E7.yaml, E12.yaml (tasks 3.9, 3.10, 3.13, 3.14)."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from evaluate.metrics import read_preds, score
from models.llm_soft_labels import DEFAULT_HEADLINES, EXPECTED_HEADLINES_SHA256
from models.transformer import (
    configure_encoder_freezing,
    distillation_kl_loss,
    format_e6_grid_markdown,
    is_classifier_head_param,
    linear_warmup_lr_multiplier,
    load_pseudo_labeled_examples,
    macro_f1_np,
    prepare_training_examples_with_pseudo,
    softmax_temperature,
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
        self.e6 = yaml.safe_load(Path("configs/E6.yaml").read_text(encoding="utf-8"))
        self.e7 = yaml.safe_load(Path("configs/E7.yaml").read_text(encoding="utf-8"))
        self.e12 = yaml.safe_load(Path("configs/E12.yaml").read_text(encoding="utf-8"))

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

    def test_e12_and_e6_configs_are_valid_and_follow_one_change_rule(self):
        validate_transformer_config(self.e12)
        validate_transformer_config(self.e6)
        self.assertEqual(self.e12["experiment"], "E12")
        self.assertEqual(self.e12["data"]["pseudo_mode"], "hard")
        self.assertEqual(self.e12["data"]["pseudo_labels"], "results/llm_soft_labels/p2_soft_labels.jsonl")

        # E12 differs from E5 only by experiment name and pseudo_labels + pseudo_mode: hard
        e5_plus_hard = copy.deepcopy(self.e5)
        e5_plus_hard["experiment"] = "E12"
        e5_plus_hard["data"]["pseudo_labels"] = "results/llm_soft_labels/p2_soft_labels.jsonl"
        e5_plus_hard["data"]["pseudo_mode"] = "hard"
        self.assertEqual(e5_plus_hard, self.e12)

        # E6 differs from E12 only by experiment name, pseudo_mode: soft, and distill_alpha/distill_temperature
        self.assertEqual(self.e6["experiment"], "E6")
        self.assertEqual(self.e6["data"]["pseudo_mode"], "soft")
        e12_plus_soft = copy.deepcopy(self.e12)
        e12_plus_soft["experiment"] = "E6"
        e12_plus_soft["data"]["pseudo_mode"] = "soft"
        e12_plus_soft["data"]["distill_alpha"] = self.e6["data"]["distill_alpha"]
        e12_plus_soft["data"]["distill_temperature"] = self.e6["data"]["distill_temperature"]
        self.assertEqual(e12_plus_soft, self.e6)

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

        bad_pseudo_mode = copy.deepcopy(self.e12)
        bad_pseudo_mode["data"]["pseudo_mode"] = "invalid_mode"
        with self.assertRaises(ValueError):
            validate_transformer_config(bad_pseudo_mode)

        bad_soft_temp = copy.deepcopy(self.e6)
        bad_soft_temp["data"]["distill_temperature"] = 0.0
        with self.assertRaises(ValueError):
            validate_transformer_config(bad_soft_temp)

    def test_temperature_scaled_teacher_distribution_and_distillation_kl_loss(self):
        # Hand-computed check for T = 2 (docs/guides/TV3-tuan4.md section 3.14)
        letter_logp = [-4.0, -2.0, 0.0]
        target_t2 = softmax_temperature(letter_logp, temperature=2.0)
        # Divided by T=2: [-2.0, -1.0, 0.0] -> exp(-2), exp(-1), 1.0
        e0, e1, e2 = math.exp(-2.0), math.exp(-1.0), 1.0
        z = e0 + e1 + e2
        expected_t2 = [e0 / z, e1 / z, e2 / z]
        for actual, exp in zip(target_t2, expected_t2):
            self.assertAlmostEqual(actual, exp, places=7)
        self.assertAlmostEqual(sum(target_t2), 1.0, places=7)

        # Higher temperature produces softer (higher entropy) distribution
        target_t1 = softmax_temperature(letter_logp, temperature=1.0)
        target_t4 = softmax_temperature(letter_logp, temperature=4.0)
        self.assertGreater(max(target_t1), max(target_t2))
        self.assertGreater(max(target_t2), max(target_t4))

        # When alpha = 0, distillation loss must be 0.0 (docs/guides/TV3-tuan4.md section 3.14)
        student_logits = [[1.0, -1.0, 0.5]]
        loss_alpha_zero = distillation_kl_loss(student_logits, [target_t2], temperature=2.0, alpha=0.0)
        self.assertEqual(loss_alpha_zero, 0.0)

        # When student logits match teacher letter_logp, KL divergence must be 0.0
        loss_exact_match = distillation_kl_loss([letter_logp], [target_t2], temperature=2.0, alpha=0.5)
        self.assertAlmostEqual(loss_exact_match, 0.0, places=7)

        # Hand-computed non-zero KL at T = 2, alpha = 0.5: alpha * T^2 = 0.5 * 4 = 2.0
        student_p_t2 = softmax_temperature(student_logits[0], temperature=2.0)
        expected_kl = sum(q * (math.log(q) - math.log(p)) for q, p in zip(target_t2, student_p_t2))
        actual_loss = distillation_kl_loss(student_logits, [target_t2], temperature=2.0, alpha=0.5)
        self.assertAlmostEqual(actual_loss, 0.5 * (2.0 ** 2) * expected_kl, places=7)

    def test_load_pseudo_labeled_examples_hard_and_soft_and_leakage_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            headlines_path = root / "headlines.jsonl"
            pseudo_path = root / "p2_soft_labels.jsonl"

            raw_rows = [
                {"id": "u-bear", "title": "Profit warning triggers sharp selloff", "pool": "train_unlabeled"},
                {"id": "u-neut", "title": "Board schedules annual shareholder meeting", "pool": "train_unlabeled"},
                {"id": "u-bull", "title": "Record quarterly revenue beats consensus", "pool": "train_unlabeled"},
                {"id": "rss-test", "title": "RSS test candidate headline", "pool": "test_candidate"},
            ]
            headlines_path.write_text("".join(json.dumps(r) + "\n" for r in raw_rows), encoding="utf-8")

            pseudo_rows = [
                {
                    "id": "u-bear",
                    "probs": [0.85, 0.10, 0.05],
                    "abc_mass": 0.999,
                    "top1_is_letter": True,
                    "letter_logp": [-0.16, -2.30, -3.00],
                },
                {
                    "id": "u-neut",
                    "probs": [0.05, 0.90, 0.05],
                    "abc_mass": 0.999,
                    "top1_is_letter": True,
                    "letter_logp": [-3.00, -0.10, -3.00],
                },
                {
                    "id": "u-bull",
                    "probs": [0.05, 0.15, 0.80],
                    "abc_mass": 0.999,
                    "top1_is_letter": True,
                    "letter_logp": [-3.00, -1.90, -0.22],
                },
            ]
            pseudo_path.write_text("".join(json.dumps(r) + "\n" for r in pseudo_rows), encoding="utf-8")

            # 1. Hard mode (E12) returns exact argmax (0=bearish, 1=neutral, 2=bullish)
            hard_ex, hard_stats = load_pseudo_labeled_examples(
                pseudo_path,
                headlines_path,
                pseudo_mode="hard",
                excluded_ids=set(),
                fns_split_ids=set(),
                expected_headlines_sha256=None,
            )
            self.assertEqual([x[0] for x in hard_ex], ["u-bear", "u-neut", "u-bull"])
            self.assertEqual([x[2] for x in hard_ex], [0, 1, 2])
            self.assertEqual(hard_stats["n_retained"], 3)
            self.assertEqual(hard_stats["hard_class_counts"], {"bearish": 1, "neutral": 1, "bullish": 1})

            # 2. Soft mode (E6) returns temperature-scaled target_T
            soft_ex, soft_stats = load_pseudo_labeled_examples(
                pseudo_path,
                headlines_path,
                pseudo_mode="soft",
                temperature=2.0,
                excluded_ids=set(),
                fns_split_ids=set(),
                expected_headlines_sha256=None,
            )
            self.assertEqual(len(soft_ex), 3)
            self.assertEqual(soft_stats["distill_temperature"], 2.0)
            for actual_p, exp_p in zip(soft_ex[0][3], softmax_temperature([-0.16, -2.30, -3.00], 2.0)):
                self.assertAlmostEqual(actual_p, exp_p, places=7)

            # 3. Guard: rejects ID in 330 excluded trial/test IDs
            with self.assertRaises(ValueError):
                load_pseudo_labeled_examples(
                    pseudo_path,
                    headlines_path,
                    pseudo_mode="hard",
                    excluded_ids={"u-bear"},
                    fns_split_ids=set(),
                    expected_headlines_sha256=None,
                )

            # 4. Guard: rejects ID belonging to FNS split
            with self.assertRaises(ValueError):
                load_pseudo_labeled_examples(
                    pseudo_path,
                    headlines_path,
                    pseudo_mode="hard",
                    excluded_ids=set(),
                    fns_split_ids={"u-neut"},
                    expected_headlines_sha256=None,
                )

            # 5. Guard: rejects duplicate IDs in pseudo_labels
            dup_pseudo = root / "dup_pseudo.jsonl"
            dup_pseudo.write_text(
                json.dumps(pseudo_rows[0]) + "\n" + json.dumps(pseudo_rows[0]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_pseudo_labeled_examples(
                    dup_pseudo,
                    headlines_path,
                    pseudo_mode="hard",
                    excluded_ids=set(),
                    fns_split_ids=set(),
                    expected_headlines_sha256=None,
                )

        # 6. If real data/raw/headlines_2026/headlines.jsonl is available locally, verify full 1,409 integration
        if DEFAULT_HEADLINES.is_file():
            fns_dummy = [("fns-1", "FNS train tweet", 1)]
            combined_hard, stats_hard = prepare_training_examples_with_pseudo(self.e12, fns_dummy, Path("."))
            self.assertEqual(stats_hard["n_retained"], 1409)
            self.assertEqual(stats_hard["hard_class_counts"], {"bearish": 81, "neutral": 1186, "bullish": 142})
            self.assertEqual(len(combined_hard), 1 + 1409)

            combined_soft, stats_soft = prepare_training_examples_with_pseudo(self.e6, fns_dummy, Path("."))
            self.assertEqual(stats_soft["n_retained"], 1409)
            self.assertEqual(len(combined_soft), 1 + 1409)
            self.assertFalse(combined_soft[0][4])  # FNS row has is_pseudo=False
            self.assertTrue(combined_soft[1][4])   # 2026 row has is_pseudo=True

    def test_format_e6_grid_markdown(self):
        rows = [
            {
                "temperature": 1.0,
                "alpha": 0.3,
                "best_epoch": 3,
                "macro_f1": 0.8010,
                "f1": {"bearish": 0.7300, "neutral": 0.8900, "bullish": 0.7830},
                "elapsed_seconds": 95.2,
                "peak_gpu_mem_gb": 1.65,
            },
            {
                "temperature": 2.0,
                "alpha": 0.5,
                "best_epoch": 4,
                "macro_f1": 0.8085,
                "f1": {"bearish": 0.7420, "neutral": 0.8950, "bullish": 0.7885},
                "elapsed_seconds": 96.1,
                "peak_gpu_mem_gb": 1.65,
            },
        ]
        md = format_e6_grid_markdown(rows, rows[1])
        self.assertIn("**`2.0`**", md)
        self.assertIn("**`0.5`**", md)
        self.assertIn("**`0.8085`**", md)

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
