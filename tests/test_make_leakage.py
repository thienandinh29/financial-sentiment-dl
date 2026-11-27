import unittest
from scripts.make_leakage import build_manifests

class LeakageTests(unittest.TestCase):
    def test_nested_direct_exposure_and_full_validation(self):
        split = {"train": ["t"], "validation": list("abcdef"), "test": ["x"]}
        clusters = {i: i for i in "tabcdefx"}
        clusters.update({"a": "a", "b": "a", "c": "a"})
        pairs = [("a", "b"), ("b", "c")]
        first, second = build_manifests(split, clusters, pairs, [2, 5], 1)
        self.assertEqual(first["validation_evaluation_ids"], list("abcdef"))
        self.assertTrue(set(first["train_add_validation_ids"]) <= set(second["train_add_validation_ids"]))
        self.assertTrue(set(first["validation_exposed_ids"]) <= set(second["validation_exposed_ids"]))
        for result in (first, second):
            donors = set(result["train_add_validation_ids"])
            direct = donors | {b for a, b in pairs if a in donors} | {a for a, b in pairs if b in donors}
            self.assertEqual(set(result["validation_exposed_ids"]), direct)
            self.assertEqual(set(result["validation_near_only_ids"]), direct - donors)
            self.assertEqual(len(result["validation_clean_ids"]) + len(direct), 6)
        self.assertEqual(build_manifests(split, clusters, pairs, [2, 5], 1), [first, second])

    def test_reject_preexisting_leakage_and_bad_targets(self):
        split = {"train": ["t"], "validation": ["a", "b"], "test": ["x"]}
        clusters = {i: i for i in "tabx"}
        with self.assertRaises(ValueError):
            build_manifests(split, clusters, [("t", "a")], [1])
        with self.assertRaises(ValueError):
            build_manifests(split, clusters, [], [3])
