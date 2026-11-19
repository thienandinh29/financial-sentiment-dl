"""Tests for scripts/dedup.py: one duplicate pair, one non-duplicate pair, clustering."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dedup  # noqa: E402

DUP_A = "Apple shares rise after strong iPhone sales beat estimates"
DUP_B = "Apple shares rise after strong iPhone sales beat estimates!"   # punctuation only
ONE_WORD = "Apple shares rise after strong iPhone sales beat forecasts"   # Jaccard 0.79 >= 0.75
NEAR = "Apple shares rise after iPhone sales beat analyst forecasts"      # Jaccard 0.55 < 0.75
OTHER = "Oil prices fall as OPEC raises output"


class ShingleTest(unittest.TestCase):
    def test_unigrams_and_bigrams_ignore_case_and_punctuation(self):
        self.assertEqual(dedup.shingles("Oil, falls!"), {"oil", "falls", "oil falls"})

    def test_empty_texts_are_not_duplicates(self):
        self.assertEqual(dedup.jaccard(set(), set()), 0.0)


class LexicalTest(unittest.TestCase):
    def test_finds_duplicate_pair_and_rejects_unrelated_and_near_pairs(self):
        pairs = dedup.lexical_pairs([DUP_A, OTHER, DUP_B, NEAR])
        self.assertEqual(set(pairs), {(0, 2)})
        self.assertEqual(pairs[(0, 2)], 1.0)
        self.assertLess(dedup.jaccard(dedup.shingles(DUP_A), dedup.shingles(NEAR)), 0.75)

    def test_one_word_change_is_still_a_duplicate(self):
        self.assertEqual(set(dedup.lexical_pairs([DUP_A, ONE_WORD])), {(0, 1)})


class SemanticTest(unittest.TestCase):
    def test_threshold_on_normalized_vectors(self):
        vecs = np.array([[1, 0], [0.9, np.sqrt(1 - 0.81)], [0, 1]], dtype=np.float32)
        pairs = dedup.semantic_pairs(vecs, threshold=0.85, block=2)
        self.assertEqual(set(pairs), {(0, 1)})


class ClusterTest(unittest.TestCase):
    def test_connected_components_are_transitive(self):
        self.assertEqual(dedup.clusters(5, [(3, 4), (0, 3)]), [0, 1, 2, 0, 0])

    def test_run_merges_lexical_and_semantic_edges_and_writes_no_text(self):
        ids, texts = ["a", "b", "c", "d"], [DUP_A, OTHER, DUP_B, NEAR]
        vecs = np.eye(4, dtype=np.float32)
        vecs[3] = vecs[0]                       # pretend "d" is a paraphrase of "a"
        pairs, cluster_ids, summary = dedup.run(ids, texts, embeddings=vecs)
        self.assertEqual(cluster_ids, ["a", "b", "a", "a"])
        self.assertEqual(summary["pairs_lexical"], 1)
        self.assertEqual(summary["pairs_semantic"], 1)
        self.assertEqual(summary["items_in_dup_clusters"], 3)
        with tempfile.TemporaryDirectory() as tmp:
            dedup.write_outputs(tmp, ids, pairs, cluster_ids, summary)
            written = "".join(p.read_text(encoding="utf-8") for p in Path(tmp).iterdir())
        self.assertNotIn("Apple", written)

    def test_read_records_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            path.write_text("\n".join(json.dumps({"id": "x", "text": t}) for t in "ab"), encoding="utf-8")
            with self.assertRaises(ValueError):
                dedup.read_records(path)


if __name__ == "__main__":
    unittest.main()
