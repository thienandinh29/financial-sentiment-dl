import unittest
from scripts.recluster_092 import recluster


class ReclusterTests(unittest.TestCase):
    def test_filter_preserves_lexical_edges_and_singletons(self):
        rows = [dict(id_a='a', id_b='b', jaccard='', cosine='0.9300'),
                dict(id_a='b', id_b='c', jaccard='', cosine='0.9100'),
                dict(id_a='c', id_b='d', jaccard='0.8000', cosine='')]
        kept, groups = recluster(['a', 'b', 'c', 'd', 'e'], rows)
        self.assertEqual(kept, [rows[0], rows[2]])
        self.assertEqual(groups, [['a', 'b'], ['c', 'd'], ['e']])

    def test_rounded_cutoff_and_missing_ids_are_rejected(self):
        row = dict(id_a='a', id_b='b', jaccard='', cosine='0.9200')
        with self.assertRaisesRegex(ValueError, 'Rounded cosine'):
            recluster(['a', 'b'], [row])
        kept, groups = recluster(['a', 'b'], [row], exact_scores={('a', 'b'): 0.92002})
        self.assertEqual(groups, [['a', 'b']])
        self.assertEqual(kept, [row])
        kept, groups = recluster(['a', 'b'], [row], exact_scores={('a', 'b'): 0.91998})
        self.assertEqual(groups, [['a'], ['b']])
        with self.assertRaisesRegex(ValueError, 'absent'):
            recluster(['a'], [row])
