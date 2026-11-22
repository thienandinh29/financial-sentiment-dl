import unittest
from scripts.audit_leakage_capacity import cluster_capacity


class CapacityTests(unittest.TestCase):
    def test_capacity_and_mixed_clusters(self):
        splits = {'train': ['a', 'b'], 'validation': [], 'test': ['c', 'd', 'e']}
        rows = [{'id': i, 'cluster_id': c} for i, c in [('a', 'x'), ('b', 'y'), ('c', 'x'), ('d', 'z'), ('e', 'z')]]
        result = cluster_capacity(splits, rows)
        self.assertEqual(result['mixed_clusters'], 1)
        self.assertEqual(result['splits']['test']['cluster_capacity'], 1)
        self.assertEqual(result['splits']['test']['items_in_multi_member_clusters'], 2)

    def test_missing_or_duplicate_ids_rejected(self):
        splits = {'test': ['a', 'b']}
        with self.assertRaises(ValueError):
            cluster_capacity(splits, [{'id': 'a', 'cluster_id': 'x'}])
        with self.assertRaises(ValueError):
            cluster_capacity(splits, [{'id': 'a', 'cluster_id': 'x'}] * 2)


if __name__ == '__main__':
    unittest.main()
