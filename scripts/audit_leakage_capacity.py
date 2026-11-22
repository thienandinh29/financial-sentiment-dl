"""Audit the frozen split's duplicate-cluster capacity without accessing texts or labels."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts.run_metadata import sha256
from train.data import load_split


def cluster_capacity(splits, rows):
    """Return per-split ID and cluster bounds; reject missing, duplicate or mixed IDs."""
    membership = {identifier: name for name, ids in splits.items() for identifier in ids}
    if len(membership) != sum(map(len, splits.values())):
        raise ValueError('Duplicate ID in split')
    clusters = defaultdict(Counter)
    seen = set()
    for row in rows:
        identifier = row['id']
        if identifier not in membership or identifier in seen or not row['cluster_id']:
            raise ValueError('Unknown, duplicate, or invalid cluster ID')
        seen.add(identifier)
        clusters[row['cluster_id']][membership[identifier]] += 1
    if seen != set(membership):
        raise ValueError('Cluster ID coverage does not match split')
    mixed = sum(len(counts) > 1 for counts in clusters.values())
    result = {}
    for name, ids in splits.items():
        sizes = [counts[name] for counts in clusters.values() if counts[name]]
        capacity = sum(size - 1 for size in sizes)
        result[name] = {'items': len(ids), 'items_in_multi_member_clusters': sum(s for s in sizes if s > 1),
                        'cluster_capacity': capacity, 'cluster_capacity_fraction': capacity / len(ids) if ids else 0,
                        'targets': {str(percent): math.ceil(len(ids) * percent / 100) for percent in (5, 10, 20)}}
    return {'mixed_clusters': mixed, 'splits': result}


def main():
    """Validate input hashes and print an ID-only audit as JSON."""
    split_file = Path('data/splits/cluster_split.jsonl')
    cluster_file = Path('results/feasibility/fns-cosine092/clusters.csv')
    source_run = json.loads(Path('results/feasibility/fns-cosine092/run.json').read_text(encoding='utf-8'))
    expected_cluster_hash = source_run['output_sha256']['clusters.csv']
    if sha256(cluster_file) != expected_cluster_hash:
        raise ValueError('Cluster SHA-256 mismatch')
    splits = load_split(split_file, '8c2c35ea789acf5b2bd9c74c5e59fa0df325034422be42ae7b47f2e8c9c46aaa')
    with cluster_file.open(encoding='utf-8', newline='') as stream:
        result = cluster_capacity(splits, csv.DictReader(stream))
    result['input_sha256'] = {'split': sha256(split_file), 'clusters': expected_cluster_hash}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
