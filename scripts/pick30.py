"""Select seeded trial IDs and URLs from a local 2026 candidate pool, without saving text."""
import argparse
import csv
import json
import random
from pathlib import Path


def select_candidates(rows, seed=2026, count=30):
    """Return a reproducible list of (ID, URL) from distinct test candidates."""
    candidates = {}
    seen = set()
    for row in rows:
        identifier = row['id']
        if identifier in seen:
            raise ValueError(f'Duplicate ID: {identifier}')
        seen.add(identifier)
        if row['pool'] == 'test_candidate':
            if not row.get('url'):
                raise ValueError(f'Missing URL for {identifier}')
            candidates[identifier] = row['url']
    if len(candidates) < count:
        raise ValueError(f'Need {count} candidates, found {len(candidates)}')
    return [(i, candidates[i]) for i in random.Random(seed).sample(sorted(candidates), count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/raw/headlines_2026/headlines.jsonl')
    parser.add_argument('--output', default='labeling/trial30_ids.csv')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    with open(args.input, encoding='utf-8') as f:
        selected = select_candidates((json.loads(line) for line in f if line.strip()), seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['id', 'url'])
        writer.writerows(selected)
    print(f'Selected {len(selected)} IDs; exclude them from the final test pool.')


if __name__ == '__main__':
    main()
