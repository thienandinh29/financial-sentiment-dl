"""Select and assign 300 candidate IDs without reading or writing headline text."""
import argparse
import csv
import json
import random
from pathlib import Path


def build_assignment(rows, excluded, seed=2026):
    """Return seeded (id, url) selection and (id, annotator, round) assignments."""
    candidates = {}
    seen = set()
    for row in rows:
        identifier = row['id']
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Missing or duplicate candidate ID')
        seen.add(identifier)
        if row['pool'] == 'test_candidate' and identifier not in excluded:
            if not row.get('url'):
                raise ValueError(f'Missing URL for {identifier}')
            candidates[identifier] = row['url']
    if len(candidates) < 300:
        raise ValueError(f'Need 300 candidates after exclusions; found {len(candidates)}')
    rng = random.Random(seed)
    ids = rng.sample(sorted(candidates), 300)
    selected = [(identifier, candidates[identifier]) for identifier in ids]
    assignments = []
    cursor = 0
    # 50 doubles and 105 singles in round 1, then 50 doubles and 95 singles.
    # This yields 69/68/68 assignments in round 1 and 65/66/64 in round 2.
    for round_no, pair_counts, single_counts in (
        (1, (17, 17, 16), (35, 35, 35)),
        (2, (17, 16, 17), (32, 32, 31)),
    ):
        for pair, count in zip(((1, 2), (1, 3), (2, 3)), pair_counts):
            for identifier in ids[cursor:cursor + count]:
                assignments.extend((identifier, f'tv{member}', round_no) for member in pair)
            cursor += count
        for member, count in enumerate(single_counts, 1):
            assignments.extend((identifier, f'tv{member}', round_no)
                               for identifier in ids[cursor:cursor + count])
            cursor += count
    return selected, assignments


def main():
    """Read the local candidate pool and trial IDs; write ID-only selection and assignments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/raw/headlines_2026/headlines.jsonl')
    parser.add_argument('--exclude', default='labeling/trial30_ids.csv')
    parser.add_argument('--selection', default='labeling/test300_ids.csv')
    parser.add_argument('--assignment', default='labeling/assignment.csv')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    outputs = [Path(args.selection), Path(args.assignment)]
    if outputs[0].resolve() == outputs[1].resolve() or any(p.exists() for p in outputs):
        raise FileExistsError('Output paths must be different and must not exist')
    with open(args.exclude, encoding='utf-8-sig', newline='') as stream:
        excluded_list = [row['id'] for row in csv.DictReader(stream)]
    if len(excluded_list) != 30 or len(set(excluded_list)) != 30:
        raise ValueError('Expected exactly 30 distinct trial IDs')
    with open(args.input, encoding='utf-8') as stream:
        selected, assignments = build_assignment(
            (json.loads(line) for line in stream if line.strip()), set(excluded_list), args.seed)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, header, rows in zip(outputs, (('id', 'url'), ('id', 'annotator', 'round')),
                                  (selected, assignments)):
        with path.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(header)
            writer.writerows(rows)
    print(f'Selected 300 IDs and {len(assignments)} assignments with seed {args.seed}.')


if __name__ == '__main__':
    main()
