"""Report human-reviewed duplicate precision by pair type; never assign review answers."""
import argparse
import csv
from collections import Counter
from pathlib import Path


def read_review(path):
    """Read review rows from a comma- or semicolon-delimited CSV file."""
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        header = f.readline()
        delimiter = ';' if header.count(';') > header.count(',') else ','
        f.seek(0)
        return list(csv.DictReader(f, delimiter=delimiter))


def summarize(rows, per_type=50):
    """Validate a completed review and return counts and proportions per pair type."""
    counts = {kind: Counter() for kind in ('lexical', 'semantic_only')}
    seen = set()
    for row in rows:
        kind = row['type']
        answer = row['is_duplicate'].strip().lower()
        pair = tuple(sorted((row['id_a'].strip(), row['id_b'].strip())))
        if kind not in counts or answer not in ('yes', 'no'):
            raise ValueError('Every row needs a known pair type and a human yes/no answer')
        if not all(pair) or pair[0] == pair[1] or pair in seen:
            raise ValueError('Invalid or repeated pair')
        seen.add(pair)
        counts[kind][answer] += 1
    for kind, answers in counts.items():
        if sum(answers.values()) != per_type:
            raise ValueError(f'{kind}: expected {per_type} reviews')
    return {kind: {'yes': answers['yes'], 'total': per_type,
                   'precision': answers['yes'] / per_type}
            for kind, answers in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', nargs='?', default='results/feasibility/manual_check_100.csv')
    args = parser.parse_args()
    result = summarize(read_review(args.review))
    for kind, stats in result.items():
        print(f"{kind}: {stats['yes']}/{stats['total']} ({stats['precision']:.1%})")


if __name__ == '__main__':
    main()
