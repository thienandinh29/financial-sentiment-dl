"""Prepare an ID-only E10 validation leakage manifest; never access test labels or text.

The group must first decide the design, donor IDs and requested number of exposed
validation examples. The fixed split and its validation/test files are not changed.
A downstream trainer must explicitly use the donor IDs and exclude them from the
research evaluation; the standard training pipeline does not yet consume this file.
"""
import argparse
import csv
import json
from pathlib import Path

from train.data import load_split, sha256_file


def make_manifest(splits, pairs, donors, requested_count):
    """Build an ID-only train overlay and disjoint validation evaluation manifest.

    Inputs: verified split ID lists, direct near-duplicate ID pairs, explicit validation
    donor IDs, and the exact desired number of exposed evaluation IDs. Output: a dict
    containing donor IDs, exposed IDs and all evaluation IDs. No raw text is read.
    """
    validation = list(splits['validation'])
    valid_set = set(validation)
    donors = list(donors)
    donor_set = set(donors)
    if not validation or len(valid_set) != len(validation):
        raise ValueError('Validation IDs must be nonempty and unique')
    if not donors or len(donor_set) != len(donors) or not donor_set <= valid_set:
        raise ValueError('Donors must be unique validation IDs, with at least one donor')
    if not isinstance(requested_count, int) or isinstance(requested_count, bool) or requested_count < 0:
        raise ValueError('requested_count must be a nonnegative integer')
    remaining = valid_set - donor_set
    exposed = set()
    for a, b in pairs:
        if a in donor_set and b in remaining:
            exposed.add(b)
        if b in donor_set and a in remaining:
            exposed.add(a)
    if len(exposed) != requested_count:
        raise ValueError(f'Donors expose {len(exposed)} validation IDs, not requested {requested_count}')
    evaluation = [i for i in validation if i not in donor_set]
    if not evaluation:
        raise ValueError('At least one validation ID must remain for evaluation')
    return {'train_add_validation_ids': sorted(donor_set),
            'validation_evaluation_ids': evaluation,
            'validation_exposed_ids': [i for i in evaluation if i in exposed],
            'n_frozen_validation': len(validation), 'n_evaluation': len(evaluation),
            'n_exposed': len(exposed), 'fraction_of_frozen_validation': len(exposed) / len(validation),
            'fraction_of_evaluation': len(exposed) / len(evaluation)}


def main():
    """Validate hashes and explicit donors, then exclusively create a JSON manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', required=True)
    parser.add_argument('--split-sha256', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--pairs-sha256', required=True)
    parser.add_argument('--donors', required=True, help='JSON array of explicit validation donor IDs')
    parser.add_argument('--requested-count', required=True, type=int)
    parser.add_argument('--output', required=True, help='New manifest file; never a split or input file')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    protected = [Path(args.split).resolve(), Path(args.pairs).resolve(), Path(args.donors).resolve()]
    if output in protected or output.exists() or output.suffix != '.json':
        raise ValueError('Output must be a new JSON file, distinct from all inputs')
    split = load_split(args.split, args.split_sha256)
    if sha256_file(args.pairs) != args.pairs_sha256:
        raise ValueError('Near-duplicate pairs SHA-256 mismatch')
    with open(args.donors, encoding='utf-8') as stream:
        donors = json.load(stream)
    if not isinstance(donors, list) or not all(isinstance(i, str) for i in donors):
        raise ValueError('Donors must be a JSON array of strings')
    with open(args.pairs, encoding='utf-8-sig', newline='') as stream:
        pairs = [(r['id_a'], r['id_b']) for r in csv.DictReader(stream)]
    result = make_manifest(split, pairs, donors, args.requested_count)
    result['input_sha256'] = {'split': args.split_sha256, 'pairs': args.pairs_sha256,
                              'donors': sha256_file(args.donors)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
