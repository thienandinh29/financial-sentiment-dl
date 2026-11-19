"""Derive 0.92-threshold ID-only clusters from the preserved 0.85 pair artifact."""
import csv
import importlib.metadata
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from train.data import DEFAULT_RAW, load_raw, sha256_file
from train.runlog import run_metadata


def recluster(ids, pairs, threshold=0.92, exact_scores=None):
    """Return retained edges and connected components for the complete input ID set."""
    parent = {identifier: identifier for identifier in ids}
    if len(parent) != len(ids):
        raise ValueError('Duplicate cluster ID')

    def find(identifier):
        if identifier not in parent:
            raise ValueError(f'Pair ID absent from clusters: {identifier}')
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    kept = []
    for row in pairs:
        a, b = find(row['id_a']), find(row['id_b'])
        cosine = float(row['cosine']) if row['cosine'] else None
        # Four-decimal rounding at the cutoff makes the unrounded value unknowable.
        if not row['jaccard'] and cosine is not None and abs(cosine - threshold) < 0.00005:
            key = (row['id_a'], row['id_b'])
            if exact_scores is None or key not in exact_scores:
                raise ValueError('Rounded cosine at threshold; recompute original similarities')
            cosine = exact_scores[key]
            if round(cosine, 4) != float(row['cosine']) or abs(cosine - threshold) < 0.000001:
                raise ValueError('Recomputed score differs from source or is too close to cutoff')
        if row['jaccard'] or (cosine is not None and cosine >= threshold):
            kept.append(row)
            parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for identifier in sorted(ids):
        groups[find(identifier)].append(identifier)
    return kept, sorted(groups.values(), key=lambda group: group[0])


def recompute_boundary_scores(pairs, metadata):
    """Recompute ambiguous cosine scores using the pinned source model on CPU."""
    boundary = [row for row in pairs if not row['jaccard'] and row['cosine']
                and abs(float(row['cosine']) - 0.92) < 0.00005]
    if not boundary:
        return {}
    import numpy as np
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    config = metadata['config']
    model_path = snapshot_download(config['model'], revision=config['model_revision'], token=False,
                                   cache_dir='data/cache/huggingface',
                                   allow_patterns=['config.json', 'config_sentence_transformers.json',
                                                   'modules.json', 'sentence_bert_config.json',
                                                   'special_tokens_map.json', 'tokenizer.json',
                                                   'tokenizer_config.json', 'vocab.txt',
                                                   'model.safetensors', '1_Pooling/config.json'])
    raw = load_raw(DEFAULT_RAW)
    model = SentenceTransformer(model_path, device='cpu')
    identifiers = sorted({row[key] for row in boundary for key in ('id_a', 'id_b')})
    embeddings = model.encode([raw[i][0] for i in identifiers], batch_size=256,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False).astype('float32')
    lookup = dict(zip(identifiers, embeddings))
    return {(row['id_a'], row['id_b']): float(np.dot(lookup[row['id_a']], lookup[row['id_b']]))
            for row in boundary}


def main():
    """Write separate, reproducible artifacts and provenance without overwriting prior runs."""
    start = datetime.now(timezone.utc)
    source = Path('results/feasibility/fns')
    original_run = Path('results/feasibility/run.json')
    output = Path('results/feasibility/fns-cosine092')
    if output.exists():
        raise FileExistsError(f'{output} already exists')
    metadata = json.loads(original_run.read_text(encoding='utf-8'))
    if metadata['config']['cosine'] != 0.85 or metadata['config']['jaccard'] != 0.75:
        raise ValueError('Unexpected source thresholds')
    for filename in DEFAULT_RAW:
        if sha256_file(filename) != metadata['config']['inputs_sha256'][filename]:
            raise ValueError(f'Raw data hash does not match original run: {filename}')
    with (source / 'clusters.csv').open(encoding='utf-8', newline='') as stream:
        ids = [row['id'] for row in csv.DictReader(stream)]
    with (source / 'pairs.csv').open(encoding='utf-8', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    exact_scores = recompute_boundary_scores(pairs, metadata)
    kept, groups = recluster(ids, pairs, exact_scores=exact_scores)
    output.mkdir(parents=True)
    for name, fields, rows in (
        ('pairs.csv', ['id_a', 'id_b', 'jaccard', 'cosine'], kept),
        ('clusters.csv', ['id', 'cluster_id'],
         [{'id': identifier, 'cluster_id': group[0]} for group in groups for identifier in group]),
    ):
        with (output / name).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    config = {'jaccard': 0.75, 'cosine': 0.92, 'source_run': str(original_run),
              'source_sha256': {str(path): sha256_file(path) for path in
                                (original_run, source / 'pairs.csv', source / 'clusters.csv')},
              'boundary_scores': [{'id_a': a, 'id_b': b, 'cosine': score}
                                  for (a, b), score in sorted(exact_scores.items())],
              'model': metadata['config']['model'], 'model_revision': metadata['config']['model_revision'],
              'note': 'Filtered saved 0.85 edges; boundary scores recomputed on CPU with pinned model. Original candidate recall limitations remain.'}
    info = run_metadata(config, 1, start)
    for package in ('sentence-transformers', 'transformers', 'huggingface-hub'):
        info['libraries'][package] = importlib.metadata.version(package) if exact_scores else None
    info.update({'items': len(ids), 'pairs': len(kept), 'clusters': len(groups),
                 'output_sha256': {name: sha256_file(output / name) for name in ('pairs.csv', 'clusters.csv')}})
    (output / 'run.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(f"Wrote {len(kept)} pairs and {len(groups)} clusters for {len(ids)} IDs")


if __name__ == '__main__':
    main()
