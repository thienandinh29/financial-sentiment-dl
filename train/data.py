"""Load train/validation examples from a frozen, ID-only split file (docs/WEEK2.md contracts).

Split file: one JSON object per line, {"id": ..., "split": "train" | "validation" | "test"}.
Its SHA-256 must match the config before anything is read (AGENTS.md rule 2). Texts and labels
come from the raw JSONL files by ID. Test rows are never returned here: the three test sets are
evaluated once, in week 6, by a separate script (AGENTS.md rule 1).
"""
import hashlib
import json
from pathlib import Path

LABELS = ("bearish", "neutral", "bullish")          # shared order: index 0, 1, 2
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
SPLITS = ("train", "validation", "test")
DEFAULT_RAW = ("data/raw/fns/train.jsonl", "data/raw/fns/validation.jsonl")


def sha256_file(path):
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path, expected_sha256):
    """Read an ID-only split file after checking its SHA-256.

    Input: split path, expected hex digest. Output: dict split name -> list of ids (file order).
    Raises on a missing or wrong hash, an unknown split name, a duplicate id or a text field.
    """
    if not expected_sha256:
        raise ValueError(f"No split_sha256 given for {path}: refusing an unverified split")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Split SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    splits, seen = {name: [] for name in SPLITS}, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"id", "split"}:
                raise ValueError("Split rows must hold only 'id' and 'split' (no text)")
            if row["split"] not in splits:
                raise ValueError(f"Unknown split '{row['split']}' for id {row['id']}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate id in split file: {row['id']}")
            seen.add(row["id"])
            splits[row["split"]].append(row["id"])
    return splits


def load_raw(paths):
    """Read raw JSONL rows {"id", "text", "label"} into a dict id -> (text, label index)."""
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["id"]] = (row["text"], LABEL_TO_ID[row["label"]])
    return rows


def load_split_data(split_file, split_sha256, raw_files=DEFAULT_RAW, splits=("train", "validation")):
    """Return {split: [(id, text, label index), ...]} for train and/or validation only.

    Input: split file + expected hash, raw JSONL files, which splits to load.
    Output: examples in split-file order. Asking for 'test' raises (week 6 uses its own script).
    """
    if "test" in splits:
        raise ValueError("Test sets are evaluated once in week 6 by a separate script, not here")
    ids = load_split(split_file, split_sha256)
    raw = load_raw(raw_files)
    missing = [i for name in splits for i in ids[name] if i not in raw]
    if missing:
        raise ValueError(f"{len(missing)} split ids missing from raw files, e.g. {missing[:3]}")
    return {name: [(i, *raw[i]) for i in ids[name]] for name in splits}


def write_split(path, assignments):
    """Write an ID-only split file from (id, split) pairs and return its SHA-256 (for tests/tools)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, name in assignments:
            f.write(json.dumps({"id": i, "split": name}) + "\n")
    return sha256_file(path)
