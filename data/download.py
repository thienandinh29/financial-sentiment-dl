"""Download pinned FNS, PhraseBank AllAgree and GloVe 6B 100d; never split data."""

import argparse
import csv
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_metadata import run_metadata, sha256

FNS_LABELS = {"0": "bearish", "1": "bullish", "2": "neutral"}
PB_LABELS = {"negative": "bearish", "positive": "bullish", "neutral": "neutral"}
SOURCES = ROOT / "data/sources.json"
RAW = ROOT / "data/raw"


def write_jsonl(rows, path):
    """Write ordered rows as deterministic UTF-8 JSONL, replacing atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def fns_rows(path, split):
    """Convert source CSV to stable IDs and canonical labels, preserving order."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if not row["text"].strip():
                raise ValueError(f"Empty FNS text at row {index}")
            yield {"id": f"fns-{split}-{index:05d}", "text": row["text"],
                   "label": FNS_LABELS[row["label"]]}


def phrasebank_rows(path):
    """Read only AllAgree from the original latin-1 archive; map known labels."""
    with zipfile.ZipFile(path) as archive:
        lines = archive.read(
            "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
        ).decode("latin-1").splitlines()
    for index, line in enumerate(line for line in lines if line.strip()):
        text, label = line.rsplit("@", 1)
        if not text.strip():
            raise ValueError(f"Empty PhraseBank text at row {index}")
        yield {"id": f"pb-{index:05d}", "text": text.strip(),
               "label": PB_LABELS[label.strip()]}


def download_source(source):
    """Fetch files at an immutable HF revision into the ignored local cache."""
    from huggingface_hub import hf_hub_download

    return [Path(hf_hub_download(
        repo_id=source["repo_id"], repo_type=source["repo_type"],
        revision=source["revision"], filename=filename,
        cache_dir=ROOT / "data/cache/huggingface", token=False,
    )) for filename in source["files"]]


def prepare(name, paths):
    """Convert one downloaded source to its agreed local output file(s)."""
    if name == "fns":
        outputs = []
        for split, source in zip(("train", "validation"), paths):
            output = RAW / "fns" / f"{split}.jsonl"
            write_jsonl(fns_rows(source, split), output)
            outputs.append(output)
        return outputs
    if name == "phrasebank":
        output = RAW / "phrasebank/allagree.jsonl"
        write_jsonl(phrasebank_rows(paths[0]), output)
        return [output]
    output = RAW / "glove/glove.6B.100d.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with zipfile.ZipFile(paths[0]) as archive:
        with archive.open(output.name) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
    os.replace(temporary, output)
    return [output]


def main():
    """Download selected sources and record hashes/provenance under results/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=("fns", "phrasebank", "glove"),
                        default=["fns", "phrasebank", "glove"])
    args = parser.parse_args()
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    start = time.perf_counter()
    for name in args.only:
        source_start = time.perf_counter()
        print(f"Preparing {name} ...", flush=True)
        paths = download_source(sources[name])
        outputs = prepare(name, paths)
        metadata = run_metadata(source_start, {
            "source": sources[name], "source_manifest_sha256": sha256(SOURCES)
        }, ["huggingface_hub"])
        metadata["downloads"] = [
            {"filename": filename, "sha256": sha256(path), "bytes": path.stat().st_size}
            for filename, path in zip(sources[name]["files"], paths)
        ]
        metadata["outputs"] = [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
             "bytes": path.stat().st_size} for path in outputs
        ]
        report = ROOT / f"results/downloads/{name}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        # Retain every actual download run, including reproducibility checks.
        history = json.loads(report.read_text()) if report.exists() else []
        history.append(metadata)
        report.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared {name}: {len(outputs)} file(s)", flush=True)
    print(f"Done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
