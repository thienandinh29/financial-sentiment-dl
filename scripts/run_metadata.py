"""Small, text-free provenance helpers for TV1 data preparation runs."""

import hashlib
import importlib.metadata
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    """Return the SHA-256 hex digest of a file, reading bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata(start, config, packages=()):
    """Return Git, configuration and runtime provenance without reading secrets."""
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()

    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "config": config,
        "seed": 1,
        "python": platform.python_version(),
        "libraries": versions,
        "gpu": "not used (CPU data preparation)",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
