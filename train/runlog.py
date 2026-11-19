"""Run metadata for results/<E>/seed<k>/run.json (AGENTS.md: commit, config, seed, versions, GPU, time)."""
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

PACKAGES = ("torch", "numpy", "scikit-learn", "pyyaml")


def _git(*args, cwd="."):
    """Run a git command; return its output, or None outside a repository."""
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(config, seed, started_at, cwd="."):
    """Return provenance for one run; no secrets or environment variables are read.

    working_tree_dirty counts changes to tracked files only (like `git describe --dirty`), so stray
    untracked files such as local notes do not mark a run as dirty.
    """
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=cwd)
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "commit": _git("rev-parse", "HEAD", cwd=cwd),
        "working_tree_dirty": bool(status) if status is not None else None,
        "config": config,
        "seed": seed,
        "python": platform.python_version(),
        "libraries": versions,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round((datetime.now(timezone.utc) - started_at).total_seconds(), 3),
    }


def write_json(path, data):
    """Write data as indented UTF-8 JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
