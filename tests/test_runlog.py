"""Tests for train/runlog.py: the dirty flag reflects tracked changes only."""
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from train.runlog import run_metadata


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                   capture_output=True)


class RunlogTest(unittest.TestCase):
    def test_dirty_flag_ignores_untracked_files_but_not_tracked_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo, "init", "-q")
            (repo / "code.py").write_text("x = 1\n", encoding="utf-8")
            git(repo, "add", "code.py")
            git(repo, "commit", "-q", "-m", "init")
            now = datetime.now(timezone.utc)
            (repo / "notes.md").write_text("local notes\n", encoding="utf-8")          # untracked
            meta = run_metadata({"a": 1}, 1, now, cwd=repo)
            self.assertFalse(meta["working_tree_dirty"])
            self.assertEqual(len(meta["commit"]), 40)
            (repo / "code.py").write_text("x = 2\n", encoding="utf-8")                # tracked change
            self.assertTrue(run_metadata({"a": 1}, 1, now, cwd=repo)["working_tree_dirty"])


if __name__ == "__main__":
    unittest.main()
