"""Inspect DeepSWE source identity and task content without selecting an agent."""

from __future__ import annotations

import subprocess
from pathlib import Path

from training.examples.harbor_rl.records import tree_sha256


def _git_value(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def inspect_deep_swe(source_repository: Path) -> dict[str, object]:
    """Return a source manifest for a clean, materialized DeepSWE checkout."""
    repository = source_repository.expanduser().resolve()
    task_root = repository / "tasks"
    if not (repository / ".git").exists() or not task_root.is_dir():
        raise ValueError("source must be a DeepSWE Git checkout containing tasks/")
    if _git_value(repository, "status", "--porcelain"):
        raise ValueError("DeepSWE source checkout must be clean")
    tasks = sorted(
        path for path in task_root.iterdir() if (path / "task.toml").is_file()
    )
    if not tasks:
        raise ValueError("DeepSWE source contains no Harbor tasks")
    return {
        "schema_version": 1,
        "dataset": "deep-swe",
        "source": {
            "remote": _git_value(repository, "remote", "get-url", "origin"),
            "revision": _git_value(repository, "rev-parse", "HEAD"),
        },
        "task_count": len(tasks),
        "tasks": [path.name for path in tasks],
        "source_sha256": {path.name: tree_sha256(path) for path in tasks},
    }
