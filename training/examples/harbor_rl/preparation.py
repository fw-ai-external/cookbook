"""Runner-owned composition of Harbor task sources and harness installers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from training.examples.harbor_rl.records import tree_sha256, write_json


def prepare_harbor_tasks(
    source: Path,
    destination: Path,
    manifest_path: Path,
    *,
    prepare_tasks: Callable[[Path, Path], list[Path]],
    harness: Mapping[str, str],
    source_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Apply the runner's installer and record source/output content separately.

    Bind version and installer options with ``functools.partial``. Dataset
    identity can be supplied by a dataset inspector; neither dataset nor
    harness names are resolved here.
    """
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    tasks = (
        [source]
        if (source / "task.toml").is_file()
        else sorted(path for path in source.iterdir() if (path / "task.toml").is_file())
    )
    if not tasks:
        raise ValueError("source contains no Harbor tasks")
    names = [path.name for path in tasks]
    hashed_roots = [*tasks, *(destination / name for name in names)]
    if any(manifest_path.is_relative_to(path.resolve()) for path in hashed_roots):
        raise ValueError(
            "manifest must be outside source and prepared task directories"
        )
    hashes = {path.name: tree_sha256(path) for path in tasks}
    manifest: dict[str, object] = {
        "schema_version": 1,
        "task_count": len(tasks),
        "tasks": names,
        "source_sha256": hashes,
    }
    if source_manifest is not None:
        for key in ("task_count", "tasks", "source_sha256"):
            if source_manifest.get(key) != manifest[key]:
                raise ValueError(f"source manifest {key} differs from the task source")
        manifest.update(source_manifest)
    prepared = prepare_tasks(source, destination)
    actual = sorted(
        path.name for path in destination.iterdir() if (path / "task.toml").is_file()
    )
    if [path.name for path in prepared] != names or actual != names:
        raise RuntimeError("prepared Harbor membership differs from the source")
    if {path.name: tree_sha256(path) for path in tasks} != hashes:
        raise RuntimeError("harness preparation changed the source tasks")
    manifest.update(
        harness=dict(harness),
        prepared_sha256={name: tree_sha256(destination / name) for name in names},
    )
    write_json(manifest_path, manifest)
    return manifest
