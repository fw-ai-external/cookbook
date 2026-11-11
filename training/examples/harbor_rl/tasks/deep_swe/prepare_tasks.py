"""Compatibility entry point for the original DeepSWE/OpenCode preparation."""

from __future__ import annotations

import argparse
import warnings
from functools import partial
from pathlib import Path

from training.examples.harbor_rl.harnesses.opencode.constants import (
    DEFAULT_OPENCODE_VERSION,
)
from training.examples.harbor_rl.harnesses.opencode.prepare_tasks import prepare
from training.examples.harbor_rl.preparation import prepare_harbor_tasks
from training.examples.harbor_rl.records import write_json
from training.examples.harbor_rl.tasks.deep_swe.dataset import inspect_deep_swe


def prepare_deep_swe(
    source_repository: Path,
    destination: Path,
    manifest_path: Path,
    *,
    opencode_version: str,
) -> dict[str, object]:
    """Preserve the old API while runners migrate to explicit composition."""
    warnings.warn(
        "prepare_deep_swe is the legacy OpenCode wrapper; use inspect_deep_swe "
        "and prepare_harbor_tasks with the runner's selected installer",
        DeprecationWarning,
        stacklevel=2,
    )
    manifest = prepare_harbor_tasks(
        source_repository.expanduser().resolve() / "tasks",
        destination,
        manifest_path,
        prepare_tasks=partial(prepare, version=opencode_version),
        harness={"name": "opencode", "version": opencode_version},
        source_manifest=inspect_deep_swe(source_repository),
    )
    manifest["opencode_version"] = opencode_version
    write_json(manifest_path.expanduser().resolve(), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repository", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    args = parser.parse_args()
    manifest = prepare_deep_swe(
        args.source_repository,
        args.destination,
        args.manifest,
        opencode_version=args.opencode_version,
    )
    print(
        f"prepared {manifest['task_count']} DeepSWE tasks at "
        f"{args.destination} from {manifest['source']['revision']}"
    )


if __name__ == "__main__":
    main()
