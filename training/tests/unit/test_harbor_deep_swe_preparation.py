from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from functools import partial

import pytest

from training.examples.harbor_rl import train_opencode
from training.examples.harbor_rl.harnesses.opencode.constants import (
    DEFAULT_OPENCODE_VERSION,
)
from training.examples.harbor_rl.harnesses.opencode import (
    prepare_tasks as prepare_opencode,
)
from training.examples.harbor_rl.harnesses.pi import prepare_tasks as prepare_pi
from training.examples.harbor_rl.harnesses.mini_swe import (
    prepare_tasks as prepare_mini_swe,
)
from training.examples.harbor_rl.preparation import prepare_harbor_tasks
from training.examples.harbor_rl.tasks.deep_swe.dataset import inspect_deep_swe
from training.examples.harbor_rl.records import tree_sha256
from training.examples.harbor_rl.tasks.deep_swe import prepare_tasks


@pytest.fixture
def source_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Host URL rewrites must not change the fixture's expected source remote.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repository = tmp_path / "deep-swe"
    for name in ("task-b", "task-a"):
        task = repository / "tasks" / name
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "task.toml").write_text(
            '[environment]\ndocker_image = "example:task"\n'
        )
        (task / "environment" / "Dockerfile").write_text(
            "FROM ubuntu:24.04\nUSER agent\n"
        )
        (task / "tests" / "test.sh").write_text("#!/bin/sh\ntest -f /app/model.py\n")
        (task / "instruction.md").write_text(f"Fix {name}.\n")
    subprocess.run(["git", "init", "-q", repository], check=True)
    subprocess.run(["git", "-C", repository, "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "remote",
            "add",
            "origin",
            "https://github.com/datacurve-ai/deep-swe.git",
        ],
        check=True,
    )
    return repository


@pytest.mark.parametrize("harness", ["opencode", "pi", "mini_swe"])
def test_runner_composes_installers_without_dataset_dispatch(
    source_repository: Path, tmp_path: Path, harness: str
) -> None:
    destination = tmp_path / "prepared"
    manifest_path = tmp_path / "manifest.json"
    snapshot = inspect_deep_swe(source_repository)
    installers = {
        "opencode": partial(prepare_opencode.prepare, version=DEFAULT_OPENCODE_VERSION),
        "pi": partial(prepare_pi.prepare, version="0.84.1"),
        "mini_swe": prepare_mini_swe.prepare,
    }
    metadata = {"name": harness}
    if harness == "pi":
        metadata["version"] = "0.84.1"
    elif harness == "opencode":
        metadata["version"] = DEFAULT_OPENCODE_VERSION
    manifest = prepare_harbor_tasks(
        source_repository / "tasks",
        destination,
        manifest_path,
        prepare_tasks=installers[harness],
        harness=metadata,
        source_manifest=snapshot,
    )
    assert json.loads(manifest_path.read_text()) == manifest
    assert manifest["harness"] == metadata
    assert "harness" not in snapshot
    assert manifest["source"] == snapshot["source"]
    assert manifest["tasks"] == ["task-a", "task-b"]
    assert manifest["source_sha256"] == snapshot["source_sha256"]
    for name in manifest["tasks"]:
        source = source_repository / "tasks" / name
        prepared = destination / name
        dockerfile = (prepared / "environment" / "Dockerfile").read_text()
        if harness == "pi":
            assert "@earendil-works/pi-coding-agent@0.84.1" in dockerfile
            assert "opencode-ai" not in dockerfile
        elif harness == "opencode":
            assert f"opencode-ai@{DEFAULT_OPENCODE_VERSION}" in dockerfile
            assert "pi-coding-agent" not in dockerfile
        else:
            assert "fireworks TITO harbor.mini_swe.prepare_tasks" in dockerfile
            assert (
                "opencode-ai" not in dockerfile and "pi-coding-agent" not in dockerfile
            )
        assert dockerfile.rstrip().endswith("USER agent")
        for relative in ["tests/test.sh", "instruction.md"]:
            assert (prepared / relative).read_bytes() == (
                source / relative
            ).read_bytes()
        assert tree_sha256(source) == snapshot["source_sha256"][name]
        assert manifest["prepared_sha256"][name] == tree_sha256(prepared)


@pytest.mark.parametrize("harness", ["opencode", "pi"])
@pytest.mark.parametrize("with_manifest", [False, True])
def test_installer_cli_manifest_is_optional(
    source_repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harness: str,
    with_manifest: bool,
) -> None:
    destination = tmp_path / "prepared"
    manifest_path = tmp_path / "manifest.json"
    args = [
        "prepare_tasks",
        "--source",
        str(source_repository / "tasks"),
        "--destination",
        str(destination),
        f"--{harness}-version",
        "1.2.3",
    ]
    if with_manifest:
        args += ["--manifest", str(manifest_path)]
    monkeypatch.setattr(sys, "argv", args)
    installer = prepare_pi if harness == "pi" else prepare_opencode
    installer.main()
    assert manifest_path.exists() == with_manifest
    if with_manifest:
        manifest = json.loads(manifest_path.read_text())
        assert manifest["harness"] == {"name": harness, "version": "1.2.3"}
        assert manifest["task_count"] == 2


def test_legacy_opencode_wrapper_preserves_old_manifest(
    source_repository: Path,
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    with pytest.warns(DeprecationWarning, match="runner's selected installer"):
        manifest = prepare_tasks.prepare_deep_swe(
            source_repository,
            tmp_path / "prepared",
            path,
            opencode_version=DEFAULT_OPENCODE_VERSION,
        )
    assert json.loads(path.read_text()) == manifest
    assert manifest["opencode_version"] == DEFAULT_OPENCODE_VERSION
    assert (
        manifest["source"]["remote"] == "https://github.com/datacurve-ai/deep-swe.git"
    )


def test_source_drift_is_rejected_before_preparation(
    source_repository: Path,
    tmp_path: Path,
) -> None:
    snapshot = inspect_deep_swe(source_repository)
    (source_repository / "tasks/task-a/instruction.md").write_text("Changed task.")
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="source manifest source_sha256"):
        prepare_harbor_tasks(
            source_repository / "tasks",
            destination,
            tmp_path / "manifest.json",
            prepare_tasks=prepare_mini_swe.prepare,
            harness={"name": "mini_swe"},
            source_manifest=snapshot,
        )
    assert not destination.exists()


def test_missing_prepared_member_does_not_publish_manifest(
    source_repository: Path,
    tmp_path: Path,
) -> None:
    def omit_task(source: Path, destination: Path) -> list[Path]:
        return prepare_mini_swe.prepare(source / "task-a", destination)

    manifest = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError, match="membership differs"):
        prepare_harbor_tasks(
            source_repository / "tasks",
            tmp_path / "prepared",
            manifest,
            prepare_tasks=omit_task,
            harness={"name": "mini_swe"},
        )
    assert not manifest.exists()


def test_dataset_inspection_rejects_dirty_checkout(source_repository: Path) -> None:
    (source_repository / "tasks/task-a/instruction.md").write_text("Dirty source")
    with pytest.raises(ValueError, match="checkout must be clean"):
        inspect_deep_swe(source_repository)


@pytest.mark.parametrize("location", ["source", "prepared"])
def test_manifest_cannot_change_hashed_task_content(
    source_repository: Path,
    tmp_path: Path,
    location: str,
) -> None:
    destination = tmp_path / "prepared"
    task_root = source_repository / "tasks"
    manifest = (
        task_root if location == "source" else destination
    ) / "task-a/manifest.json"
    with pytest.raises(ValueError, match="outside source and prepared task"):
        prepare_harbor_tasks(
            task_root,
            destination,
            manifest,
            prepare_tasks=prepare_mini_swe.prepare,
            harness={"name": "mini_swe"},
        )
    assert not destination.exists()
    assert not manifest.exists()


@pytest.mark.parametrize("prepare_first", [False, True])
def test_training_runner_prepares_only_when_requested(
    source_repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepare_first: bool,
) -> None:
    run_dir = tmp_path / "run"
    argv = [
        "--base-model",
        "accounts/example/models/policy",
        "--tokenizer-model",
        "tokenizer",
        "--renderer-name",
        "renderer",
        "--harbor-dataset",
        str(source_repository / "tasks"),
        "--run-dir",
        str(run_dir),
        "--sampling-only",
        "--opencode-version",
        "1.2.3",
    ]
    if prepare_first:
        argv.append("--prepare-harbor-tasks")
    args = train_opencode.parse_args(argv)
    monkeypatch.setattr(train_opencode, "parse_args", lambda: args)
    loaded = []

    def load(dataset: str, **_kwargs: object) -> list[dict[str, str]]:
        loaded.append(dataset)
        return [{"task_name": "task-a"}]

    monkeypatch.setattr(train_opencode, "load_harbor_rows", load)
    monkeypatch.setattr(train_opencode, "rollout_fn_factory", lambda *_args: None)
    monkeypatch.setattr(
        train_opencode.common, "run_sampling_only", lambda *_args, **_kwargs: None
    )
    train_opencode.run()
    assert loaded == [
        str(run_dir / "prepared-tasks") if prepare_first else args.harbor_dataset
    ]
    assert (run_dir / "preparation.json").exists() == prepare_first
    if prepare_first:
        manifest = json.loads((run_dir / "preparation.json").read_text())
        assert manifest["harness"] == {"name": "opencode", "version": "1.2.3"}
        assert (
            "opencode-ai@1.2.3"
            in (run_dir / "prepared-tasks/task-a/environment/Dockerfile").read_text()
        )
