"""Run the complete Cookbook CPU suite in three ordered, whole-module processes."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest

ROOTS = ("tests/unit", "tests/test_smoke_imports.py", "examples/rl/frozen_lake/test_masking.py")
BOUNDARY = "tests/unit/test_renderer_properties.py"
OUTPUT = "COOKBOOK_COLLECTION_OUTPUT"
EXPECTED = "COOKBOOK_COLLECTION_EXPECTED"


def pytest_collection_finish(session: pytest.Session) -> None:
    """Check each real execution's collection before any tests can run."""
    nodeids = [item.nodeid for item in session.items]
    Path(os.environ[OUTPUT]).write_text(json.dumps(nodeids) + "\n")
    if len(nodeids) != len(set(nodeids)):
        pytest.exit("Cookbook collection contains duplicate node IDs", returncode=1)
    if expected := os.environ.get(EXPECTED):
        if nodeids != json.loads(Path(expected).read_text()):
            pytest.exit("Cookbook batch collection differs from its ordered manifest", returncode=1)


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(Path(__file__).resolve().parent), env.get("PYTHONPATH"))))
    env.pop(EXPECTED, None)
    pytest_command = ["pytest", "-q", "-p", Path(__file__).stem]
    with tempfile.TemporaryDirectory(prefix="cookbook-unit-batches-") as directory:
        work = Path(directory)
        env[OUTPUT] = str(work / "full.json")
        subprocess.run([*pytest_command, "--collect-only", "--quiet", "--no-cov", *ROOTS], env=env, check=True)
        nodeids = json.loads(Path(env[OUTPUT]).read_text())
        modules = [
            (name, list(items)) for name, items in itertools.groupby(nodeids, key=lambda node: node.split("::", 1)[0])
        ]
        names = [name for name, _ in modules]
        if len(names) != len(set(names)) or names.count(BOUNDARY) != 1:
            raise RuntimeError("Cookbook collection must contain one contiguous block per module")
        boundary = names.index(BOUNDARY)
        batches = (modules[:boundary], modules[boundary : boundary + 1], modules[boundary + 1 :])
        if not all(batches):
            raise RuntimeError("Cookbook requires three nonempty test batches")
        expected_batches = [[node for _, items in batch for node in items] for batch in batches]
        if list(itertools.chain.from_iterable(expected_batches)) != nodeids:
            raise RuntimeError("Cookbook batch partition changed the complete ordered collection")
        subprocess.run(["coverage", "erase"], env=env, check=True)
        for index, (batch, expected) in enumerate(zip(batches, expected_batches), start=1):
            expected_path = work / f"expected-{index}.json"
            expected_path.write_text(json.dumps(expected) + "\n")
            env[EXPECTED] = str(expected_path)
            env[OUTPUT] = str(work / f"actual-{index}.json")
            reports = (
                ["--cov-report="]
                if index < 3
                else ["--cov-report=term-missing", "--cov-report=xml:coverage.xml", "--cov-report=json:coverage.json"]
            )
            print(f"Cookbook batch {index}/3: {len(expected)} tests in {len(batch)} modules", flush=True)
            subprocess.run(
                [*pytest_command, *(name for name, _ in batch), "--cov=.", "--cov-append", *reports],
                env=env,
                check=True,
            )
        actual = [node for index in range(1, 4) for node in json.loads((work / f"actual-{index}.json").read_text())]
        if actual != nodeids:
            raise RuntimeError("Executed Cookbook batches do not match the complete ordered collection")
        print(f"Execution batches matched all {len(nodeids)} Cookbook tests exactly once in order", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"Cookbook subprocess failed with return code {error.returncode}", flush=True)
        raise SystemExit(error.returncode)
