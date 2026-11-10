"""Exercise shell-wrapper arguments through the real Python config builders."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("method", ["sft", "dpo"])
@pytest.mark.parametrize(
    ("base_model", "tokenizer", "extra_args", "expected"),
    [
        ("kimi-k3", None, [], "moonshotai/Kimi-K3"),
        ("qwen3p8-27b", None, [], "Qwen/Qwen3.8-27B"),
        ("qwen3p8-27b", "", [], "Qwen/Qwen3.8-27B"),
        ("kimi-k3", None, ["--base-model", "accounts/fireworks/models/gemma-4-31b-it"], "google/gemma-4-31B-it"),
        ("qwen3p8-27b", "org/custom", [], "org/custom"),
        ("qwen3p8-27b", "org/custom", ["--tokenizer-model", "org/override"], "org/override"),
    ],
)
def test_wrapper_resolves_tokenizer_from_final_arguments(
    method: str,
    base_model: str,
    tokenizer: str | None,
    extra_args: list[str],
    expected: str,
    tmp_path: Path,
) -> None:
    capture = tmp_path / "args.json"
    python = tmp_path / "python"
    # Intercept both data prep and training; never download data or create GPUs.
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['WRAPPER_ARGS_PATH'], 'w') as f:\n"
        "    json.dump(sys.argv[1:], f)\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "FIREWORKS_API_KEY": "fw-test-key",
        "BASE_MODEL": f"accounts/fireworks/models/{base_model}",
        "WRAPPER_ARGS_PATH": str(capture),
    }
    env.pop("TOKENIZER_MODEL", None)
    if tokenizer is not None:
        env["TOKENIZER_MODEL"] = tokenizer
    script = Path(__file__).resolve().parents[2] / "examples" / method / "run_serverless.sh"
    subprocess.run(["bash", str(script), *extra_args], env=env, capture_output=True, text=True, check=True)
    argv = json.loads(capture.read_text())
    assert argv[:2] == ["-m", f"examples.{method}.train_{method}_serverless"]
    module = importlib.import_module(f"training.examples.{method}.train_{method}_serverless")
    cfg = module.build_config(module.parse_args(argv[2:]))
    assert cfg.tokenizer_model == expected
