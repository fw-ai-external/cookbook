"""Tokenizer inference without changing infrastructure or explicit overrides."""

from __future__ import annotations

import ast
import importlib
import json
import runpy
from pathlib import Path

import pytest

from training import model_defaults
from training.model_defaults import resolve_tokenizer_model

QWEN = "accounts/fireworks/models/qwen3-8b"


@pytest.mark.parametrize(
    ("model", "tokenizer"),
    [
        ("gemma-4-31b-it", "google/gemma-4-31B-it"),
        ("llama-v3p3-70b-instruct", "meta-llama/Llama-3.3-70B-Instruct"),
        ("gpt-oss-120b", "openai/gpt-oss-120b"),
    ],
)
def test_infers_models_beyond_the_serverless_examples(model: str, tokenizer: str) -> None:
    assert resolve_tokenizer_model(f"accounts/fireworks/models/{model}") == tokenizer


def test_ambiguous_model_requires_an_explicit_identity() -> None:
    model = "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"
    with pytest.raises(ValueError, match="Multiple tokenizer identities"):
        resolve_tokenizer_model(model)
    tokenizer = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
    assert resolve_tokenizer_model(model, tokenizer) == tokenizer


@pytest.mark.parametrize(
    "model",
    [
        "nexa-place2x2-short-hard-lora-r128-s7312",
        "exa-alpha-offset-r8-fix32-s5501",
        "nexa-alpha-offset-r8-fix32-s5501",
        "tb-lora-fullft-reliable-20260723-fullft-lr1e6-step-1",
    ],
)
def test_experimental_models_are_not_defaults(model: str) -> None:
    with pytest.raises(ValueError, match="--tokenizer-model"):
        resolve_tokenizer_model(f"accounts/fireworks/models/{model}")


def test_generated_defaults_match_the_owning_manifest() -> None:
    root = Path(__file__).resolve().parents[3]
    manifest = root.parents[1] / "control_plane/pkg/trainingrenderer/renderer_ids.json"
    if not manifest.exists():
        pytest.skip("The owning manifest is only present in the Fireworks source checkout")
    generator = runpy.run_path(str(root / "training/tests/_tokenizer_defaults_generator.py"))
    expected = generator["render_catalog"](json.loads(manifest.read_text()))
    source = Path(model_defaults.__file__).read_text()
    actual = source.split(generator["START"], 1)[1].split(generator["END"], 1)[0]
    assert actual == expected, "Regenerate tokenizer defaults from the owning manifest"


def test_generation_does_not_guess_between_conflicting_identities() -> None:
    root = Path(__file__).resolve().parents[3]
    generator = runpy.run_path(str(root / "training/tests/_tokenizer_defaults_generator.py"))
    manifest = {
        "version": 1,
        "source_account": "fireworks",
        "renderer_ids": [
            {
                "id": "provider/a",
                "fireworks_models": ["accounts/fireworks/models/shared", "accounts/fireworks/models/unique"],
            },
            {
                "id": "provider/b",
                "fireworks_models": ["accounts/fireworks/models/shared", "accounts/customer/models/custom"],
            },
        ],
    }
    rendered = generator["render_catalog"](manifest)
    assignments = {node.target.id: ast.literal_eval(node.value) for node in ast.parse(rendered).body}
    assert assignments["TOKENIZER_MODELS"] == {"accounts/fireworks/models/unique": "provider/a"}
    assert assignments["AMBIGUOUS_TOKENIZER_MODELS"] == {
        "accounts/fireworks/models/shared": ("provider/a", "provider/b")
    }


def test_tokenizer_never_guesses_a_customer_models_identity() -> None:
    assert resolve_tokenizer_model(QWEN) == "Qwen/Qwen3-8B"
    assert resolve_tokenizer_model(QWEN, "./reviewed-tokenizer") == "./reviewed-tokenizer"
    with pytest.raises(ValueError, match="--tokenizer-model"):
        resolve_tokenizer_model("accounts/customer/models/qwen3-8b")


@pytest.mark.parametrize("method", ["sft", "dpo"])
def test_examples_follow_base_model_instead_of_demo_tokenizer(method: str) -> None:
    module = importlib.import_module(f"training.examples.{method}.train_{method}_serverless")
    cfg = module.build_config(module.parse_args(["--base-model", QWEN]))
    assert cfg.tokenizer_model == "Qwen/Qwen3-8B"


@pytest.mark.parametrize("method", ["sft", "dpo"])
def test_recipe_infers_tokenizer_before_provisioning(
    method: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = importlib.import_module(f"training.recipes.{method}_loop")
    monkeypatch.setattr(recipe, "validate_config", lambda *args, **kwargs: None)

    def stop_before_provisioning(**kwargs: object) -> str:
        assert kwargs["tokenizer_model"] == "Qwen/Qwen3-8B"
        raise RuntimeError("stop before provisioning")

    resolver = "_resolved_renderer_name" if method == "sft" else "resolve_renderer_snapshot"
    monkeypatch.setattr(recipe, resolver, stop_before_provisioning)
    cfg = recipe.Config(base_model=QWEN, tokenizer_model="", log_path=str(tmp_path))
    with pytest.raises(RuntimeError, match="stop before provisioning"):
        recipe.main(cfg)
