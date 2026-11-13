"""Exact renderer selection; no checkpoint Python or model weights."""

import json
from pathlib import Path

import pytest

from training.renderer.model_selection import resolve_exact_model_renderer
from training.utils.supervised import resolve_renderer_name
from training.utils.tokenizers import needs_mistral_regex_fix


@pytest.mark.parametrize(
    ("model", "renderer"),
    [
        ("moonshotai/Kimi-K3", "kimi_k3"),
        ("moonshotai/Kimi-K3.1-Pro", "kimi_k31"),
        ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek_v41"),
        ("zai-org/GLM-5.3", "glm53"),
        ("zai-org/GLM-5", "glm5_release"),
        ("zai-org/GLM-5.3-BF16", "glm53"),
        ("zai-org/GLM-5.3-Flash-BF16", "glm53_flash"),
        ("Qwen/Qwen3.8-27B", "qwen3_8"),
        ("MiniMaxAI/MiniMax-M3", "minimax_m3"),
        ("google/gemma-4-31B-it", "gemma4"),
        ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "nemotron35_vl"),
    ],
)
def test_full_identity_not_basename_selects_renderer(model: str, renderer: str) -> None:
    assert resolve_renderer_name(model) == renderer
    assert resolve_renderer_name(f"https://huggingface.co/{model}") == renderer
    basename = model.rsplit("/", 1)[-1]
    for unreviewed in (f"custom/{basename}", f"{model}-unreviewed", f"/models/{basename}", f"{model}/hf"):
        assert resolve_exact_model_renderer(unreviewed) is None
        with pytest.raises(ValueError, match="Set Config.renderer_name explicitly"):
            resolve_renderer_name(unreviewed)
        assert resolve_renderer_name(unreviewed, renderer) == renderer


@pytest.mark.parametrize(
    "dirname",
    ["opaque", "kimi_k3", "Kimi-K3.1-Pro", "DeepSeek-V4.1-Flash", "GLM-5.3", "MiniMax-M3", "Qwen3.8-27B"],
)
def test_local_renaming_cannot_select_a_renderer(tmp_path: Path, dirname: str) -> None:
    directory = tmp_path / dirname
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": "unrelated"}))
    with pytest.raises(ValueError, match="Set Config.renderer_name explicitly"):
        resolve_renderer_name(str(directory))
    assert resolve_renderer_name(str(directory), "kimi_k3") == "kimi_k3"


@pytest.mark.parametrize("dirname", ["opaque", "mistral", "ministral-3-3b", "Qwen3-8B"])
def test_mistral_tokenization_flag_follows_metadata_not_directory(tmp_path: Path, dirname: str) -> None:
    root = tmp_path / dirname
    root.mkdir()
    assert not needs_mistral_regex_fix(str(root))
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert not needs_mistral_regex_fix(str(root))
    (root / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    assert needs_mistral_regex_fix(str(root))
    (root / "tokenizer_config.json").write_text(json.dumps({"fix_mistral_regex": False}))
    assert not needs_mistral_regex_fix(str(root))
    assert not needs_mistral_regex_fix("custom/mistral-small-unreviewed")
    assert needs_mistral_regex_fix("mistralai/Mistral-Small-24B-Instruct-2501")
