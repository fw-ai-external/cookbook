"""Pinned GLM release loading and renderer contracts, without checkpoint code."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers.models.auto.image_processing_auto import AutoImageProcessor

from training import tokenizer_identity
from training.renderer import get_renderer
from training.renderer._image_processing_utils import get_image_processor
from training.renderer.tokenizer import get_tokenizer
from training.renderer.verifier.utils.tokenizer import load_image_processor
from training.utils.supervised import (
    build_tool_prefixed_messages,
    normalize_messages,
    populate_render_worker_state,
    resolve_renderer_name,
)
from training.utils.tokenizers import load_tokenizer

SOURCES = {
    "GLM-5": ("c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2", "glm5_release"),
    "GLM-5.3": ("ba32f8c19c8738e2c57c71798c601d477a7bf844", "glm53"),
    "GLM-5.3-Flash": ("46a050bec13002ad29ee795c16e61cf4fa3c4a9c", "glm53_flash"),
}


@pytest.fixture(scope="module", params=SOURCES)
def release(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path, Any]:
    name = request.param
    revision, _ = SOURCES[name]
    root = tmp_path_factory.mktemp(name)
    local = os.environ.get("GLM5_FIXTURE_ROOT")
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source = (
            Path(local) / name / filename
            if local
            else Path(hf_hub_download(f"zai-org/{name}", filename, revision=revision))
        )
        shutil.copyfile(source, root / filename)
    stock = AutoTokenizer.from_pretrained(root, trust_remote_code=False)
    return name, root, stock


def test_release_code_free_boundaries_and_saved_reload(
    release: tuple[str, Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, source, stock = release
    staged = tmp_path / "opaque"
    shutil.copytree(source, staged)
    marker = tmp_path / "executed"
    config = json.loads((staged / "tokenizer_config.json").read_text())
    config["auto_map"] = {"AutoTokenizer": [None, "canary_tokenizer.CanaryTokenizer"]}
    (staged / "tokenizer_config.json").write_text(json.dumps(config))
    (staged / "canary_tokenizer.py").write_text(
        "from pathlib import Path\n"
        "from transformers import PreTrainedTokenizerFast\n"
        "class CanaryTokenizer(PreTrainedTokenizerFast):\n"
        "    def __init__(self, *args, **kwargs):\n"
        f"        Path({str(marker)!r}).touch()\n"
        "        super().__init__(*args, **kwargs)\n"
    )
    AutoTokenizer.from_pretrained(staged, trust_remote_code=True, local_files_only=True)
    assert marker.exists(), "positive control must execute the synthetic canary"
    marker.unlink()
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    actual = load_tokenizer(str(staged), trust_remote_code=True, local_files_only=True)
    for text in ("Hello 世界", "<think>why</think>answer", "<|assistant|> literal"):
        assert actual.encode(text) == stock.encode(text)
    assert actual.chat_template == stock.chat_template
    assert not marker.exists()
    assert not tokenizer_identity.is_deepseek_v4_tokenizer_source(str(staged))
    model_config = json.loads((staged / "config.json").read_text())
    model_config["model_type"] = "qwen3"
    (staged / "config.json").write_text(json.dumps(model_config))
    with (staged / "tokenizer.json").open("ab") as stream:
        stream.write(b"\n")
    assert not tokenizer_identity.requires_code_free_tokenizer_loading(str(staged))
    assert load_tokenizer(str(staged), trust_remote_code=True, base_model=f"zai-org/{name}").encode(
        "Hello"
    ) == stock.encode("Hello")
    assert get_tokenizer(str(staged), renderer_name=SOURCES[name][1]).encode("Hello") == stock.encode("Hello")
    assert not marker.exists()
    processor_calls = []

    def processor(source: str, **kwargs: Any) -> Any:
        assert source == str(staged)
        processor_calls.append(kwargs.get("trust_remote_code"))
        return stock

    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", processor)
    assert get_image_processor(str(staged), base_model=f"zai-org/{name}") is stock
    assert load_image_processor(str(staged), renderer_name=SOURCES[name][1]) is stock
    assert processor_calls == [False, False]
    saved = tmp_path / "saved"
    stock.save_pretrained(saved)
    saved_config = json.loads((saved / "tokenizer_config.json").read_text())
    saved_config["auto_map"] = config["auto_map"]
    (saved / "tokenizer_config.json").write_text(json.dumps(saved_config))
    shutil.copyfile(staged / "canary_tokenizer.py", saved / "canary_tokenizer.py")
    assert not (saved / "config.json").exists()
    assert tokenizer_identity.requires_code_free_tokenizer_loading(str(saved))
    assert load_tokenizer(str(saved), trust_remote_code=True).encode("Hello 世界") == stock.encode("Hello 世界")
    assert not marker.exists()


def test_release_generation_and_sft_match_their_own_template(release: tuple[str, Path, Any]) -> None:
    name, _, tokenizer = release
    renderer_name = resolve_renderer_name(f"zai-org/{name}")
    assert renderer_name == SOURCES[name][1]
    renderer = get_renderer(renderer_name, tokenizer)
    for messages in (
        [{"role": "user", "content": "Hello"}],
        [
            {"role": "user", "content": "Add."},
            {"role": "assistant", "reasoning_content": "Compute.", "content": "Four."},
            {"role": "user", "content": "Explain."},
        ],
    ):
        expected_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        expected = tokenizer.encode(expected_text, add_special_tokens=False)
        assert renderer.build_generation_prompt(normalize_messages(messages)).to_ints() == expected
    messages = [{"role": "user", "content": "Add."}, {"role": "assistant", "content": "Four."}]
    expected_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    actual, weights = renderer.build_supervised_example(normalize_messages(messages))
    assert actual.to_ints()[:-1] == tokenizer.encode(expected_text, add_special_tokens=False)
    assert actual.to_ints()[-1] in renderer.get_stop_sequences()
    assert len(weights) == len(actual.to_ints())


@pytest.mark.parametrize("trust_remote_code", [None, True])
@pytest.mark.parametrize("renderer_name_is_resolved", [False, True])
def test_worker_preserves_glm_renderer_code_free_context(
    release: tuple[str, Path, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trust_remote_code: bool | None,
    renderer_name_is_resolved: bool,
) -> None:
    """Synthetic staged-checkpoint canary, not a customer or served-model reproduction."""
    name, source, stock = release
    staged = tmp_path / "opaque"
    shutil.copytree(source, staged)
    # Remove family metadata and change the byte fingerprint without changing
    # tokenization. The selected renderer must independently enforce policy.
    (staged / "config.json").unlink()
    with (staged / "tokenizer.json").open("ab") as stream:
        stream.write(b"\n")
    # The text-only worker setup needs a Flash counter, not pixel processing.
    # Supply synthetic default-counter metadata; image behavior is tested elsewhere.
    (staged / "preprocessor_config.json").write_text("{}")
    marker = tmp_path / "checkpoint-python-executed"
    config = json.loads((staged / "tokenizer_config.json").read_text())
    config["auto_map"] = {"AutoTokenizer": [None, "worker_canary.WorkerCanaryTokenizer"]}
    (staged / "tokenizer_config.json").write_text(json.dumps(config))
    (staged / "worker_canary.py").write_text(
        "from pathlib import Path\n"
        "from transformers import PreTrainedTokenizerFast\n"
        "class WorkerCanaryTokenizer(PreTrainedTokenizerFast):\n"
        "    def __init__(self, *args, **kwargs):\n"
        f"        Path({str(marker)!r}).touch()\n"
        "        super().__init__(*args, **kwargs)\n"
    )
    assert not tokenizer_identity.requires_code_free_tokenizer_loading(str(staged), local_files_only=True)
    AutoTokenizer.from_pretrained(staged, trust_remote_code=True, local_files_only=True)
    assert marker.exists(), "positive control must execute synthetic checkpoint Python"
    marker.unlink()
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "1")
    state: dict[str, Any] = {}
    populate_render_worker_state(
        state,
        tokenizer_model=str(staged),
        tokenizer_trust_remote_code=trust_remote_code,
        renderer_name=SOURCES[name][1],
        renderer_name_is_resolved=renderer_name_is_resolved,
        base_model="accounts/example/models/unrecognized-alias",
        max_seq_len=128,
    )
    assert not marker.exists()
    assert state["tokenizer"].encode("Hello 世界") == stock.encode("Hello 世界")
    assert state["max_seq_len"] == 128


def test_release_tool_schema_matches_its_own_template(release: tuple[str, Path, Any]) -> None:
    name, _, tokenizer = release
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Look up weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    messages = [{"role": "user", "content": "Weather?"}]
    renderer = get_renderer(SOURCES[name][1], tokenizer)
    prefixed = build_tool_prefixed_messages(messages, renderer=renderer, tools=tools)
    expected_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    assert renderer.build_generation_prompt(prefixed).to_ints() == tokenizer.encode(
        expected_text, add_special_tokens=False
    )
