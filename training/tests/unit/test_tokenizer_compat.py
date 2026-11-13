"""Both Cookbook entry points retain explicit regex-repair metadata."""

from __future__ import annotations

import importlib.util
import inspect
import json
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import tokenizers
import transformers
from training import tokenizer_compat

backend_module = pytest.importorskip("transformers.tokenization_utils_tokenizers")
TokenizersBackend = backend_module.TokenizersBackend


def leaf(relative: str, name: str) -> Any:
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated_backport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TokenizersBackend, "__init__", inspect.unwrap(TokenizersBackend.__init__))
    patch = TokenizersBackend.__dict__["_patch_mistral_regex"].__func__
    monkeypatch.setattr(TokenizersBackend, "_patch_mistral_regex", classmethod(inspect.unwrap(patch)))


@pytest.mark.parametrize("reader", ["renderer", "runner"])
@pytest.mark.parametrize("fix", [None, False, True])
def test_actual_cookbook_loaders_preserve_flag_and_saved_readers(
    tmp_path: Path, reader: str, fix: bool | None
) -> None:
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    backend.save(str(tmp_path / "tokenizer.json"))
    # A literal WordLevel fixture is generic, not a Qwen2 BPE checkpoint.
    config: dict[str, Any] = {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}
    if fix is not None:
        config["fix_mistral_regex"] = fix
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    module = leaf("renderer/_tokenizer_utils.py" if reader == "renderer" else "utils/tokenizers.py", "compat_reader")
    load = module.get_tokenizer if reader == "renderer" else module.load_tokenizer
    loaded = load(str(tmp_path))
    assert type(loaded.backend_tokenizer.model).__name__ == "WordLevel"
    assert tokenizers.Tokenizer.from_file(str(tmp_path / "tokenizer.json")).encode("hello\nhello").ids == [1, 1]
    assert loaded.encode("hello\nhello", add_special_tokens=False) == [1, 1]
    if fix is not None:
        assert loaded.init_kwargs["fix_mistral_regex"] is fix
    saved = tmp_path / "saved"
    loaded.save_pretrained(saved)
    for restored in (
        load(str(saved)),
        transformers.AutoTokenizer.from_pretrained(saved, trust_remote_code=False, local_files_only=True),
    ):
        assert restored.encode("hello\nhello", add_special_tokens=False) == [1, 1]
        assert restored.all_special_ids == loaded.all_special_ids
        if fix is not None:
            assert restored.init_kwargs["fix_mistral_regex"] is fix


def test_backport_is_idempotent() -> None:
    tokenizer_compat.patch_duplicate_mistral_regex_kwarg()
    first = TokenizersBackend.__init__
    patch = TokenizersBackend.__dict__["_patch_mistral_regex"].__func__
    tokenizer_compat.patch_duplicate_mistral_regex_kwarg()
    assert TokenizersBackend.__init__ is first
    assert TokenizersBackend.__dict__["_patch_mistral_regex"].__func__ is patch


@pytest.mark.parametrize("reader", ["renderer", "runner"])
@pytest.mark.parametrize("precision", ["bf16", "nvfp4"])
def test_retained_supervl_data_ids_flag_and_reload(tmp_path: Path, reader: str, precision: str) -> None:
    fixture_root = os.environ.get("FIREWORKS_TOKENIZER_COMPAT_FIXTURES")
    if fixture_root is None:
        pytest.skip("set FIREWORKS_TOKENIZER_COMPAT_FIXTURES to the retained pinned data directory")
    source = Path(fixture_root) / f"nvidia-nvidia-nemotron-3-5-super-vl-120b-a12b-{precision}"
    assert hashlib.sha256((source / "tokenizer.json").read_bytes()).hexdigest() == (
        "d0432ad8b1c2984a2e2dca82e1933e3fd5628df6e95c51fba4c3ee9950dce299"
    )
    assert hashlib.sha256((source / "tokenizer_config.json").read_bytes()).hexdigest() == (
        "89fa70d31734c0c75ffdcee4114449f8cf8abd64bc233d420b58e25763f91aba"
    )
    oracle = tokenizers.Tokenizer.from_file(str(source / "tokenizer.json"))
    module = leaf("renderer/_tokenizer_utils.py" if reader == "renderer" else "utils/tokenizers.py", "real_reader")
    load = module.get_tokenizer if reader == "renderer" else module.load_tokenizer
    loaded = load(str(source))
    saved = tmp_path / "saved"
    loaded.save_pretrained(saved)
    for restored in (
        loaded,
        load(str(saved)),
        transformers.AutoTokenizer.from_pretrained(saved, trust_remote_code=False, local_files_only=True),
    ):
        assert restored.init_kwargs["fix_mistral_regex"] is True
        for text in ("hello世界🙂", "a\n\nb\tc", "x" * 25005, "<think>reason</think>"):
            assert restored.encode(text, add_special_tokens=False) == oracle.encode(text, add_special_tokens=False).ids
        assert restored.all_special_ids == loaded.all_special_ids
        assert restored.chat_template == loaded.chat_template
