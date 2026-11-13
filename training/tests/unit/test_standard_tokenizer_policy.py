"""Independent restrictive context; no Hub, renderer plugins or checkpoint code."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import tokenizers
import transformers

from training import tokenizer_identity as identity

CONTEXTS = [
    "Qwen/Qwen3.8-27B",
    "google/gemma-4-31B-it",
    "MiniMaxAI/MiniMax-M3",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    "accounts/fireworks/models/llama-v3p1-nemotron-70b-instruct",
]


@pytest.mark.parametrize("backend", ["sdk", "snapshot"])
@pytest.mark.parametrize(
    "renderer_name",
    ["qwen3_5_disable_thinking", "qwen3_6_preserved", "qwen3_8_interleaved", "gemma4_thinking", "nemotron35_vl"],
)
def test_explicit_renderer_variants_keep_opaque_tokenizer_code_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, renderer_name: str
) -> None:
    # lazy: validate both shipped backends against the actual direct/worker loaders.
    from fireworks.training.sdk import tokenizer_identity as sdk
    from training import _tokenizer_identity_snapshot as snapshot

    monkeypatch.setattr(identity, "_identity", sdk if backend == "sdk" else snapshot)
    monkeypatch.chdir(tmp_path)
    (tmp_path / renderer_name).mkdir()
    assert not identity.is_standard_tokenizer_renderer_name(f"custom/{renderer_name}")
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "true")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: calls.append(k) or object())
    renderer = leaf("renderer/_tokenizer_utils.py", "variant_renderer")
    renderer.get_tokenizer(str(tmp_path), renderer_name=renderer_name)
    assert calls[0]["trust_remote_code"] is False
    worker = importlib.import_module("training.utils.supervised")
    monkeypatch.setattr(worker, "build_renderer_from_resolved_name", lambda *a, **k: object())
    worker.populate_render_worker_state(
        {},
        tokenizer_model=str(tmp_path),
        renderer_name=renderer_name,
        renderer_name_is_resolved=True,
        tokenizer_trust_remote_code=True,
        max_seq_len=128,
    )
    assert len(calls) == 2 and all(call["trust_remote_code"] is False for call in calls)


def leaf(relative: str, name: str) -> Any:
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("policy", [None, True, False])
def test_runner_context_cannot_reopen_k2_override(
    monkeypatch: pytest.MonkeyPatch, context: str, policy: bool | None
) -> None:
    runner = leaf("utils/tokenizers.py", "policy_runner")
    monkeypatch.setattr(runner, "is_kimi_k3_k31_tokenizer_source", lambda *a, **k: False)
    monkeypatch.setattr(runner, "requires_code_free_tokenizer_loading", lambda *a, **k: False)
    monkeypatch.setattr(runner, "_read_repo_json", lambda *a, **k: None)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: calls.append(k) or object())
    runner.load_tokenizer("moonshotai/Kimi-K2.6", trust_remote_code=policy, base_model=context)
    assert len(calls) == 1 and calls[0]["trust_remote_code"] is False


@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("local_override", [False, True])
def test_renderer_context_blocks_ambient_and_direct_k2_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: str, local_override: bool
) -> None:
    renderer = leaf("renderer/_tokenizer_utils.py", "policy_renderer")
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "true")
    monkeypatch.setattr(renderer, "is_kimi_k3_k31_tokenizer_source", lambda *a, **k: False)
    monkeypatch.setattr(renderer, "requires_code_free_tokenizer_loading", lambda *a, **k: False)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: calls.append(k) or object())

    def reject(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("legacy K2 checkpoint class must not be imported")

    monkeypatch.setattr(transformers.dynamic_module_utils, "get_class_from_dynamic_module", reject)
    renderer.get_tokenizer(str(tmp_path) if local_override else "moonshotai/Kimi-K2.6", base_model=context)
    assert len(calls) == 1 and calls[0]["trust_remote_code"] is False


def test_renderer_unrelated_context_preserves_legacy_k2(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = leaf("renderer/_tokenizer_utils.py", "legacy_renderer")
    monkeypatch.setattr(renderer, "is_kimi_k3_k31_tokenizer_source", lambda *a, **k: False)
    monkeypatch.setattr(renderer, "requires_code_free_tokenizer_loading", lambda *a, **k: False)
    calls: list[tuple[Any, ...]] = []
    sentinel = object()

    class LegacyTokenizer:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> object:
            calls.append(args)
            return sentinel

    monkeypatch.setattr(
        transformers.dynamic_module_utils, "get_class_from_dynamic_module", lambda *a, **k: LegacyTokenizer
    )
    assert renderer.get_tokenizer("moonshotai/Kimi-K2.6", base_model="acme/unrelated") is sentinel
    assert calls == [("moonshotai/Kimi-K2.6",)]


@pytest.mark.parametrize("override", ["MiniMaxAI/MiniMax-M2.5", "acme/opaque", "moonshotai/Kimi-K2.6", "local"])
def test_renderer_without_optional_backend_keeps_generic_false_and_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    # Resolve the current lazy module after other test modules finish collecting.
    transformers = importlib.import_module("transformers")
    renderer = leaf("renderer/_tokenizer_utils.py", "missing_backend_renderer")
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "true")
    monkeypatch.setattr(renderer, "is_kimi_k3_k31_tokenizer_source", lambda *a, **k: False)
    monkeypatch.setattr(renderer, "requires_code_free_tokenizer_loading", lambda *a, **k: False)
    # Simulate an older installed runtime without this optional exported capability.
    monkeypatch.setattr(transformers, "TokenizersBackend", None)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sentinel = object()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: calls.append((a, k)) or sentinel
    )

    def reject(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("missing-backend fallback must not add metadata reads or checkpoint code")

    monkeypatch.setattr(transformers.models.auto.tokenization_auto, "get_tokenizer_config", reject)
    monkeypatch.setattr(transformers.dynamic_module_utils, "get_class_from_dynamic_module", reject)
    source = str(tmp_path) if override == "local" else override
    assert renderer.get_tokenizer(source, base_model="MiniMaxAI/MiniMax-M2.5") is sentinel
    assert calls == [((source,), {"trust_remote_code": False})]


@pytest.mark.parametrize("reader", ["renderer", "runner"])
@pytest.mark.parametrize("context", [None, "MiniMaxAI/MiniMax-M2.5"])
@pytest.mark.parametrize("add_bos,add_eos", [(False, False), (True, False), (False, True), (True, True)])
def test_minimax_literal_json_and_saved_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str, context: str | None, add_bos: bool, add_eos: bool
) -> None:
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "minimax_m2"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "GPT2Tokenizer",
                "auto_map": {"AutoTokenizer": ["poison.Bad", None]},
                "bos_token": "<s>",
                "eos_token": "</s>",
                "add_bos_token": add_bos,
                "add_eos_token": add_eos,
            }
        )
    )
    (tmp_path / "poison.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "true")
    module = leaf("renderer/_tokenizer_utils.py" if reader == "renderer" else "utils/tokenizers.py", "json_reader")
    load = module.get_tokenizer if reader == "renderer" else module.load_tokenizer
    loaded = load(str(tmp_path), base_model=context)
    expected = ([1] if add_bos else []) + [3, 4] + ([2] if add_eos else [])
    assert loaded.encode("a\n\nb") == expected
    saved = tmp_path / "saved"
    loaded.save_pretrained(saved)
    assert "auto_map" not in json.loads((saved / "tokenizer_config.json").read_text())
    assert load(str(saved), base_model=context).encode("a\n\nb") == expected
    assert transformers.AutoTokenizer.from_pretrained(saved, trust_remote_code=False).encode("a\n\nb") == expected


@pytest.mark.parametrize(
    "model_type,context",
    [("minimax_m2", None), ("gpt2", "MiniMaxAI/MiniMax-M2.5"), (None, "MiniMaxAI/MiniMax-M2.5")],
)
@pytest.mark.parametrize("serialized_specials", [False, True])
def test_runner_minimax_without_optional_metadata_keeps_serialized_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str | None,
    context: str | None,
    serialized_specials: bool,
) -> None:
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    if serialized_specials:
        backend.post_processor = tokenizers.processors.TemplateProcessing(
            single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
        )
    backend.save(str(tmp_path / "tokenizer.json"))
    if model_type is not None:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "poison.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "true")
    current_transformers = importlib.import_module("transformers")
    generic_load = current_transformers.AutoTokenizer.from_pretrained

    def reject(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("MiniMax JSON must not fall through to AutoTokenizer or checkpoint code")

    monkeypatch.setattr(current_transformers.AutoTokenizer, "from_pretrained", reject)
    monkeypatch.setattr(current_transformers.dynamic_module_utils, "get_class_from_dynamic_module", reject)
    runner = leaf("utils/tokenizers.py", "missing_metadata_runner")
    loaded = runner.load_tokenizer(str(tmp_path), base_model=context, local_files_only=True)
    assert loaded.encode("a\n\nb") == backend.encode("a\n\nb").ids
    assert loaded.encode("a\n\nb", add_special_tokens=False) == [3, 4]
    saved = tmp_path / "saved"
    loaded.save_pretrained(saved)
    # Saved generic readers need no MiniMax identity; checkpoint-code imports remain blocked.
    monkeypatch.setattr(current_transformers.AutoTokenizer, "from_pretrained", generic_load)
    assert runner.load_tokenizer(str(saved), base_model=context).encode("a\n\nb") == backend.encode("a\n\nb").ids


def test_standard_architecture_context_is_retained() -> None:
    runner = leaf("utils/tokenizers.py", "context_runner")
    assert runner.resolve_tokenizer_base_model("acme/custom", "qwen3_5") == "qwen3_5"
    assert identity.is_standard_tokenizer_model_name("qwen3_5")


@pytest.mark.parametrize("backend", ["sdk", "snapshot"])
def test_standard_policy_matches_both_identity_backends(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    # lazy: exercise older-SDK fallback without replacing the installed SDK.
    from fireworks.training.sdk import tokenizer_identity as sdk
    from training import _tokenizer_identity_snapshot as snapshot

    monkeypatch.setattr(identity, "_identity", sdk if backend == "sdk" else snapshot)
    for context in CONTEXTS:
        assert identity.is_standard_tokenizer_model_name(context)
    assert not identity.is_standard_tokenizer_model_name("acme/unrelated")
    assert identity.is_minimax_m2_model_name("MiniMaxAI/MiniMax-M2.5")
    assert not identity.is_minimax_m2_model_name("acme/MiniMax-M2.5")


@pytest.mark.parametrize("backend", ["sdk", "snapshot"])
def test_standard_policy_does_not_infer_directory_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str
) -> None:
    # lazy: verify the standalone fallback as well as the installed SDK.
    from fireworks.training.sdk import tokenizer_identity as sdk
    from training import _tokenizer_identity_snapshot as snapshot

    monkeypatch.setattr(identity, "_identity", sdk if backend == "sdk" else snapshot)
    for context in CONTEXTS:
        assert identity.is_standard_tokenizer_model_name(context)
        assert not identity.is_standard_tokenizer_model_name(f"acme/{context.rsplit('/', 1)[-1]}")
    root = tmp_path / "Qwen3-8B"
    root.mkdir()
    assert not identity.is_standard_tokenizer_model_name(str(root))
    (root / "config.json").write_text(json.dumps({"model_type": "minimax_m2"}))
    assert identity.is_standard_tokenizer_model_name(str(root))


@pytest.mark.parametrize("backend", ["sdk", "snapshot"])
def test_local_minimax_context_collision_in_both_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str
) -> None:
    # lazy: exercise the independently shipped fallback as well as SDK delegation.
    from fireworks.training.sdk import tokenizer_identity as sdk
    from training import _tokenizer_identity_snapshot as snapshot

    monkeypatch.setattr(identity, "_identity", sdk if backend == "sdk" else snapshot)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "MiniMaxAI/MiniMax-M2.5"
    root.mkdir(parents=True)
    for model_type in ("qwen3", "minimax_m2"):
        (root / "config.json").write_text(json.dumps({"model_type": model_type}))
        for source in ("MiniMaxAI/MiniMax-M2.5", str(root)):
            assert identity.is_minimax_m2_model_name(source) is (model_type == "minimax_m2")
            assert identity.is_minimax_m2_tokenizer_source(source) is (model_type == "minimax_m2")


@pytest.mark.parametrize("reader", ["renderer", "runner"])
@pytest.mark.parametrize("model", ["moonshotai/Kimi-K3", "moonshotai/Kimi-K3.1-Pro"])
def test_selected_k3_context_keeps_native_dispatch(monkeypatch: pytest.MonkeyPatch, reader: str, model: str) -> None:
    native = leaf("utils/kimi_native_tokenizer.py", "training.utils.kimi_native_tokenizer")
    monkeypatch.setitem(sys.modules, "training.utils.kimi_native_tokenizer", native)
    sentinel = object()
    monkeypatch.setattr(native, "load_native_tokenizer", lambda *a, **k: sentinel)
    module = leaf("renderer/_tokenizer_utils.py" if reader == "renderer" else "utils/tokenizers.py", "native_reader")
    load = module.get_tokenizer if reader == "renderer" else module.load_tokenizer
    assert load("Qwen/Qwen3-8B", base_model=model) is sentinel
