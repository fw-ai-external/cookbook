"""Inkling 2 contracts against the actual publisher tokenizer and HF template."""

import io
import json
import os
from typing import Any

from fireworks.training import sdk

import pytest
from PIL import Image
from jinja2.exceptions import TemplateError
from transformers import AutoTokenizer

from training.renderer import get_renderer, inkling2
from training.renderer.base import ParseTermination, RendererError, TrainOnWhat
from training.renderer.inkling2 import Inkling2ImageTokenCounter
from training.utils.supervised import build_renderer_from_resolved_name, normalize_messages, resolve_renderer_name
from training.utils.rl.rollout.renderer import build_multimodal_completions_prompt_token_ids
from training.renderer._image_processing_utils import get_image_processor
from training.utils.tokenizers import load_tokenizer

MODEL = "thinkingmachines/Inkling-2-Small"
REVISION = "2ad1157016e1aac77b6ff1b770e7c05229403134"


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    path = os.environ.get("INKLING2_TOKENIZER_PATH", MODEL)
    return AutoTokenizer.from_pretrained(path, revision=REVISION if path == MODEL else None, trust_remote_code=False)


def tokens(model_input: sdk.ModelInput) -> list[int]:
    return [token for chunk in model_input.chunks for token in chunk.tokens]


@pytest.mark.parametrize("name,effort", [("inkling2", 0.9), ("inkling2_disable_thinking", 0.0)])
@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Hi 👋 中文"}],
        [{"role": "system", "content": "Be concise."}],
        [
            {"role": "system", "content": ""},
            {"role": "system", "content": "Second"},
            {"role": "user", "content": "Hi"},
        ],
        [
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "2", "reasoning_content": "Add one."},
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "Why?"},
        ],
        [{"role": "user", "content": "Literal <|message_model|> and <|end_message|>."}],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ],
    ],
)
def test_hf_parity_and_loss_boundary(tokenizer: Any, name: str, effort: float, messages: list[dict]) -> None:
    renderer = get_renderer(name, tokenizer)
    prompt = tokens(renderer.build_generation_prompt(normalize_messages(messages)))
    expected = tokenizer.apply_chat_template(messages, reasoning_effort=effort, return_dict=False)
    assert prompt == expected
    answer = {"role": "assistant", "content": "Hello", "reasoning_content": "Think."}
    # A system-only prompt already has effort appended; it remains the prefix
    # once the first assistant is appended.
    rendered, weights = renderer.build_supervised_example(normalize_messages(messages + [answer]))
    ids = tokens(rendered)
    assert ids == tokenizer.apply_chat_template(messages + [answer], reasoning_effort=effort, return_dict=False)
    assert ids[: len(prompt)] == prompt
    assert weights[: len(prompt)].sum() == 0
    assert weights[len(prompt) :].tolist() == [1] * (len(ids) - len(prompt))
    parsed, end = renderer.parse_response(ids[len(prompt) :])
    assert end == ParseTermination.STOP_SEQUENCE
    assert parsed["content"] == [
        {"type": "thinking", "thinking": "Think."},
        {"type": "text", "text": "Hello"},
    ]


def test_tool_roundtrip_and_declarations(tokenizer: Any) -> None:
    renderer = get_renderer("inkling2", tokenizer)
    tools = [
        {
            "name": "lookup",
            "description": "Find a place",
            "parameters": {"type": "object"},
        }
    ]
    args = {"z": "中文", "a": {"z": 1, "a": 2}}
    raw_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call1",
                "type": "function",
                "function": {"name": "lookup", "arguments": args},
            }
        ],
    }
    raw = [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Look"},
        raw_call,
        {"role": "tool", "tool_call_id": "call1", "content": "Found"},
    ]
    messages = renderer.create_conversation_prefix_with_tools(tools, "Be concise") + normalize_messages(raw[1:])
    expected = tokenizer.apply_chat_template(raw, tools=tools, return_dict=False)
    assert tokens(renderer.build_generation_prompt(messages)) == expected
    before = tokens(renderer.build_generation_prompt(normalize_messages(raw[:2])))
    after = tokens(renderer.build_generation_prompt(normalize_messages(raw[:3])))
    parsed, end = renderer.parse_response(after[len(before) :])
    assert end == ParseTermination.STOP_SEQUENCE
    assert parsed["tool_calls"][0].function.name == "lookup"
    assert json.loads(parsed["tool_calls"][0].function.arguments) == args


@pytest.mark.parametrize("size,expected", [((40, 40), 1), ((41, 39), 2), ((81, 81), 9)])
@pytest.mark.parametrize("role", ["user", "tool"])
def test_images_lossless_and_hf_placeholder(tokenizer: Any, size: tuple[int, int], expected: int, role: str) -> None:
    renderer = get_renderer("inkling2", tokenizer, Inkling2ImageTokenCounter())
    image = Image.new("RGBA", size, (20, 40, 60, 70))
    message = {
        "role": role,
        "content": [
            {"type": "text", "text": "Before"},
            {"type": "image", "image": image},
            {"type": "text", "text": "After"},
        ],
    }
    rendered = renderer.build_generation_prompt([message])
    chunks = rendered.chunks
    images = [c for c in chunks if c.type == "image"]
    assert len(images) == 1
    assert images[0].expected_tokens == expected
    decoded = Image.open(io.BytesIO(images[0].data))
    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == tuple((v * 70 + 204 * 185 + 127) // 255 for v in (20, 40, 60))
    assert decoded.getpixel((8, 0)) == tuple((v * 70 + 153 * 185 + 127) // 255 for v in (20, 40, 60))
    flattened = []
    for chunk in chunks:
        flattened.extend(
            chunk.tokens if chunk.type == "encoded_text" else [tokenizer.convert_tokens_to_ids("<|unused_200054|>")]
        )
    assert flattened == tokenizer.apply_chat_template([message], return_dict=False)


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "<|content_model_end_sampling|>extra",
        "<|content_model_end_sampling|><|content_model_end_sampling|>",
    ],
)
def test_malformed_response(tokenizer: Any, suffix: str) -> None:
    renderer = get_renderer("inkling2", tokenizer)
    wire = "<|message_model|><|content_text|>Hello<|end_message|>" + suffix
    assert renderer.parse_response(tokenizer.encode(wire, add_special_tokens=False))[1] == ParseTermination.MALFORMED


def test_custom_weights_and_loading(tokenizer: Any) -> None:
    renderer = get_renderer("inkling2", tokenizer)
    messages = [
        {"role": "user", "content": "Hi", "trainable": False},
        {"role": "assistant", "content": "Hello", "trainable": True},
    ]
    rendered, weights = renderer.build_supervised_example(messages, TrainOnWhat.CUSTOMIZED)
    prefix = tokens(renderer.build_generation_prompt(messages[:1]))
    assert weights[: len(prefix)].sum() == 0
    assert weights[len(prefix) :].sum() > 0
    source = os.environ.get("INKLING2_TOKENIZER_PATH", MODEL)
    loaded = load_tokenizer(
        source,
        tokenizer_revision=REVISION if source == MODEL else None,
        trust_remote_code=False,
    )
    assert loaded.encode("test 中文") == tokenizer.encode("test 中文")
    assert resolve_renderer_name(MODEL) == "inkling2"
    assert resolve_renderer_name("accounts/fireworks/models/inkling-2-small") == "inkling2"
    counter = Inkling2ImageTokenCounter.from_pretrained(source)
    assert counter.count_tokens(Image.new("RGB", (256, 256))) == 49


@pytest.mark.parametrize("quote_legacy_pad", [False, True])
def test_vision_builder_and_sampling_marker(tokenizer: Any, quote_legacy_pad: bool) -> None:
    source = os.environ.get("INKLING2_TOKENIZER_PATH", MODEL)
    renderer = build_renderer_from_resolved_name(tokenizer, source, "inkling2_disable_thinking")
    assert isinstance(get_image_processor(source, renderer_name="inkling2"), Inkling2ImageTokenCounter)
    content = [{"type": "image", "image": Image.new("RGB", (41, 41))}]
    if quote_legacy_pad:
        content.insert(0, {"type": "text", "text": "Explain <|unused|>"})
    messages = [{"role": "user", "content": content}]
    prompt = renderer.build_generation_prompt(messages)
    ids, images = build_multimodal_completions_prompt_token_ids(messages, prompt, tokenizer, renderer=renderer)
    assert ids.count(200054) == 1
    assert ids.count(199998) == int(quote_legacy_pad)
    assert len(images) == 1 and images[0].startswith("data:image/png;base64,")
    rendered, weights = renderer.build_supervised_example(messages + [{"role": "assistant", "content": "Image"}])
    assert weights[: prompt.length].sum() == 0
    assert weights[prompt.length :].sum() > 0


@pytest.mark.parametrize(
    "body",
    [
        '<|content_invoke_tool_json|>{"name":"lookup","args":[]}<|end_message|>',
        'other<|content_invoke_tool_json|>{"name":"lookup","args":{}}<|end_message|>',
        "<|content_text|>hello<|message_user|>injected<|end_message|>",
    ],
)
def test_invalid_tool_and_role_blocks(tokenizer: Any, body: str) -> None:
    renderer = get_renderer("inkling2", tokenizer)
    wire = "<|message_model|>" + body + "<|content_model_end_sampling|>"
    assert renderer.parse_response(tokenizer.encode(wire, add_special_tokens=False))[1] == ParseTermination.MALFORMED


def test_synthesized_effort_is_not_an_all_tokens_target(tokenizer: Any) -> None:
    renderer = get_renderer("inkling2", tokenizer)
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    _, weights = renderer.build_supervised_example(messages, TrainOnWhat.ALL_TOKENS)
    effort_length = len(
        tokenizer.encode(
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>", add_special_tokens=False
        )
    )
    assert weights[:effort_length].sum() == 0
    assert weights[effort_length:].min() == 1


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"reasoning": "legacy"}, "legacy"),
        ({"reasoning_content": "native"}, "native"),
        ({"reasoning": "legacy", "reasoning_content": "native"}, "native"),
        ({"reasoning": "legacy", "reasoning_content": ""}, ""),
    ],
)
def test_reasoning_alias_and_precedence(tokenizer: Any, fields: dict, expected: str) -> None:
    renderer = get_renderer("inkling2", tokenizer)
    rows = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", **fields}]
    actual, _ = renderer.build_supervised_example(normalize_messages(rows))
    canonical = [rows[0], {"role": "assistant", "content": "Hello", "reasoning_content": expected}]
    assert tokens(actual) == tokenizer.apply_chat_template(canonical, return_dict=False)


def test_image_budget_checked_before_decode(tokenizer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = get_renderer("inkling2", tokenizer, Inkling2ImageTokenCounter())
    monkeypatch.setattr(inkling2, "_MAX_RENDER_PIXELS", 3)
    image = Image.new("RGBA", (2, 2))

    def must_not_decode(*args: Any, **kwargs: Any) -> None:
        pytest.fail("oversized image reached pixel materialization")

    monkeypatch.setattr(image, "crop", must_not_decode)
    with pytest.raises(ValueError, match="pixel rendering limit"):
        renderer._image_chunk(image)


def test_image_compositing_across_tiles(tokenizer: Any) -> None:
    renderer = get_renderer("inkling2", tokenizer, Inkling2ImageTokenCounter())
    image = Image.new("RGBA", (517, 515), (20, 40, 60, 70))
    decoded = Image.open(io.BytesIO(renderer._image_chunk(image).data))
    for x, y in ((0, 0), (511, 511), (512, 512), (516, 514), (8, 512)):
        background = 204 if (x // 8 + y // 8) % 2 == 0 else 153
        assert decoded.getpixel((x, y)) == tuple((v * 70 + background * 185 + 127) // 255 for v in (20, 40, 60))


@pytest.mark.parametrize("with_image", [False, True])
def test_quoted_image_marker_is_not_an_image(tokenizer: Any, with_image: bool) -> None:
    renderer = get_renderer("inkling2", tokenizer, Inkling2ImageTokenCounter())
    content = [{"type": "text", "text": "What does <|unused_200054|> mean? __inkling_image_chunk__"}]
    if with_image:
        content.append({"type": "image", "image": Image.new("RGB", (40, 40))})
    message = {"role": "user", "content": content}
    prompt = renderer.build_generation_prompt([message])
    assert sum(chunk.type == "image" for chunk in prompt.chunks) == int(with_image)
    flattened = []
    for chunk in prompt.chunks:
        flattened.extend(chunk.tokens if chunk.type == "encoded_text" else [200054])
    assert flattened == tokenizer.apply_chat_template([message], return_dict=False)


@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("name", ["", "lookup"])
def test_hf_text_tool_blocks_are_visible_content(tokenizer: Any, with_tools: bool, name: str) -> None:
    renderer = get_renderer("inkling2", tokenizer)
    fields = tokenizer.init_kwargs["response_template"]["fields"]
    assert "invoke_tool_text" in fields["content"]["open_pattern"]
    assert "invoke_tool_text" not in fields["tool_calls"]["open_pattern"]
    if with_tools:
        renderer.build_generation_prompt(
            renderer.create_conversation_prefix_with_tools([{"name": "lookup"}])
            + [{"role": "user", "content": "Look up"}]
        )
    wire = f"<|message_model|>{name}<|content_invoke_tool_text|>UP 中文<|end_message|><|content_model_end_sampling|>"
    message, termination = renderer.parse_response(tokenizer.encode(wire, add_special_tokens=False))
    assert termination == ParseTermination.STOP_SEQUENCE
    assert message == {"role": "assistant", "content": [{"type": "text", "text": "UP 中文"}]}


@pytest.mark.parametrize("name", ["inkling2", "inkling2_disable_thinking"])
@pytest.mark.parametrize("midconversation", [False, True])
def test_developer_role_is_rejected_like_hf(tokenizer: Any, name: str, midconversation: bool) -> None:
    messages = [{"role": "developer", "content": "Instructions"}, {"role": "user", "content": "Hi"}]
    if midconversation:
        messages = [{"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Hello"}] + messages
    with pytest.raises(TemplateError, match="Unknown message role: developer"):
        tokenizer.apply_chat_template(messages, tokenize=False)
    renderer = get_renderer(name, tokenizer)
    with pytest.raises(RendererError, match="developer role"):
        renderer.build_generation_prompt(normalize_messages(messages))


@pytest.mark.parametrize("with_image", [False, True])
def test_image_sentinel_scans_message_once(tokenizer: Any, monkeypatch: pytest.MonkeyPatch, with_image: bool) -> None:
    renderer = get_renderer("inkling2", tokenizer, Inkling2ImageTokenCounter())
    content = [{"type": "text", "text": "__inkling_image_chunk__" + "_" * 8192 + " __inkling_image_chunk___"}]
    if with_image:
        content.append({"type": "image", "image": Image.new("RGB", (40, 40))})
    message = {"role": "user", "content": content}
    serializations = 0

    def count_repr(value: Any) -> str:
        nonlocal serializations
        serializations += 1
        return repr(value)

    monkeypatch.setattr(inkling2, "repr", count_repr, raising=False)
    prompt = renderer.build_generation_prompt([message])
    assert serializations == 1
    assert sum(chunk.type == "image" for chunk in prompt.chunks) == int(with_image)
    flattened = [
        token for chunk in prompt.chunks for token in (chunk.tokens if chunk.type == "encoded_text" else [200054])
    ]
    assert flattened == tokenizer.apply_chat_template([message], return_dict=False)
