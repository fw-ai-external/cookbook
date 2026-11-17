"""Inkling 2 adapter using the publisher's HF chat template.

Reference: thinkingmachines/Inkling-2-Small at
2ad1157016e1aac77b6ff1b770e7c05229403134. Composite transparency before
lossless transport; FireTitan owns pixel normalization and patchification.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fireworks.training import sdk
from huggingface_hub import hf_hub_download
from PIL import Image

from training.renderer import register_renderer
from training.renderer.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
)
from training.renderer.errors import RendererError
from training.renderer.image_fetcher import read_image_url
from training.renderer.message_weights import untrained_synthesized_context
from training.renderer.reasoning_fields import (
    original_reasoning_content,
)

_IMAGE = "<|unused_200054|>"
_END = "<|end_message|>"
_STOP = "<|content_model_end_sampling|>"
_WIRE = "_inkling2_wire"
_MAX_RENDER_PIXELS = 25_000_000


class Inkling2ImageTokenCounter:
    """No resizing: one decoder token per ceil(H/40) x ceil(W/40) cell."""

    @classmethod
    def from_pretrained(cls, model_name: str) -> Inkling2ImageTokenCounter:
        path = Path(model_name) / "processor_config.json"
        if not path.is_file():
            path = Path(hf_hub_download(model_name, "processor_config.json"))
        config = json.loads(path.read_text())
        processor = config.get("image_processor", {})
        if (
            config.get("processor_class") != "Inkling2Processor"
            or config.get("image_token") != _IMAGE
            or processor.get("size") != {"height": 40, "width": 40}
            or processor.get("image_processor_type") != "Inkling2ImageProcessor"
            or processor.get("image_mean") != [0.5, 0.5, 0.5]
            or processor.get("image_std") != [0.5, 0.5, 0.5]
            or processor.get("rescale_factor") != 1 / 255
            or any(processor.get(key) is not True for key in ("do_convert_rgb", "do_normalize", "do_rescale"))
        ):
            raise RendererError("Unsupported Inkling 2 image processor configuration")
        return cls()

    def count_tokens(self, image: Image.Image) -> int:
        width, height = image.size
        return ((height + 39) // 40) * ((width + 39) // 40)


class Inkling2Renderer(Renderer):
    def __init__(
        self,
        tokenizer: Any,
        image_processor: Any = None,
        *,
        reasoning_effort: float = 0.9,
    ) -> None:
        super().__init__(tokenizer)
        if not getattr(tokenizer, "chat_template", None):
            raise RendererError("Inkling 2 requires the publisher's chat_template.jinja")
        self.image_processor = image_processor
        self.reasoning_effort = reasoning_effort
        effort = "0" if reasoning_effort == 0 else str(reasoning_effort)
        self._effort = f"<|message_system|><|content_text|>Thinking effort level: {effort}{_END}"

    @property
    def has_extension_property(self) -> bool:
        return True

    @property
    def image_placeholder_token_id(self) -> int:
        # Sampling uses the same image placeholder as the HF processor.
        ids = self.tokenizer.encode(_IMAGE, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == getattr(self.tokenizer, "unk_token_id", None):
            raise RendererError("Inkling 2 serving image marker must be one known token")
        return ids[0]

    def get_stop_sequences(self) -> list[str]:
        return [_STOP]

    def _template(self, messages: list[dict], **kwargs: Any) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            reasoning_effort=self.reasoning_effort,
            **kwargs,
        )

    def _encode(self, text: str) -> sdk.EncodedTextChunk:
        # Deliberately recognize quoted special-token spellings, like HF's
        # apply_chat_template(tokenize=True), rather than encode_ordinary.
        return sdk.EncodedTextChunk(tokens=self.tokenizer.encode(text, add_special_tokens=False))

    def to_openai_message(self, message: Message) -> dict[str, Any]:
        if message.get("unparsed_tool_calls"):
            raise RendererError("Cannot render malformed Inkling 2 tool calls")
        content = message["content"]
        raw = {k: message[k] for k in ("role", "name", "tool_call_id") if k in message}
        if raw["role"] == "developer":
            raise RendererError("Inkling 2 does not support the developer role")
        if isinstance(content, str):
            raw["content"] = content
        else:
            raw["content"] = [p for p in content if p["type"] != "thinking"]
            raw["reasoning_content"] = "".join(p["thinking"] for p in content if p["type"] == "thinking")
        has_original, reasoning = original_reasoning_content(message)
        if has_original:
            # Explicit reasoning_content wins; otherwise keep the cookbook's
            # normalized thinking parts, including its legacy reasoning alias.
            raw["reasoning_content"] = reasoning
        if (
            raw.get("reasoning_content")
            and isinstance(raw["content"], list)
            and all(part.get("type") == "text" and part.get("text") == "" for part in raw["content"])
        ):
            raw["content"] = ""
        if message.get("tool_calls"):
            raw["tool_calls"] = []
            for call in message["tool_calls"]:
                args = json.loads(call.function.arguments)
                if not isinstance(args, dict):
                    raise RendererError("Inkling 2 tool arguments must be a JSON object")
                raw["tool_calls"].append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": args},
                    }
                )
        return raw

    def _prepare(self, messages: list[Message]) -> list[Message]:
        prepared = []
        names = {}
        effort_emitted = False
        for message in messages:
            if not effort_emitted and message["role"] not in {"system", "developer"}:
                prepared.append({"role": "system", "content": "", _WIRE: self._effort})
                effort_emitted = True
            current = dict(message)
            if current["role"] == "tool" and not current.get("name"):
                name = names.get(current.get("tool_call_id"))
                if name:
                    current["name"] = name
            for call in current.get("tool_calls", []):
                if call.id is not None:
                    names[call.id] = call.function.name
            prepared.append(current)
        if not effort_emitted:
            prepared.append({"role": "system", "content": "", _WIRE: self._effort})
        return untrained_synthesized_context(prepared)

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> sdk.ModelInput:
        if role != "assistant":
            raise RendererError("Inkling 2 generation requires the assistant role")
        return super().build_generation_prompt(self._prepare(messages), role, prefill)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[sdk.ModelInput, torch.Tensor]:
        prepared = self._prepare(messages)
        model_input, weights = super().build_supervised_example(prepared, train_on_what)
        if train_on_what == TrainOnWhat.ALL_TOKENS:
            # Template-injected effort/tool context is never a dataset target.
            offset = 0
            for index, message in enumerate(prepared):
                rendered = self.render_message(message, RenderContext(idx=index, is_last=index == len(prepared) - 1))
                length = sum(chunk.length for chunk in ([rendered.header] if rendered.header else []) + rendered.output)
                if _WIRE in message:
                    weights[offset : offset + length] = 0
                offset += length
        return model_input, weights

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        # HF emits no assistant prefix: the model generates its typed headers.
        return []

    def _image_chunk(self, source: Any) -> sdk.types.ImageChunk:
        if self.image_processor is None:
            raise RendererError("Inkling 2 image rendering requires its image processor")
        image = Image.open(io.BytesIO(read_image_url(source))) if isinstance(source, str) else source
        if not isinstance(image, Image.Image):
            raise RendererError("Expected a PIL image or image URL")
        if image.width * image.height > _MAX_RENDER_PIXELS:
            raise RendererError(f"Image exceeds the {_MAX_RENDER_PIXELS}-pixel rendering limit")
        output = io.BytesIO()
        # Composite in bounded tiles: full-image uint16 arithmetic would
        # amplify an untrusted image's decoded memory several times over.
        rgb_image = Image.new("RGB", image.size)
        for top in range(0, image.height, 512):
            for left in range(0, image.width, 512):
                box = (left, top, min(left + 512, image.width), min(top + 512, image.height))
                rgba = np.asarray(image.crop(box).convert("RGBA"), dtype=np.uint16)
                y, x = np.ogrid[top : box[3], left : box[2]]
                background = np.where((x // 8 + y // 8) % 2 == 0, 204, 153).astype(np.uint16)
                alpha = rgba[:, :, 3:4]
                rgb = ((rgba[:, :, :3] * alpha + background[:, :, None] * (255 - alpha) + 127) // 255).astype(np.uint8)
                rgb_image.paste(Image.fromarray(rgb), (left, top))
        rgb_image.save(output, format="PNG")
        return sdk.types.ImageChunk(
            data=output.getvalue(),
            format="png",
            expected_tokens=self.image_processor.count_tokens(image),
        )

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        if _WIRE in message:
            return RenderedMessage(header=self._encode(message[_WIRE]), output=[])
        raw = self.to_openai_message(message)
        # Replace only the template's image placeholder, never quoted payload
        # text. Pick a deterministic sentinel absent from the input data.
        # A suffix longer than any underscore run cannot occur in the payload.
        # Scan once so adversarial marker prefixes cannot cause quadratic work.
        longest_run = max((len(match[0]) for match in re.finditer(r"_+", repr(raw))), default=0)
        image_marker = "__inkling_image_chunk__" + "_" * (longest_run + 1)
        template = self.tokenizer.chat_template.replace(_IMAGE, image_marker)
        wire = self._template([raw], chat_template=template)
        if raw["role"] == "system":
            if not wire.endswith(self._effort):
                raise RendererError("Unexpected Inkling 2 reasoning-effort template")
            wire = wire[: -len(self._effort)]
        else:
            if not wire.startswith(self._effort):
                raise RendererError("Unexpected Inkling 2 reasoning-effort template")
            wire = wire[len(self._effort) :]
        content = message["content"]
        images = [p["image"] for p in content if p["type"] == "image"] if isinstance(content, list) else []
        segments = wire.split(image_marker)
        if len(segments) != len(images) + 1:
            raise RendererError("Inkling 2 image placeholders do not match image content")
        chunks = []
        for index, segment in enumerate(segments):
            if segment:
                chunks.append(self._encode(segment))
            if index < len(images):
                chunks.append(self._image_chunk(images[index]))
        # Assistant headers must carry loss: unlike most renderers there is no
        # generation prefix that would supply them to the model.
        return RenderedMessage(header=None, output=chunks)

    def create_conversation_prefix_with_tools(self, tools: list[ToolSpec], system_prompt: str = "") -> list[Message]:
        messages = []
        if tools:
            sentinel = {"role": "user", "content": ""}
            wire = self._template([sentinel], tools=tools)
            suffix = self._effort + f"<|message_user|><|content_text|>{_END}"
            if not wire.endswith(suffix):
                raise RendererError("Unexpected Inkling 2 tool template preamble")
            messages.append({"role": "system", "content": "", _WIRE: wire[: -len(suffix)]})
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        return messages

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        # Parse token IDs, not decoded marker strings: literal text resembling
        # a control marker must not become a framing delimiter.
        special = self.tokenizer.convert_tokens_to_ids
        model, end, stop = map(special, ("<|message_model|>", _END, _STOP))
        kinds = {
            special(f"<|content_{kind}|>"): kind
            for kind in (
                "text",
                "thinking",
                "tool_error",
                "invoke_tool_text",
                "invoke_tool_json",
            )
        }
        controls = (
            set(kinds)
            | {
                special(marker)
                for marker in (
                    "<|message_user|>",
                    "<|message_system|>",
                    "<|message_tool|>",
                    "<|content_image|>",
                    "<|content_xml|>",
                    "<|begin_of_text|>",
                    "<|endoftext|>",
                )
            }
            | {model, end, stop}
        )
        message: Message = {"role": "assistant", "content": []}
        pos = 0
        while pos < len(response):
            if response[pos] == stop:
                return message, ParseTermination.STOP_SEQUENCE if pos == len(
                    response
                ) - 1 else ParseTermination.MALFORMED
            if response[pos] != model:
                return message, ParseTermination.MALFORMED
            pos += 1
            name_start = pos
            while pos < len(response) and response[pos] not in kinds:
                if response[pos] in controls:
                    return message, ParseTermination.MALFORMED
                pos += 1
            if pos == len(response):
                return message, ParseTermination.MALFORMED
            name = self.tokenizer.decode(response[name_start:pos])
            kind = kinds[response[pos]]
            pos += 1
            start = pos
            while pos < len(response) and response[pos] != end:
                if response[pos] in controls:
                    return message, ParseTermination.MALFORMED
                pos += 1
            body = self.tokenizer.decode(response[start:pos], skip_special_tokens=False)
            if pos == len(response):
                return message, ParseTermination.MALFORMED
            pos += 1
            if kind == "invoke_tool_json":
                try:
                    call = json.loads(body)
                    if (
                        not isinstance(call["name"], str)
                        or not isinstance(call["args"], dict)
                        or (name and name != call["name"])
                    ):
                        raise ValueError("Invalid tool call")
                    message.setdefault("tool_calls", []).append(
                        ToolCall(
                            function=ToolCall.FunctionBody(
                                name=call["name"],
                                arguments=json.dumps(call["args"], ensure_ascii=False),
                            )
                        )
                    )
                except (ValueError, KeyError, TypeError):
                    return message, ParseTermination.MALFORMED
            else:
                # HF response_template treats invoke_tool_text as content;
                # it must not become a synthetic tool call when tools exist.
                message["content"].append(
                    {"type": "thinking", "thinking": body} if kind == "thinking" else {"type": "text", "text": body}
                )
        return message, ParseTermination.MALFORMED


register_renderer(
    "inkling2",
    lambda tokenizer, image_processor=None: Inkling2Renderer(tokenizer, image_processor),
)
register_renderer(
    "inkling2_disable_thinking",
    lambda tokenizer, image_processor=None: Inkling2Renderer(tokenizer, image_processor, reasoning_effort=0.0),
)
