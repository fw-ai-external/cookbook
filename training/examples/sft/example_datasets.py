"""Download public Hugging Face examples as chat JSONL, without Fireworks datasets.

From ``cookbook/training``, run ``python -m examples.sft.example_datasets
--example prompt-router`` or ``--example cord-receipts``. The subsets and conversions match the prompt-router and CORD-v2 case-study notebooks. Conversion happens at
runtime; the output is cached on disk for subsequent runs.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

_PROMPT_ROUTER_REPO = "SupraLabs/Prompt-Routing-Dataset"
_CORD_RECEIPTS_REPO = "naver-clova-ix/cord-v2"
HF_EXAMPLES = ("prompt-router", "cord-receipts")

_ROUTING_SCHEMA = """Respond with ONLY a compact JSON object (no prose, no code fence) with exactly these keys, in this order:
- "complexity": integer 1, 2, 3, 4, 5 (1=trivial, 5=advanced)
- "math": true, false (needs symbolic math / proofs / multi-step word math)
- "code": true, false (needs code generation / debugging)
- "reasoning": true, false (needs multi-step reasoning)
- "route": one of "small model", "big model"

Example: {"complexity": 3, "math": true, "code": false, "reasoning": true, "route": "big model"}"""
_ROUTING_PROMPT = """You are a prompt-routing classifier for an LLM serving stack. Given a user prompt, analyze the task and decide whether a small local model can handle it or whether it must be escalated to a big frontier model.

""" + _ROUTING_SCHEMA + "\n/no_think"
_RECEIPT_INSTRUCTION = """Extract the receipt as a JSON object. Include the menu line items (name, quantity, unit price), plus subtotal, tax, and total when present. Respond with only the JSON object."""


def _prompt_router_row(example: dict[str, Any]) -> dict[str, Any]:
    label = {
        "complexity": int(example["complexity_score"]),
        "math": bool(example["math_task"]),
        "code": bool(example["coding_task"]),
        "reasoning": bool(example["requires_reasoning"]),
        "route": example["routing_choice"],
    }
    return {
        "messages": [
            {"role": "system", "content": _ROUTING_PROMPT},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": json.dumps(label)},
        ]
    }


def _receipt_image(image: Any) -> str:
    # Match the CORD notebook: bound the long side, RGB JPEG, quality 90.
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = 1024 / max(width, height)
    if scale < 1:
        rgb = rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _cord_receipt_row(example: dict[str, Any]) -> dict[str, Any]:
    ground_truth = example["ground_truth"]
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    label = ground_truth.get("gt_parse", ground_truth)
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _receipt_image(example["image"])}},
                    {"type": "text", "text": _RECEIPT_INSTRUCTION},
                ],
            },
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False, separators=(",", ":"))},
        ]
    }


def _cord_receipts() -> tuple[str, Iterable[dict[str, Any]], Callable[[dict[str, Any]], dict[str, Any]]]:
    from datasets import load_dataset

    return _CORD_RECEIPTS_REPO, load_dataset(_CORD_RECEIPTS_REPO, split="train", streaming=True), _cord_receipt_row


def _prompt_router() -> tuple[str, Iterable[dict[str, Any]], Callable[[dict[str, Any]], dict[str, Any]]]:
    from datasets import load_dataset

    # Same deterministic 600-row training subset as the prompt-router notebooks.
    dataset = load_dataset(_PROMPT_ROUTER_REPO, split="train").shuffle(seed=0).select(range(600))
    return _PROMPT_ROUTER_REPO, dataset, _prompt_router_row


def download_example_dataset(example: str, output_path: str | Path | None = None) -> str:
    """Convert one of ``HF_EXAMPLES`` to JSONL and return its absolute path.

    An existing output is reused. Replace it explicitly to fetch a fresh copy.
    Download or conversion errors propagate without leaving a partial output.
    """
    if example not in HF_EXAMPLES:
        raise ValueError(f"Unknown example {example!r}; choose one of {', '.join(HF_EXAMPLES)}.")
    output = Path(output_path or f"{example}.jsonl").expanduser().resolve()
    if output.exists():
        return str(output)
    repository, rows, convert = _prompt_router() if example == "prompt-router" else _cord_receipts()
    iterator: Iterator[dict[str, Any]] = iter(rows)
    count = 0
    limit = 600 if example == "prompt-router" else 300
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            for row in iterator:
                if count >= limit:
                    break
                file.write(json.dumps(convert(row), ensure_ascii=False) + "\n")
                count += 1
        if not count:
            raise ValueError(f"No rows downloaded from {repository}.")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded {count} rows from https://huggingface.co/datasets/{repository} to {output}")
    return str(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True, choices=HF_EXAMPLES)
    parser.add_argument("--output", help="Default: ./<example>.jsonl")
    args = parser.parse_args(argv)
    download_example_dataset(args.example, args.output)


if __name__ == "__main__":
    main()
