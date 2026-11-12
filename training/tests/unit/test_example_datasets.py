"""Local tests for public Hugging Face example conversion."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from training.examples.sft.example_datasets import download_example_dataset, main


class _FakeRouter:
    def shuffle(self, *, seed):
        assert seed == 0
        return self

    def select(self, indices):
        assert list(indices) == list(range(600))
        return [
            {
                "prompt": "route this",
                "complexity_score": 3,
                "math_task": True,
                "coding_task": False,
                "requires_reasoning": True,
                "routing_choice": "big model",
            }
        ]


def test_prompt_router_converts_chat_without_network_when_cached(tmp_path):
    output = tmp_path / "prompts.jsonl"
    with patch("datasets.load_dataset", return_value=_FakeRouter()) as load:
        assert download_example_dataset("prompt-router", output) == str(output)
    load.assert_called_once_with("SupraLabs/Prompt-Routing-Dataset", split="train")
    row = json.loads(output.read_text())
    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
    assert json.loads(row["messages"][-1]["content"])["route"] == "big model"
    with patch("datasets.load_dataset", side_effect=AssertionError("must reuse local copy")):
        download_example_dataset("prompt-router", output)


def test_receipt_conversion_matches_notebook():
    row = {
        "ground_truth": json.dumps({"gt_parse": {"total": {"price": "12.00"}}}),
        "image": None,
    }
    from training.examples.sft.example_datasets import _cord_receipt_row
    with patch("training.examples.sft.example_datasets._receipt_image", return_value="data:image/jpeg;base64,abc"):
        converted = _cord_receipt_row(row)
    content = converted["messages"][0]["content"]
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,abc"
    assert json.loads(converted["messages"][1]["content"]) == {"total": {"price": "12.00"}}


def test_unknown_example_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown example"):
        download_example_dataset("other", tmp_path / "other.jsonl")


def test_failed_conversion_does_not_leave_output(tmp_path):
    output = tmp_path / "receipts.jsonl"
    with patch("training.examples.sft.example_datasets._cord_receipts", return_value=("repo", [object()], lambda row: (_ for _ in ()).throw(ValueError("bad row")))):
        with pytest.raises(ValueError, match="bad row"):
            download_example_dataset("cord-receipts", output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_cli_writes_named_example():
    with patch("training.examples.sft.example_datasets.download_example_dataset") as download:
        main(["--example", "cord-receipts", "--output", "receipts.jsonl"])
    download.assert_called_once_with("cord-receipts", "receipts.jsonl")
