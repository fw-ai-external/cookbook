"""Tests for the sidecar's empty-reasoning lineage canonicalization.

The sidecar folds protocol-equivalent *empty* ``reasoning_content``/``reasoning``
spellings (None / absent / whitespace) together so that a reasoning-free assistant
turn committed by the sampler and replayed by the agent do not open a false TITO
history-rewrite segment. A non-empty reasoning string must still be preserved so a
genuine reasoning difference keeps splitting (on-policy correctness).
"""

from __future__ import annotations

import pytest

from fireworks.training.sdk.tito import _types as tito_types
from training.examples.harbor_rl.tito.sidecar import (
    _install_assistant_reasoning_canonicalization,
)


@pytest.fixture
def canonicalizer():
    """Install the wrapper and restore the pristine SDK function afterwards."""
    original = tito_types._normalize_chat_message
    try:
        _install_assistant_reasoning_canonicalization()
        yield tito_types._normalize_chat_message
    finally:
        tito_types._normalize_chat_message = original


def _assistant(**extra):
    msg = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_x", "type": "function",
         "function": {"name": "read", "arguments": '{"path":"/a"}'}}
    ]}
    msg.update(extra)
    return msg


def test_empty_reasoning_spellings_collapse_equal(canonicalizer):
    none_form = canonicalizer(_assistant(reasoning_content=None))
    empty_form = canonicalizer(_assistant(reasoning_content=""))
    ws_form = canonicalizer(_assistant(reasoning_content="  \n"))
    absent_form = canonicalizer(_assistant())  # no reasoning_content key at all
    # All four "no reasoning" spellings must canonicalize identically, so the
    # committed turn and its replay append instead of forcing a new segment.
    assert none_form == empty_form == ws_form == absent_form
    assert "reasoning_content" not in none_form


def test_legacy_reasoning_field_also_collapses(canonicalizer):
    assert canonicalizer(_assistant(reasoning=None)) == canonicalizer(_assistant(reasoning=""))
    assert "reasoning" not in canonicalizer(_assistant(reasoning="   "))


def test_nonempty_reasoning_is_preserved_and_still_differs(canonicalizer):
    with_text = canonicalizer(_assistant(reasoning_content="real reasoning"))
    without = canonicalizer(_assistant(reasoning_content=""))
    # A genuine reasoning difference must survive canonicalization so TITO still
    # opens a new segment rather than training an off-policy context.
    assert with_text != without
    assert with_text["reasoning_content"] == "real reasoning"


def test_non_assistant_messages_untouched(canonicalizer):
    # Only assistant turns participate in the assistant_roundtrip lineage check.
    user = canonicalizer({"role": "user", "content": "hi", "reasoning_content": ""})
    assert user.get("reasoning_content") == ""


def test_existing_content_normalization_preserved(canonicalizer):
    # The wrapper must not regress the SDK's own null/whitespace content handling
    # for tool-call-only assistant messages.
    null_content = canonicalizer(_assistant(content=None, reasoning_content=None))
    empty_content = canonicalizer(_assistant(content="", reasoning_content=""))
    assert null_content == empty_content
    assert null_content["content"] == ""


def test_install_is_idempotent():
    original = tito_types._normalize_chat_message
    try:
        _install_assistant_reasoning_canonicalization()
        once = tito_types._normalize_chat_message
        _install_assistant_reasoning_canonicalization()
        twice = tito_types._normalize_chat_message
        assert once is twice  # second install is a no-op, no double-wrapping
    finally:
        tito_types._normalize_chat_message = original
