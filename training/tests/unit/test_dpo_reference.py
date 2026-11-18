"""Frozen-reference recovery is durable and fails closed on corrupt state."""

import pytest

from training.utils.dpo_reference import (
    read_reference_checkpoint,
    write_reference_checkpoint,
)
from training.utils.errors import UserConfigError


def test_reference_persists_and_cannot_be_replaced(tmp_path):
    path = str(tmp_path / "run" / "reference_checkpoint.json")
    assert read_reference_checkpoint(path) is None
    write_reference_checkpoint(path, "snapshot-original")
    assert read_reference_checkpoint(path) == "snapshot-original"
    write_reference_checkpoint(path, "snapshot-original")
    with pytest.raises(UserConfigError, match="remain fixed"):
        write_reference_checkpoint(path, "snapshot-new-policy")
    assert read_reference_checkpoint(path) == "snapshot-original"


@pytest.mark.parametrize("raw", ["invalid", "[]", "{}", '{"reference_checkpoint": ""}'])
def test_corrupt_reference_is_not_silently_recreated(tmp_path, raw):
    path = tmp_path / "reference_checkpoint.json"
    path.write_text(raw)
    with pytest.raises(UserConfigError):
        read_reference_checkpoint(str(path))
