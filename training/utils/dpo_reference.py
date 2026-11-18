"""Persist the frozen DPO reference across serverless resumes."""

import json

from training.utils import fileio
from training.utils.errors import UserConfigError


def read_reference_checkpoint(path: str) -> str | None:
    raw = fileio.read_text(path)
    if not raw:
        return None
    try:
        value = json.loads(raw).get("reference_checkpoint")
    except (ValueError, AttributeError) as exc:
        raise UserConfigError("Frozen reference recovery state is corrupt") from exc
    if not isinstance(value, str) or not value:
        raise UserConfigError("Frozen reference recovery state is malformed")
    return value


def write_reference_checkpoint(path: str, reference: str) -> None:
    previous = read_reference_checkpoint(path)
    if previous is not None and previous != reference:
        raise UserConfigError("Frozen reference must remain fixed during a run")
    fileio.makedirs(path.rsplit("/", 1)[0] if "/" in path else ".")
    fileio.write_json(path, {"reference_checkpoint": reference})
