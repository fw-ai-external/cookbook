"""Pi-specific inspection of retained Harbor agent events."""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path


def tool_timeout_count(trial_path: Path) -> int:
    timed_out_calls: set[str] = set()
    pending_bash_calls: set[str] = set()
    for path in sorted((trial_path / "agent").glob("*.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping):
                continue
            if event.get("type") == "tool_execution_start":
                if event.get("toolName") == "bash":
                    pending_bash_calls.add(str(event.get("toolCallId") or ""))
                continue
            if event.get("type") != "tool_execution_end":
                continue
            call_id = str(event.get("toolCallId") or "")
            pending_bash_calls.discard(call_id)
            result = event.get("result")
            encoded = json.dumps(result, sort_keys=True) if result is not None else ""
            if event.get("toolName") == "bash" and "timed out after" in encoded:
                timed_out_calls.add(call_id or str(len(timed_out_calls)))

    # Pi can be killed before its terminal tool event is flushed. Attribute the
    # timeout only when the pending bash call, exit 137, and Harbor matcher agree.
    result_path = trial_path / "result.json"
    if pending_bash_calls and result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            result = {}
        exception = result.get("exception_info") or {}
        message = str(exception.get("exception_message") or "")
        trial_log_path = trial_path / "trial.log"
        try:
            trial_log = trial_log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            trial_log = ""
        if "exit 137" in message and "pattern: 'Request timed out'" in trial_log:
            timed_out_calls.update(pending_bash_calls)
    return len(timed_out_calls)


def tool_execution_metrics(trial_path: Path) -> dict[str, float]:
    """Read Pi's bounded numeric summary, with missing evidence left uncovered."""
    path = trial_path / "agent" / "pi-tools.json"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return {}
            payload = stream.read(4097)
        if len(payload) > 4096:
            return {}
        document = json.loads(payload)
    except (OSError, RecursionError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    fields = ("succeeded", "failed", "unclosed", "work_seconds", "busy_seconds")
    result = {}
    for field in fields:
        value = document.get(field)
        if type(value) not in (int, float):
            return {}
        try:
            number = float(value)
        except OverflowError:
            return {}
        if not math.isfinite(number) or number < 0:
            return {}
        if not field.endswith("_seconds") and not number.is_integer():
            return {}
        result[f"tito/tool/{field}"] = number
    result["tito/tool/observed_rollouts"] = 1.0
    return result


__all__ = ["tool_execution_metrics", "tool_timeout_count"]
