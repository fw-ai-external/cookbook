"""Recognize typed E2B setup and command transport failures."""

from __future__ import annotations

import re
from typing import Any

E2B_RETRY_EXCEPTION_MODULES = ("harbor.environments.e2b", "e2b.exceptions")


def _is_retryable_e2b_stream_open_timeout(
    exception: Any,
    *,
    harbor_environment: str,
) -> bool:
    """Recognize the provider failure that proves no command stream opened.

    Harbor serializes trial failures to names, messages, and tracebacks.  E2B's
    generic ``TimeoutException`` name is too broad for the public retry
    allowlist, while agent setup may wrap the same provider error in
    ``RuntimeError``.  Require both the E2B backend and the exact first-event
    timeout evidence from the retained traceback.
    """

    if harbor_environment != "e2b" or exception is None:
        return False
    traceback = str(getattr(exception, "exception_traceback", "") or "")
    timeout_marker = (
        "Request timed out: the stream didn't open within 'request_timeout'"
    )
    if timeout_marker not in traceback:
        return False
    provider_trace = (
        "site-packages/e2b/" in traceback
        and "harbor/environments/e2b.py" in traceback
        and re.search(
            rf"(?m)^e2b\.exceptions\.TimeoutException: {re.escape(timeout_marker)}",
            traceback,
        )
        is not None
    )
    cleanup_trace = re.search(
        rf"(?m)^TITO sidecar failure cleanup failed: {re.escape(timeout_marker)}",
        traceback,
    )
    return provider_trace or cleanup_trace is not None


def _is_retryable_e2b_sidecar_readiness_timeout(
    exception: Any,
    *,
    harbor_environment: str,
) -> bool:
    """Recognize Harbor's wrapper around a live E2B sidecar readiness timeout."""

    if harbor_environment != "e2b" or exception is None:
        return False
    traceback = str(getattr(exception, "exception_traceback", "") or "")
    exception_type = str(getattr(exception, "exception_type", "") or "")
    exception_message = str(getattr(exception, "exception_message", "") or "")
    marker = r"TITO sidecar did not become ready within [0-9]+(?:\.[0-9]+)?s"
    inner_timeout = (
        "training/examples/harbor_rl/tito/sidecar.py" in traceback
        and re.search(rf"(?m)^TimeoutError: {marker}$", traceback) is not None
    )
    if not inner_timeout:
        return False
    explicit_wrapper = (
        re.search(rf"(?m)^RuntimeError: Agent install failed: {marker}$", traceback)
        is not None
    )
    harbor_wrapper = (
        exception_type == "RuntimeError" and exception_message == "Agent install failed"
    )
    return explicit_wrapper or harbor_wrapper
