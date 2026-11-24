"""Read files through E2B's unbuffered filesystem API."""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator
from contextlib import aclosing
from functools import lru_cache, partial
from typing import Any

from .._files import ArtifactReader, ArtifactSourceUnavailable


@lru_cache(maxsize=1)
def e2b_supports_artifact_streaming() -> bool:
    try:
        # lazy: E2B is an optional Harbor provider.
        from e2b import exceptions
        from e2b.sandbox_async.filesystem.filesystem import Filesystem
    except ImportError:
        return False
    # Older versions expose format="stream" but buffer the HTTP response.
    return (
        hasattr(exceptions, "SandboxNotRunningException")
        and "stream_idle_timeout" in inspect.signature(Filesystem.read).parameters
    )


async def artifact_reader(environment: Any, timeout: int) -> ArtifactReader:
    sandbox = environment._sandbox
    if sandbox is None:
        raise ArtifactSourceUnavailable("Harbor E2B agent sandbox is unavailable")
    return partial(_e2b_stream, sandbox.files, timeout=timeout)


async def _e2b_stream(
    filesystem: Any, path: str, *, timeout: int
) -> AsyncGenerator[bytes, None]:
    # lazy: E2B is an optional Harbor provider.
    from e2b.exceptions import (
        FileNotFoundException,
        SandboxNotFoundException,
        SandboxNotRunningException,
    )

    try:
        source = await filesystem.read(path, format="stream", request_timeout=timeout)
        async with aclosing(source):
            async for block in source:
                yield block
    except FileNotFoundException as exc:
        raise FileNotFoundError("E2B artifact file is missing") from exc
    except (SandboxNotFoundException, SandboxNotRunningException) as exc:
        raise ArtifactSourceUnavailable(
            "Harbor E2B agent sandbox is unavailable"
        ) from exc
