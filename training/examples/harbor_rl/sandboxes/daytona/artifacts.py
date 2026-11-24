"""Read files through Daytona's sandbox filesystem streaming API."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from functools import partial
from typing import Any

from .._files import ArtifactReader


async def artifact_reader(environment: Any, timeout: int) -> ArtifactReader:
    sandbox = environment._sandbox
    if sandbox is None:
        raise OSError("Harbor Daytona agent sandbox is unavailable")
    filesystem = sandbox.fs
    if not callable(getattr(filesystem, "download_file_stream", None)):
        raise OSError("Daytona artifact reads require a Daytona SDK with file streaming")
    return partial(_daytona_stream, filesystem, timeout=timeout)


async def _daytona_stream(
    filesystem: Any, path: str, *, timeout: int
) -> AsyncGenerator[bytes, None]:
    # lazy: Daytona is an optional Harbor provider.
    from daytona.common.errors import DaytonaNotFoundError

    try:
        source = await filesystem.download_file_stream(path, timeout=timeout)
        async with aclosing(source):
            async for data in source:
                yield data
    except DaytonaNotFoundError as exc:
        raise FileNotFoundError("Daytona artifact file is missing") from exc
