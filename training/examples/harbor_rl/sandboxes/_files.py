"""File reader contract shared by the sandbox adapters and TITO collector."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

CHUNK_BYTES = 64 * 1024
ArtifactReader = Callable[[str], AsyncGenerator[bytes, None]]
ArtifactReaderFactory = Callable[[Any, int], Awaitable[ArtifactReader]]


class ArtifactSourceUnavailable(OSError):
    """The sandbox is gone and its file transfer cannot be retried."""
