"""Select a sandbox reader while Harbor can still use legacy collection."""

from collections.abc import Callable

from ._files import ArtifactReaderFactory
from .daytona.artifacts import artifact_reader as daytona_reader
from .docker.artifacts import artifact_reader as docker_reader
from .e2b.artifacts import artifact_reader as e2b_reader
from .e2b.artifacts import e2b_supports_artifact_streaming


def artifact_reader_factory(
    provider: str, *, uses_compose: Callable[[], bool]
) -> ArtifactReaderFactory | None:
    if provider == "docker":
        return docker_reader
    if provider == "daytona" and not uses_compose():
        return daytona_reader
    if provider == "e2b" and e2b_supports_artifact_streaming():
        return e2b_reader
    return None
