"""Read files from Harbor's live Docker agent container."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator
from functools import partial
from typing import Any

from .._files import CHUNK_BYTES, ArtifactReader


async def artifact_reader(environment: Any, _timeout: int) -> ArtifactReader:
    result = await environment._run_docker_compose_command(["ps", "-q", "main"])
    containers = [
        line
        for line in (result.stdout or "").splitlines()
        if re.fullmatch(r"[0-9a-f]{12,64}", line)
    ]
    if len(containers) != 1:
        raise OSError("Harbor did not identify one live Docker agent container")
    return partial(_docker_stream, containers[0])


async def _docker_stream(container_id: str, path: str) -> AsyncGenerator[bytes, None]:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "exec",
        container_id,
        "sh",
        "-c",
        'if test -e "$1"; then exec cat -- "$1"; else exit 66; fi',
        "tito-read-artifact",
        path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        assert proc.stdout is not None
        while data := await proc.stdout.read(CHUNK_BYTES):
            yield data
        code = await proc.wait()
        if code == 66:
            raise FileNotFoundError("Docker artifact file is missing")
        if code != 0:
            raise OSError("Docker artifact file read failed")
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.communicate()
