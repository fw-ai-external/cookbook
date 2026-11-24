"""Recover a sealed trajectory over the sandbox's existing provider connection."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import logging
import re
import time
import zlib
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary

from fireworks.training.sdk import TITOTrajectoryArtifact

from training.examples.harbor_rl.sandboxes._files import (
    CHUNK_BYTES,
    ArtifactReader,
    ArtifactReaderFactory,
    ArtifactSourceUnavailable,
)
from training.examples.harbor_rl.tito._artifact_io import wait_for_artifact_worker
from training.examples.harbor_rl.tito.sidecar import (
    SIDECAR_ARTIFACT_MANIFEST_PATH,
    SIDECAR_ARTIFACT_PATH,
    SIDECAR_COMPLETE_PATH,
)
from training.utils.rl.async_rl.errors import RecoverableRolloutError

logger = logging.getLogger(__name__)

MAX_ARTIFACT_BYTES = 2 * 1024**3
MAX_EXPANDED_ARTIFACT_BYTES = 8 * 1024**3
MAX_MANIFEST_BYTES = 16 * 1024
_TRANSFER_TIMEOUT_SECONDS = 300
_READ_IDLE_TIMEOUT_SECONDS = 30
_LOCAL_STORAGE_ERRORS = {
    errno.ENOSPC,
    errno.EDQUOT,
    errno.EACCES,
    errno.EROFS,
    errno.EFBIG,
}
_MAX_ACTIVE_TRANSFERS = 4
_SPOOL_EXECUTOR = ThreadPoolExecutor(
    max_workers=_MAX_ACTIVE_TRANSFERS, thread_name_prefix="tito-trajectory-spool"
)
_RECEIPT_SLOTS: WeakValueDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    WeakValueDictionary()
)


def artifact_transfer_slots() -> asyncio.Semaphore:
    """Limit simultaneous reads across trials sharing a runner."""
    loop = asyncio.get_running_loop()
    permits = _RECEIPT_SLOTS.get(loop)
    if permits is None:
        permits = _RECEIPT_SLOTS[loop] = asyncio.Semaphore(_MAX_ACTIVE_TRANSFERS)
    return permits


def read_manifest(payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("TITO sidecar artifact manifest exceeds limit")
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError("TITO sidecar artifact manifest is not an object")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("unsupported TITO sidecar artifact manifest")
    byte_count, digest = manifest.get("bytes"), manifest.get("sha256")
    if type(byte_count) is not int or not 0 < byte_count <= MAX_ARTIFACT_BYTES:
        raise ValueError("TITO sidecar artifact byte count exceeds accepted bounds")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid TITO sidecar artifact checksum")
    return manifest


def _check_legacy_compression(encoded: bytearray) -> None:
    # SDKs before unpack_stream inflate the whole payload. Check the v1
    # framing and expansion limit without retaining any plaintext first.
    magic = b"TITOART\x01"
    if not encoded.startswith(magic):
        raise ValueError("missing TITO artifact magic/version")
    inflater = zlib.decompressobj()
    expanded = 0
    wire = memoryview(encoded)
    for offset in range(len(magic), len(wire), CHUNK_BYTES):
        pending = wire[offset : offset + CHUNK_BYTES]
        while pending:
            if inflater.eof:
                raise ValueError(
                    "trailing bytes after TITO artifact compression stream"
                )
            block = inflater.decompress(
                pending, min(CHUNK_BYTES, MAX_EXPANDED_ARTIFACT_BYTES - expanded + 1)
            )
            expanded += len(block)
            if expanded > MAX_EXPANDED_ARTIFACT_BYTES:
                raise ValueError("TITO artifact expanded size exceeds limit")
            if inflater.unused_data:
                raise ValueError(
                    "trailing bytes after TITO artifact compression stream"
                )
            pending = inflater.unconsumed_tail
    if not inflater.eof:
        raise ValueError("incomplete TITO artifact compression stream")


def decode_artifact(
    chunks: Iterable[bytes], manifest: Mapping[str, Any], *, verified: bool = False
) -> TITOTrajectoryArtifact:
    expected_sha256 = None if verified else manifest["sha256"]
    stream_unpack = getattr(TITOTrajectoryArtifact, "unpack_stream", None)
    if stream_unpack is None:
        # The Cookbook's declared SDK minimum predates the incremental codec.
        encoded = bytearray()
        for block in chunks:
            if len(block) > manifest["bytes"] - len(encoded):
                raise ValueError("TITO sidecar artifact byte count mismatch")
            encoded.extend(block)
        if len(encoded) != manifest["bytes"]:
            raise ValueError("TITO sidecar artifact byte count mismatch")
        if (
            expected_sha256 is not None
            and hashlib.sha256(encoded).hexdigest() != expected_sha256
        ):
            raise ValueError("TITO sidecar artifact checksum mismatch")
        _check_legacy_compression(encoded)
        artifact = TITOTrajectoryArtifact.unpack(bytes(encoded))
    else:
        artifact = stream_unpack(
            chunks,
            expected_bytes=manifest["bytes"],
            expected_sha256=expected_sha256,
            max_expanded_bytes=MAX_EXPANDED_ARTIFACT_BYTES,
        )
    for field, description in (
        ("trajectory_id", "trajectory identity"),
        ("status", "terminal status"),
        ("terminal_reason", "terminal reason"),
    ):
        if getattr(artifact, field) != manifest.get(field):
            raise ValueError(f"TITO sidecar artifact {description} mismatch")
    return artifact


async def _small_file(
    open_stream: ArtifactReader,
    path: str,
    limit: int,
    measurements: dict[str, float],
) -> bytes:
    for attempt in range(2):
        payload = bytearray()
        try:
            # The small control files have a whole-request deadline.
            async with asyncio.timeout(_READ_IDLE_TIMEOUT_SECONDS):
                async with aclosing(open_stream(path)) as source:
                    async for chunk in source:
                        if len(chunk) > limit - len(payload):
                            raise ValueError(
                                f"TITO sidecar file {Path(path).name} exceeds limit"
                            )
                        payload.extend(chunk)
            return bytes(payload)
        except (ArtifactSourceUnavailable, FileNotFoundError, ValueError):
            raise
        except Exception:
            if attempt:
                raise
            measurements["artifact_metadata_retries"] += 1
    raise AssertionError("artifact metadata retry loop returned no result")


class _ShortTransfer(OSError):
    """The source ended before all advertised bytes arrived."""


async def _spool_artifact(
    open_stream: ArtifactReader,
    path: Path,
    manifest: Mapping[str, Any],
    measurements: dict[str, float],
) -> None:
    expected_bytes = manifest["bytes"]
    digest = hashlib.sha256()
    byte_count = 0
    batch = bytearray()

    with path.open("wb") as output:

        def write(data: bytearray) -> None:
            digest.update(data)
            output.write(data)

        async with aclosing(open_stream(SIDECAR_ARTIFACT_PATH)) as source:
            while True:
                before = time.monotonic()
                try:
                    async with asyncio.timeout(_READ_IDLE_TIMEOUT_SECONDS):
                        block = await anext(source, None)
                finally:
                    now = time.monotonic()
                    measurements["artifact_read_seconds"] += now - before
                if block is None:
                    break
                if not block:
                    continue
                if byte_count == 0:
                    measurements["artifact_first_byte_seconds"] += now - before
                byte_count += len(block)
                measurements["artifact_transferred_bytes"] += len(block)
                if byte_count > expected_bytes:
                    raise ValueError("TITO sidecar artifact exceeded advertised size")
                batch.extend(block)
                if len(batch) >= 1024 * 1024:
                    await wait_for_artifact_worker(_SPOOL_EXECUTOR.submit(write, batch))
                    batch.clear()
            if batch:
                await wait_for_artifact_worker(_SPOOL_EXECUTOR.submit(write, batch))
    if byte_count != expected_bytes:
        raise _ShortTransfer("TITO sidecar artifact stream ended early")
    if digest.hexdigest() != manifest["sha256"]:
        raise ValueError("TITO sidecar artifact checksum mismatch")


async def collect_trajectory(
    environment: Any,
    reader_factory: ArtifactReaderFactory,
    *,
    trajectory_id: str,
    bundle_digest: str,
    spool: Path,
    measurements: dict[str, float],
) -> tuple[Path, dict[str, Any]]:
    """Receive a sealed compressed artifact before the sandbox is torn down."""
    started = time.monotonic()
    measurements.update(
        artifact_transferred_bytes=0.0,
        artifact_retries=0.0,
        artifact_metadata_retries=0.0,
        artifact_read_seconds=0.0,
        artifact_first_byte_seconds=0.0,
    )
    reason = "artifact_transfer"
    partial_path = spool.with_suffix(".partial")
    try:
        async with asyncio.timeout(_TRANSFER_TIMEOUT_SECONDS):
            open_stream = await reader_factory(environment, _TRANSFER_TIMEOUT_SECONDS)

            try:
                marker = await _small_file(
                    open_stream, SIDECAR_COMPLETE_PATH, 32, measurements
                )
            except (FileNotFoundError, ValueError):
                reason = "artifact_unsealed"
                raise
            if marker != b"complete\n":
                reason = "artifact_unsealed"
                raise ValueError("TITO sidecar did not publish a completion marker")
            payload = await _small_file(
                open_stream,
                SIDECAR_ARTIFACT_MANIFEST_PATH,
                MAX_MANIFEST_BYTES,
                measurements,
            )
            reason = "artifact_manifest"
            manifest = read_manifest(payload)
            if (
                manifest.get("trajectory_id") != trajectory_id
                or manifest.get("bundle_digest") != bundle_digest
            ):
                reason = "artifact_identity"
                raise ValueError(
                    "TITO sidecar artifact does not belong to the current trial"
                )

        for attempt in range(2):
            try:
                async with asyncio.timeout(_TRANSFER_TIMEOUT_SECONDS):
                    await _spool_artifact(
                        open_stream, partial_path, manifest, measurements
                    )
                partial_path.replace(spool)
                return spool, manifest
            except Exception as exc:
                if (
                    isinstance(exc, (ArtifactSourceUnavailable, FileNotFoundError))
                    or getattr(exc, "errno", None) in _LOCAL_STORAGE_ERRORS
                ):
                    reason = "artifact_transfer"
                    raise
                if attempt:
                    reason = (
                        "artifact_integrity"
                        if isinstance(exc, ValueError)
                        else "artifact_transfer"
                    )
                    raise
                measurements["artifact_retries"] += 1
    except Exception as exc:
        raise RecoverableRolloutError(
            f"TITO artifact collection failed ({type(exc).__name__}): {reason}",
            reason=reason,
        ) from exc
    finally:
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove TITO partial receipt before trial cleanup")
        measurements["artifact_collection_seconds"] = time.monotonic() - started
    raise AssertionError("artifact retry loop returned no result")
