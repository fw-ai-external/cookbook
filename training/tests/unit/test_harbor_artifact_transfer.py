"""Exercise provider streaming and the publication boundary before Harbor teardown."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from fireworks.training.sdk import TITOMetricSummary, TITOTrajectoryArtifact

from training.examples.harbor_rl.sandboxes.daytona.artifacts import artifact_reader as daytona_reader
from training.examples.harbor_rl.sandboxes.e2b.artifacts import artifact_reader as e2b_reader
from training.examples.harbor_rl.tito._artifact_transfer import (
    collect_trajectory,
    decode_artifact,
)
from training.examples.harbor_rl.tito.sidecar import (
    SIDECAR_ARTIFACT_MANIFEST_PATH,
    SIDECAR_ARTIFACT_PATH,
    SIDECAR_COMPLETE_PATH,
)
from training.utils.rl.async_rl.errors import (
    DiscardTrajectoryError,
    RecoverableRolloutError,
)


class _DaytonaNotFoundError(Exception):
    pass


@pytest.fixture(autouse=True)
def _stub_daytona_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests exercise the adapter with a fake Daytona filesystem.
    monkeypatch.setitem(
        sys.modules,
        "daytona.common.errors",
        SimpleNamespace(DaytonaNotFoundError=_DaytonaNotFoundError),
    )


def _artifact_files():
    artifact = TITOTrajectoryArtifact(
        trajectory_id="current-attempt",
        serving_affinity_key_hash="affinity",
        metadata={"text": "雪" * 16384},
        status="completed",
        terminal_reason=None,
        segments=(),
        calls=(),
        response_attempts=(),
        metrics=TITOMetricSummary(counters={}, distributions={}),
        started_at=1,
        finished_at=2,
    )
    packed = artifact.pack()
    return artifact, {
        SIDECAR_COMPLETE_PATH: b"complete\n",
        SIDECAR_ARTIFACT_MANIFEST_PATH: json.dumps(
            {
                "schema_version": 1,
                "bundle_digest": "expected-bundle",
                "trajectory_id": "current-attempt",
                "status": "completed",
                "terminal_reason": None,
                "bytes": len(packed),
                "sha256": hashlib.sha256(packed).hexdigest(),
            }
        ).encode(),
        SIDECAR_ARTIFACT_PATH: packed,
    }


class Files:
    def __init__(self, files, fault=None):
        self.files = files
        self.fault = fault
        self.calls = []
        self.fault_entered = asyncio.Event()
        self.fault_closed = asyncio.Event()

    async def download_file_stream(self, path, **_kwargs):
        self.calls.append(path)
        try:
            payload = self.files[path]
        except KeyError as exc:
            raise _DaytonaNotFoundError(path) from exc

        async def stream():
            try:
                for offset in range(0, len(payload), 31):
                    if (
                        self.fault == "truncate_once"
                        and path == SIDECAR_ARTIFACT_PATH
                        and self.calls.count(path) == 1
                        and offset >= 31
                    ):
                        return
                    if self.fault == "stall" and path == SIDECAR_ARTIFACT_PATH:
                        self.fault_entered.set()
                        await asyncio.Future()
                    yield payload[offset : offset + 31]
            finally:
                if path == SIDECAR_ARTIFACT_PATH:
                    self.fault_closed.set()

        return stream()


async def _receive(files, measurements, tmp_path):
    return await collect_trajectory(
        SimpleNamespace(_sandbox=SimpleNamespace(fs=files)),
        daytona_reader,
        trajectory_id="current-attempt",
        bundle_digest="expected-bundle",
        spool=tmp_path / "trajectory.tito",
        measurements=measurements,
    )


def test_verified_retry_reuses_the_manifest_and_only_reopens_the_artifact(tmp_path):
    async def run():
        artifact, payloads = _artifact_files()
        files = Files(payloads, fault="truncate_once")
        measurements = {}
        path, manifest = await _receive(files, measurements, tmp_path)
        assert path.read_bytes() == payloads[SIDECAR_ARTIFACT_PATH]
        assert decode_artifact([path.read_bytes()], manifest) == artifact
        assert not path.with_suffix(".partial").exists()
        assert manifest["trajectory_id"] == artifact.trajectory_id
        assert files.calls.count(SIDECAR_ARTIFACT_PATH) == 2
        assert files.calls.count(SIDECAR_ARTIFACT_MANIFEST_PATH) == 1
        assert measurements["artifact_retries"] == 1
        assert (
            measurements["artifact_transferred_bytes"]
            == len(payloads[SIDECAR_ARTIFACT_PATH]) + 31
        )

    asyncio.run(run())


@pytest.mark.parametrize("exists", [True, False])
def test_missing_completion_marker_never_starts_the_large_read(tmp_path, exists):
    async def run():
        _, payloads = _artifact_files()
        if exists:
            payloads[SIDECAR_COMPLETE_PATH] = b""
        else:
            del payloads[SIDECAR_COMPLETE_PATH]
        files = Files(payloads)
        with pytest.raises(RecoverableRolloutError) as error:
            await _receive(files, {}, tmp_path)
        assert error.value.reason == "artifact_unsealed"
        assert files.calls == [SIDECAR_COMPLETE_PATH]

    asyncio.run(run())


def test_cancelling_a_stalled_read_closes_the_provider_stream(tmp_path):
    async def run():
        _, payloads = _artifact_files()
        files = Files(payloads, fault="stall")
        receiver = asyncio.create_task(_receive(files, {}, tmp_path))
        await asyncio.wait_for(files.fault_entered.wait(), timeout=5)
        receiver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(receiver, timeout=5)
        assert files.fault_closed.is_set()
        assert list(tmp_path.iterdir()) == []

    asyncio.run(run())


@pytest.mark.parametrize(
    ("path", "transform", "reason", "attempts"),
    [
        (SIDECAR_ARTIFACT_MANIFEST_PATH, lambda raw: b"{}", "artifact_manifest", 0),
        (
            SIDECAR_ARTIFACT_MANIFEST_PATH,
            lambda raw: raw.replace(b"current-attempt", b"other-attempt"),
            "artifact_identity",
            0,
        ),
        (SIDECAR_ARTIFACT_PATH, lambda raw: raw[:-1], "artifact_transfer", 2),
        (
            SIDECAR_ARTIFACT_PATH,
            lambda raw: raw[:-1] + bytes([raw[-1] ^ 1]),
            "artifact_integrity",
            2,
        ),
    ],
)
def test_bad_receipts_never_publish_a_committed_file(
    tmp_path, path, transform, reason, attempts
):
    async def run():
        _, payloads = _artifact_files()
        files = Files(payloads | {path: transform(payloads[path])})
        with pytest.raises(RecoverableRolloutError) as error:
            await _receive(files, {}, tmp_path)
        assert error.value.reason == reason
        assert files.calls.count(SIDECAR_ARTIFACT_PATH) == attempts
        assert list(tmp_path.iterdir()) == []

    asyncio.run(run())


def test_current_cookbook_loads_an_artifact_with_published_sdk_decoder(monkeypatch):
    artifact, payloads = _artifact_files()
    chunks = [payloads[SIDECAR_ARTIFACT_PATH]]
    manifest = json.loads(payloads[SIDECAR_ARTIFACT_MANIFEST_PATH])
    monkeypatch.delattr(TITOTrajectoryArtifact, "unpack_stream")
    assert decode_artifact(chunks, manifest) == artifact
    with pytest.raises(ValueError, match="checksum mismatch"):
        decode_artifact([chunks[0][:-1] + bytes([chunks[0][-1] ^ 1])], manifest)


def test_running_materializer_drains_before_cancellation_releases_workspace(tmp_path):
    from threading import Event

    from training.examples.harbor_rl.tito._artifact_io import run_artifact_task

    async def run():
        entered, release = Event(), Event()
        marker = tmp_path / "input"
        marker.write_text("test")

        def materializer():
            entered.set()
            release.wait(5)
            assert marker.read_text() == "test"

        task = asyncio.create_task(run_artifact_task(materializer))
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        marker.unlink()

    asyncio.run(run())


def _read_materialized_trajectory_id(outcome):
    return outcome.trajectory_artifact.trajectory_id


def test_stream_receipt_survives_verification_and_process_materialization(tmp_path):
    from training.examples.harbor_rl.tito._artifact_io import ArtifactProcessPool
    from training.examples.harbor_rl.tito.trial import (
        HarborTrialOutcome,
        _finish_harbor_trial,
    )

    async def run():
        artifact, payloads = _artifact_files()
        received = await _receive(Files(payloads), {}, tmp_path)
        processor = ArtifactProcessPool()
        try:
            outcome = await processor.run(
                _finish_harbor_trial,
                HarborTrialOutcome("example", "run", tmp_path, None, {}, None, None),
                raw_rewards={"reward": 1},
                reward_key="reward",
                terminal_failure_reward=None,
                retry_names=frozenset(),
                retryable_e2b_timeout=False,
                retryable_sidecar_readiness=False,
                materializer=_read_materialized_trajectory_id,
                received_artifact=received,
                stream_compact=True,
            )
            assert outcome.rollout == artifact.trajectory_id
            assert outcome.trajectory_artifact is None
            assert outcome.reward == 1
            assert outcome.phase_timings["artifact_codec_seconds"] > 0
        finally:
            await processor.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "exception", [None, "AgentTimeoutError", "NonZeroAgentExitCodeError"]
)
def test_missing_stream_hook_preserves_the_original_agent_error(tmp_path, exception):
    from training.examples.harbor_rl.tito.trial import (
        HarborTrialOutcome,
        _finish_harbor_trial,
    )

    expected = {
        None: "artifact_unsealed",
        "AgentTimeoutError": "AgentTimeoutError",
    }
    error_type = RecoverableRolloutError if exception in expected else DiscardTrajectoryError
    with pytest.raises(error_type) as error:
        _finish_harbor_trial(
            HarborTrialOutcome("example", "run", tmp_path, None, {}, exception, None),
            raw_rewards=None,
            reward_key="reward",
            terminal_failure_reward=None,
            retry_names=frozenset({"AgentTimeoutError"}),
            retryable_e2b_timeout=False,
            retryable_sidecar_readiness=False,
            materializer=lambda outcome: pytest.fail(
                "unverified artifact reached materializer"
            ),
            stream_compact=True,
        )
    if exception in expected:
        assert error.value.reason == expected[exception]
    else:
        assert error.value.diagnostics["final_error_kind"] == exception


@pytest.mark.parametrize(
    ("transform", "limit", "message"),
    [
        (lambda data: data + b"x", 1024 * 1024, "trailing bytes"),
        (lambda data: data[:-1], 1024 * 1024, "incomplete"),
        (lambda data: data, 16, "expanded size"),
    ],
)
def test_old_sdk_still_requires_bounded_compression(
    monkeypatch, transform, limit, message
):
    from training.examples.harbor_rl.tito import _artifact_transfer

    _, payloads = _artifact_files()
    encoded = transform(payloads[SIDECAR_ARTIFACT_PATH])
    manifest = json.loads(payloads[SIDECAR_ARTIFACT_MANIFEST_PATH])
    manifest.update(bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())
    monkeypatch.delattr(TITOTrajectoryArtifact, "unpack_stream")
    monkeypatch.setattr(_artifact_transfer, "MAX_EXPANDED_ARTIFACT_BYTES", limit)
    with pytest.raises(ValueError, match=message):
        decode_artifact([encoded], manifest)


def test_retries_a_stalled_manifest_and_payload_separately(monkeypatch, tmp_path):
    from training.examples.harbor_rl.tito import _artifact_transfer

    class StallOnce(Files):
        async def download_file_stream(self, path, **kwargs):
            stream = await super().download_file_stream(path, **kwargs)
            if (
                path in (SIDECAR_ARTIFACT_MANIFEST_PATH, SIDECAR_ARTIFACT_PATH)
                and self.calls.count(path) == 1
            ):

                async def stalled():
                    await asyncio.Future()
                    yield b""

                return stalled()
            return stream

    async def run():
        _, payloads = _artifact_files()
        files = StallOnce(payloads)
        timings = {}
        path, _ = await _receive(files, timings, tmp_path)
        assert path.read_bytes() == payloads[SIDECAR_ARTIFACT_PATH]
        assert timings["artifact_metadata_retries"] == 1
        assert timings["artifact_retries"] == 1
        assert timings["artifact_read_seconds"] >= 0.045
        assert files.calls.count(SIDECAR_COMPLETE_PATH) == 1
        assert list(tmp_path.iterdir()) == [path]

    monkeypatch.setattr(_artifact_transfer, "_READ_IDLE_TIMEOUT_SECONDS", 0.05)
    asyncio.run(run())


def test_local_disk_full_does_not_reopen_remote_file(monkeypatch, tmp_path):
    import errno
    from pathlib import Path

    open_path = Path.open

    def disk_full(path, *args, **kwargs):
        if path.suffix == ".partial":
            raise OSError(errno.ENOSPC, "full")
        return open_path(path, *args, **kwargs)

    async def run():
        _, payloads = _artifact_files()
        files = Files(payloads)
        timings = {}
        with pytest.raises(RecoverableRolloutError) as failure:
            await _receive(files, timings, tmp_path)
        assert failure.value.reason == "artifact_transfer"
        assert timings["artifact_retries"] == 0
        assert SIDECAR_ARTIFACT_PATH not in files.calls
        assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(Path, "open", disk_full)
    asyncio.run(run())


def test_verified_receipt_has_no_second_hash_pass(monkeypatch, tmp_path):
    async def run():
        artifact, payloads = _artifact_files()
        path, manifest = await _receive(Files(payloads), {}, tmp_path)

        def unexpected_hash(*args, **kwargs):
            raise AssertionError("the verified spool must not be hashed again")

        with monkeypatch.context() as patch:
            patch.setattr(hashlib, "sha256", unexpected_hash)
            assert (
                decode_artifact([path.read_bytes()], manifest, verified=True)
                == artifact
            )

    asyncio.run(run())


@pytest.mark.parametrize("sandbox_lost", [False, True])
def test_e2b_stream_uses_shared_retry_and_abandons_a_gone_sandbox(
    tmp_path, sandbox_lost
):
    e2b = pytest.importorskip("e2b.exceptions")

    class E2BFiles(Files):
        async def read(self, path, *, format, request_timeout):
            assert format == "stream" and request_timeout > 0
            if sandbox_lost and path == SIDECAR_ARTIFACT_PATH:
                self.calls.append(path)
                raise e2b.SandboxNotFoundException("sandbox gone")
            return await self.download_file_stream(path)

    async def run():
        artifact, payloads = _artifact_files()
        files = E2BFiles(payloads, fault="truncate_once")
        environment = SimpleNamespace(_sandbox=SimpleNamespace(files=files))
        timings = {}
        receive = collect_trajectory(
            environment,
            e2b_reader,
            trajectory_id="current-attempt",
            bundle_digest="expected-bundle",
            spool=tmp_path / "trajectory.tito",
            measurements=timings,
        )
        if sandbox_lost:
            with pytest.raises(RecoverableRolloutError) as error:
                await receive
            assert error.value.reason == "artifact_transfer"
            assert list(tmp_path.iterdir()) == []
        else:
            path, manifest = await receive
            assert (
                decode_artifact([path.read_bytes()], manifest, verified=True)
                == artifact
            )
        assert files.calls.count(SIDECAR_ARTIFACT_MANIFEST_PATH) == 1
        assert files.calls.count(SIDECAR_ARTIFACT_PATH) == (1 if sandbox_lost else 2)
        assert timings["artifact_retries"] == (0 if sandbox_lost else 1)

    asyncio.run(run())
