"""Regression coverage for a sidecar lost during a live Harbor agent run."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.examples.harbor_rl.tito import sidecar


def _paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sidecar, "SIDECAR_PID_PATH", str(tmp_path / "sidecar.pid"))
    monkeypatch.setattr(
        sidecar, "SIDECAR_EXIT_PATH", str(tmp_path / "sidecar-exit.json")
    )
    monkeypatch.setattr(
        sidecar, "SIDECAR_EXIT_DETAIL_PATH", str(tmp_path / "sidecar-exit-detail.json")
    )
    monkeypatch.setattr(
        sidecar, "SIDECAR_STDERR_PATH", str(tmp_path / "sidecar.stderr")
    )
    monkeypatch.setattr(
        sidecar, "SIDECAR_ENDPOINT_PATH", str(tmp_path / "endpoint.json")
    )


def test_supervisor_retains_killed_child_diagnostics_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _paths(monkeypatch, tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"api_key": "inference-secret"}))
    Path(sidecar.SIDECAR_ENDPOINT_PATH).write_text(
        json.dumps({"api_key": "local-secret"})
    )
    Path(sidecar.SIDECAR_STDERR_PATH).write_text(
        "Traceback: model connection closed\n"
        "Authorization: Bearer local-secret\n"
        "api_key=inference-secret\n"
        "sampled private completion\n"
    )
    memory = iter(
        (
            {"events_oom_kill": 4, "peak_bytes": 100},
            {"events_oom_kill": 5, "peak_bytes": 200},
        )
    )
    monkeypatch.setattr(sidecar, "_cgroup_memory_snapshot", lambda: next(memory))
    original_popen = subprocess.Popen
    monkeypatch.setattr(
        sidecar.subprocess,
        "Popen",
        lambda _command, **kwargs: original_popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        ),
    )

    errors: list[BaseException] = []

    def kill_child() -> None:
        deadline = time.monotonic() + 5
        while (
            not Path(sidecar.SIDECAR_PID_PATH).exists() and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        try:
            pid = int(Path(sidecar.SIDECAR_PID_PATH).read_text().strip())
            os.kill(pid, signal.SIGKILL)
        except BaseException as exc:
            errors.append(exc)

    killer = threading.Thread(target=kill_child, daemon=True)
    killer.start()
    try:
        code = sidecar.supervise(spec_path)
    finally:
        killer.join(timeout=5)
    assert not errors
    assert code == 128 + signal.SIGKILL
    record = json.loads(Path(sidecar.SIDECAR_EXIT_PATH).read_text())
    assert record["exit_code"] == -signal.SIGKILL
    assert record["signal"] == "SIGKILL"
    assert record["child_pid"] > 0
    assert record["supervisor_pid"] == os.getpid()
    assert record["started_at"] and record["exited_at"]
    assert record["cgroup_memory"]["delta"]["events_oom_kill"] == 1
    assert "stderr_tail" not in record
    assert "sampled private completion" not in json.dumps(record)
    detail = json.loads(Path(sidecar.SIDECAR_EXIT_DETAIL_PATH).read_text())
    assert "Traceback: model connection closed" in detail["stderr_tail"]
    assert "inference-secret" not in detail["stderr_tail"]
    assert "local-secret" not in detail["stderr_tail"]
    assert "<redacted>" in detail["stderr_tail"]


def test_unreadable_credential_source_suppresses_stderr_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _paths(monkeypatch, tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{malformed")
    Path(sidecar.SIDECAR_STDERR_PATH).write_text("unrecognized-secret\n")
    assert sidecar._redacted_stderr_tail(spec_path) == (
        "<unavailable: credential source unreadable>"
    )


def test_reward_only_diagnostic_keeps_parser_reason_without_sampled_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sidecar, "SIDECAR_DIAGNOSTIC_PATH", str(tmp_path / "diagnostic.json")
    )
    sidecar._write_sidecar_diagnostic(
        SimpleNamespace(
            status="failed",
            metadata={
                "tito_failure": {
                    "code": "tito_sdk_parse_error",
                    "origin": "sdk_parser",
                    "parser_reason_code": "native_parse_rejected",
                    "parser_error_type": "TITOParseError",
                    "completion_tokens": 42,
                    "message": "sampled text and private context",
                    "trace_path": "/logs/tito/failures/raw.jsonl",
                    "renderer_id": "unsafe value with sampled text",
                }
            },
        )
    )
    encoded = Path(sidecar.SIDECAR_DIAGNOSTIC_PATH).read_text()
    record = json.loads(encoded)
    assert record["status"] == "failed"
    assert record["failure"] == {
        "code": "tito_sdk_parse_error",
        "origin": "sdk_parser",
        "parser_reason_code": "native_parse_rejected",
        "parser_error_type": "TITOParseError",
        "completion_tokens": 42,
    }
    assert "sampled text" not in encoded
    assert "trace_path" not in encoded


def test_probe_distinguishes_reachable_http_error_from_lost_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _paths(monkeypatch, tmp_path)
    Path(sidecar.SIDECAR_PID_PATH).write_text(f"{os.getpid()}\n")

    class Handler(BaseHTTPRequestHandler):
        status = 200

        def do_GET(self) -> None:
            assert self.headers.get("Authorization") == "Bearer local-secret"
            self.send_response(type(self).status)
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        Path(sidecar.SIDECAR_ENDPOINT_PATH).write_text(
            json.dumps(
                {
                    "openai_base_url": (
                        f"http://127.0.0.1:{server.server_port}/trajectories/t/v1"
                    ),
                    "api_key": "local-secret",
                }
            )
        )
        assert sidecar.probe_sidecar() == {"state": "healthy"}
        Handler.status = 401
        assert sidecar.probe_sidecar() == {"state": "http_error"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert sidecar.probe_sidecar() == {"state": "unavailable"}


def test_live_harness_stops_on_proven_sidecar_exit_without_terminalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Environment:
        async def exec(self, *, command: str, cwd: str) -> SimpleNamespace:
            assert "tito.sidecar probe" in command
            assert cwd == "/"
            return SimpleNamespace(
                return_code=0,
                stdout=json.dumps(
                    {
                        "state": "exited",
                        "exit_code": -9,
                        "signal": "SIGKILL",
                        "oom_kill_delta": 1,
                    }
                ),
            )

    async def harness() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            calls.append("harness_cancelled")

    async def quiesce(_environment: object, *, process_pattern: str) -> None:
        assert process_pattern == "[o]pencode --model="
        calls.append("quiesced")

    async def terminalize(*_args: object, **_kwargs: object) -> None:
        calls.append("terminalized")

    monkeypatch.setattr(sidecar, "quiesce_harness_process", quiesce)
    monkeypatch.setattr(sidecar, "terminalize_sidecar", terminalize)
    monkeypatch.setattr(sidecar, "_SIDECAR_MONITOR_POLL_SECONDS", 0)

    async def run() -> None:
        with pytest.raises(sidecar.TITOSidecarInfrastructureError, match="SIGKILL"):
            await sidecar.run_harness_with_sidecar(
                Environment(), harness(), process_pattern="[o]pencode --model="
            )
        await asyncio.sleep(0)

    asyncio.run(run())
    assert sorted(calls) == ["harness_cancelled", "quiesced"]


def test_sidecar_loss_keeps_typed_error_when_quiesce_transport_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Environment:
        async def exec(self, *, command: str, cwd: str) -> SimpleNamespace:
            assert "tito.sidecar probe" in command and cwd == "/"
            return SimpleNamespace(
                return_code=0,
                stdout='{"state":"exited","exit_code":-9,"signal":"SIGKILL"}',
            )

    async def harness() -> None:
        await asyncio.Event().wait()

    release = asyncio.Event()

    async def quiesce(_environment: object, *, process_pattern: str) -> None:
        assert process_pattern == "[o]pencode --model="
        try:
            await release.wait()
        except asyncio.CancelledError:
            # The E2B control request can outlive cancellation.
            await release.wait()

    monkeypatch.setattr(sidecar, "quiesce_harness_process", quiesce)
    monkeypatch.setattr(sidecar, "_SIDECAR_MONITOR_POLL_SECONDS", 0)
    monkeypatch.setattr(sidecar, "_SIDECAR_QUIESCE_TIMEOUT_SECONDS", 0.001)

    async def run() -> None:
        with pytest.raises(sidecar.TITOSidecarInfrastructureError) as failure:
            await sidecar.run_harness_with_sidecar(
                Environment(), harness(), process_pattern="[o]pencode --model="
            )
        assert any(
            "quiesce" in note and "exceeded" in note for note in failure.value.__notes__
        )
        release.set()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_http_error_keeps_agent_failure_and_normal_terminalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Environment:
        async def exec(self, *, command: str, cwd: str) -> SimpleNamespace:
            assert "tito.sidecar probe" in command
            assert cwd == "/"
            return SimpleNamespace(return_code=0, stdout='{"state": "http_error"}')

    async def harness() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("model rejected request")

    async def disposition(_environment: object) -> None:
        return None

    async def terminalize(
        _environment: object, *, status: str, reason: str | None = None
    ) -> None:
        calls.append(f"{status}:{reason}")

    monkeypatch.setattr(sidecar, "sidecar_failure_disposition", disposition)
    monkeypatch.setattr(sidecar, "terminalize_sidecar", terminalize)
    with pytest.raises(RuntimeError, match="model rejected request"):
        asyncio.run(
            sidecar.run_harness_with_sidecar(
                Environment(), harness(), process_pattern="[o]pencode --model="
            )
        )
    assert calls == ["failed:RuntimeError: model rejected request"]


def test_endpoint_loss_requires_consecutive_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter(
        (
            "unavailable",
            "unavailable",
            "http_error",
            "unavailable",
            "unavailable",
            "unavailable",
        )
    )
    count = 0

    class Environment:
        async def exec(self, *, command: str, cwd: str) -> SimpleNamespace:
            nonlocal count
            assert "tito.sidecar probe" in command
            assert cwd == "/"
            count += 1
            return SimpleNamespace(
                return_code=0, stdout=json.dumps({"state": next(states)})
            )

    monkeypatch.setattr(sidecar, "_SIDECAR_MONITOR_POLL_SECONDS", 0)
    monkeypatch.setattr(sidecar, "_SIDECAR_UNAVAILABLE_PROBES", 3)
    with pytest.raises(sidecar.TITOSidecarInfrastructureError, match="loopback"):
        asyncio.run(sidecar._monitor_sidecar(Environment()))
    assert count == 6


def test_isolated_sidecar_monitor_probes_sidecar_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Environment:
        async def exec(self, *, command: str, cwd: str) -> SimpleNamespace:
            raise AssertionError("sidecar probe reached main service")

        async def sidecar_exec(self, *, command: str, cwd: str) -> SimpleNamespace:
            assert "tito.sidecar probe" in command and cwd == "/"
            calls.append(command)
            return SimpleNamespace(
                return_code=0,
                stdout=json.dumps(
                    {"state": "exited", "exit_code": -9, "signal": "SIGKILL"}
                ),
            )

    monkeypatch.setattr(sidecar, "_SIDECAR_MONITOR_POLL_SECONDS", 0)
    with pytest.raises(sidecar.TITOSidecarInfrastructureError, match="exited"):
        asyncio.run(sidecar._monitor_sidecar(Environment()))
    assert len(calls) == 1


def test_monitor_never_overlaps_probes_when_provider_ignores_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sidecar, "_SIDECAR_MONITOR_POLL_SECONDS", 0.001)
    monkeypatch.setattr(sidecar, "_SIDECAR_PROBE_EXEC_TIMEOUT_SECONDS", 0.001)

    async def run() -> None:
        gate = asyncio.Event()
        started = 0
        active = 0
        peak_active = 0

        class Environment:
            async def exec(self, *, command: str, cwd: str) -> SimpleNamespace:
                nonlocal started, active, peak_active
                assert "tito.sidecar probe" in command and cwd == "/"
                started += 1
                active += 1
                peak_active = max(peak_active, active)
                try:
                    try:
                        await gate.wait()
                    except asyncio.CancelledError:
                        # Simulate an E2B provider request that ignores
                        # cancellation while its remote stream is pending.
                        await gate.wait()
                    return SimpleNamespace(return_code=0, stdout='{"state": "healthy"}')
                finally:
                    active -= 1

        monitor = asyncio.create_task(sidecar._monitor_sidecar(Environment()))
        await asyncio.sleep(0.03)
        assert started == 1
        assert peak_active == 1
        gate.set()
        await asyncio.sleep(0.02)
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        assert started > 1
        assert peak_active == 1
        assert active == 0

    asyncio.run(run())
