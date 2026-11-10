"""Tests for batch_to_realtime.py against a local fake inference server."""

import http.server
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_to_realtime as b2r


class FakeServer:
    """Serves chat/completions and embeddings. Each row's prompt can script failures."""

    def __init__(self):
        self.requests: list[dict] = []
        self.attempts: dict[str, int] = {}
        self.lock = threading.Lock()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                key = json.dumps(
                    body.get("messages") or body.get("input"), sort_keys=True
                )
                with server.lock:
                    server.requests.append(
                        {
                            "path": self.path,
                            "body": body,
                            "auth": self.headers["Authorization"],
                        }
                    )
                    attempt = server.attempts.get(key, 0)
                    server.attempts[key] = attempt + 1
                text = key
                if "fail-twice-429" in text and attempt < 2:
                    return self._send(
                        429, {"error": "rate limited"}, {"Retry-After": "0"}
                    )
                if "fail-once-503" in text and attempt < 1:
                    return self._send(503, {"error": "replica preempted"})
                if "drop-once" in text and attempt < 1:
                    # Headers promise more bytes than are sent, then the socket
                    # closes: what a replica preempted mid-response looks like.
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "1000")
                    self.end_headers()
                    self.wfile.write(b'{"choices": [')
                    self.wfile.flush()
                    self.close_connection = True
                    return None
                if "scaling-up-thrice" in text and attempt < 3:
                    return self._send(
                        503,
                        {"error": {"code": "DEPLOYMENT_SCALING_UP"}},
                        {"Retry-After": "0"},
                    )
                if "always-scaling-up" in text:
                    return self._send(
                        503, {"error": "scaling up"}, {"Retry-After": "60"}
                    )
                if "status-530-once" in text and attempt < 1:
                    return self._send(530, {"error": "origin unreachable"})
                if "status-501" in text:
                    return self._send(501, {"error": "not implemented"})
                if "model-not-deployed" in text:
                    return self._send(
                        404,
                        {"error": {"code": "NOT_FOUND", "param": "model"}},
                    )
                if "forbidden-endpoint" in text:
                    return self._send(
                        403, {"error": {"message": "endpoint not in plan"}}
                    )
                if "chat-on-classifier" in text:
                    # The serving 404 for one row, unlike the gateway's model 404.
                    return self._send(
                        404,
                        {"error": {"message": "please use /embeddings API instead"}},
                    )
                if "unauthorized" in text:
                    return self._send(401, {"error": "invalid api key"})
                if "bad-request" in text:
                    return self._send(400, {"error": "invalid max_tokens"})
                if self.path.endswith("/embeddings"):
                    return self._send(
                        200,
                        {
                            "data": [{"embedding": [0.1, 0.2]}],
                            "usage": {"prompt_tokens": 1},
                        },
                    )
                return self._send(
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": f"echo:{body['model']}",
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    },
                )

            def _send(self, code, payload, headers=None):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/inference/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def chat(text):
    return [{"role": "user", "content": text}]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        f.writelines((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows)


def read_jsonl(path):
    """Parses complete rows, skipping a partial line left by an interrupted run."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def run(tmp_path, server, rows, *extra, model="accounts/fireworks/models/m"):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    if rows is not None:
        write_jsonl(inp, rows)
    args = b2r.parse_args(
        [
            "--input",
            str(inp),
            "--output",
            str(out),
            "--model",
            model,
            "--base-url",
            server.url,
            "--initial-backoff",
            "0",
            "--max-backoff",
            "0",
            "--progress-every",
            "0",
            *extra,
        ]
    )
    cfg = b2r.Config(
        base_url=args.base_url,
        api_key="test-key",
        model=args.model,
        timeout=10,
        max_retries=args.max_retries,
        initial_backoff=0,
        max_backoff=0,
        max_retry_time=args.max_retry_time,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    stats = b2r.run(args, cfg)
    return stats, read_jsonl(out), read_jsonl(str(out) + ".errors.jsonl")


def test_openai_format_rows(tmp_path, server):
    stats, out, err = run(
        tmp_path,
        server,
        [
            {"custom_id": "a", "body": {"messages": chat("hi"), "model": "ignored"}},
            {"custom_id": "e", "url": "/v1/embeddings", "body": {"input": "embed me"}},
        ],
    )
    assert (stats.succeeded, stats.failed) == (2, 0)
    by_id = {r["custom_id"]: r for r in out}
    assert (
        by_id["a"]["response"]["choices"][0]["message"]["content"]
        == "echo:accounts/fireworks/models/m"
    )
    assert by_id["e"]["response"]["data"][0]["embedding"] == [0.1, 0.2]
    assert err == []
    paths = sorted(r["path"] for r in server.requests)
    assert paths == ["/inference/v1/chat/completions", "/inference/v1/embeddings"]
    assert all(r["auth"] == "Bearer test-key" for r in server.requests)


def test_chat_format_row_keeps_extra_fields(tmp_path, server):
    stats, out, _ = run(
        tmp_path, server, [{"request_id": "r1", "messages": chat("q"), "label": "x"}]
    )
    assert stats.succeeded == 1
    row = out[0]
    assert row["request_id"] == "r1"
    assert row["label"] == "x"
    assert row["original_messages"] == chat("q")
    assert row["messages"][-1] == {
        "role": "assistant",
        "content": "echo:accounts/fireworks/models/m",
    }
    assert row["usage"]["total_tokens"] == 5


def test_deployment_model_is_sent_as_given(tmp_path, server):
    model = "accounts/fireworks/models/m#accounts/acme/deployments/abc123"
    run(
        tmp_path,
        server,
        [{"custom_id": "a", "body": {"messages": chat("hi")}}],
        model=model,
    )
    assert server.requests[0]["body"]["model"] == model


def test_retries_429_and_503(tmp_path, server):
    stats, out, err = run(
        tmp_path,
        server,
        [
            {"custom_id": "rl", "body": {"messages": chat("fail-twice-429")}},
            {"custom_id": "pre", "body": {"messages": chat("fail-once-503")}},
        ],
    )
    assert (stats.succeeded, stats.failed) == (2, 0)
    assert err == []
    assert {r["custom_id"] for r in out} == {"rl", "pre"}
    assert len(server.requests) == 5


def test_gives_up_after_max_retries(tmp_path, server):
    stats, _, err = run(
        tmp_path,
        server,
        [{"custom_id": "rl", "body": {"messages": chat("fail-twice-429")}}],
        "--max-retries",
        "1",
    )
    assert (stats.succeeded, stats.failed) == (0, 1)
    assert err == [
        {
            "custom_id": "rl",
            "error": {"code": "429", "message": '{"error": "rate limited"}'},
        }
    ]


def test_client_error_is_not_retried(tmp_path, server):
    stats, _, err = run(
        tmp_path, server, [{"request_id": "b", "messages": chat("bad-request")}]
    )
    assert stats.failed == 1
    assert len(server.requests) == 1
    assert err[0]["request_id"] == "b"
    assert err[0]["error"]["code"] == "400"


def test_bad_lines_and_duplicates_go_to_errors(tmp_path, server):
    stats, _, err = run(
        tmp_path,
        server,
        [
            "not json",
            {"neither": "format"},
            {"custom_id": "d", "body": {"messages": chat("one")}},
            {"custom_id": "d", "body": {"messages": chat("two")}},
            {"custom_id": "u", "url": "/v1/images", "body": {}},
        ],
    )
    assert stats.succeeded == 1
    codes = sorted(e["error"]["code"] for e in err)
    assert codes == [
        "duplicate_id",
        "invalid_json",
        "validation_error",
        "validation_error",
    ]


def test_job_defaults_fill_only_missing_fields(tmp_path, server):
    run(
        tmp_path,
        server,
        [
            {"custom_id": "set", "body": {"messages": chat("a"), "max_tokens": 5}},
            {"custom_id": "unset", "body": {"messages": chat("b"), "stream": True}},
        ],
        "--max-tokens",
        "100",
        "--temperature",
        "0.5",
    )
    bodies = {json.dumps(r["body"]["messages"]): r["body"] for r in server.requests}
    assert bodies[json.dumps(chat("a"))]["max_tokens"] == 5
    unset = bodies[json.dumps(chat("b"))]
    assert unset["max_tokens"] == 100 and unset["temperature"] == 0.5
    assert "stream" not in unset


def test_rerun_skips_done_rows_and_retries_failed_ones(tmp_path, server):
    # Like a Batch job continuation: only rows with a result are skipped.
    rows = [
        {"custom_id": "ok", "body": {"messages": chat("fine")}},
        {"custom_id": "bad", "body": {"messages": chat("bad-request")}},
    ]
    run(tmp_path, server, rows)
    assert len(server.requests) == 2

    stats, out, err = run(tmp_path, server, rows)
    assert stats.skipped == 1 and stats.failed == 1
    assert len(server.requests) == 3
    assert len(out) == 1
    assert len(err) == 1, "the retried row replaces its old error row"


def test_skip_errors_leaves_failed_rows_alone(tmp_path, server):
    rows = [
        {"custom_id": "ok", "body": {"messages": chat("fine")}},
        {"custom_id": "bad", "body": {"messages": chat("bad-request")}},
    ]
    run(tmp_path, server, rows)
    stats, _, err = run(tmp_path, server, rows, "--skip-errors")
    assert stats.skipped == 2
    assert (stats.failed, stats.outstanding) == (0, 1)
    assert len(server.requests) == 2
    assert len(err) == 1


def test_rerun_after_a_stop_sends_unfinished_and_failed_rows(tmp_path, server):
    # A run stopped midway leaves some rows unrecorded; the rerun sends those
    # and the failed ones, and skips the finished ones.
    rows = [
        {"custom_id": f"r{i}", "body": {"messages": chat(f"row {i}")}} for i in range(4)
    ]
    rows.append({"custom_id": "bad", "body": {"messages": chat("bad-request")}})
    out_path = tmp_path / "out.jsonl"
    out_path.write_text(
        json.dumps({"custom_id": "r0", "response": {}})
        + "\n"
        + json.dumps({"custom_id": "r1", "response": {}})
        + "\n"
    )
    (tmp_path / "out.jsonl.errors.jsonl").write_text(
        json.dumps({"custom_id": "bad", "error": {"code": "503", "message": "x"}})
        + "\n"
    )
    stats, out, err = run(tmp_path, server, rows)
    sent = sorted(
        json.loads(json.dumps(r["body"]["messages"]))[0]["content"]
        for r in server.requests
    )
    assert sent == ["bad-request", "row 2", "row 3"]
    assert stats.skipped == 2
    assert sorted(r["custom_id"] for r in out) == ["r0", "r1", "r2", "r3"]
    assert [e["error"]["code"] for e in err] == ["400"], "the old 503 row is replaced"


def test_concurrency_handles_many_rows(tmp_path, server):
    rows = [
        {"custom_id": str(i), "body": {"messages": chat(f"row {i}")}}
        for i in range(200)
    ]
    stats, out, _ = run(tmp_path, server, rows, "--concurrency", "8")
    assert stats.succeeded == 200
    assert {r["custom_id"] for r in out} == {str(i) for i in range(200)}


def test_main_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    write_jsonl(tmp_path / "in.jsonl", [])
    assert (
        b2r.main(
            [
                "--input",
                str(tmp_path / "in.jsonl"),
                "--output",
                str(tmp_path / "o.jsonl"),
                "--model",
                "m",
            ]
        )
        == 2
    )


def test_chat_row_matches_batch_shape(tmp_path, server):
    _, out, _ = run(tmp_path, server, [{"request_id": "r1", "messages": chat("q")}])
    row = out[0]
    assert row["rollout_status"] == {
        "code": 100,
        "message": "Rollout finished",
        "details": [],
    }
    assert row["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "prompt_token_details": {"cached_tokens": 0},
    }


def test_resume_after_partial_last_line(tmp_path, server):
    out_path = tmp_path / "out.jsonl"
    out_path.write_text(
        json.dumps({"custom_id": "a", "response": {}})
        + "\n"
        + '{"custom_id": "b", "resp'
    )
    stats, out, _ = run(
        tmp_path,
        server,
        [
            {"custom_id": "a", "body": {"messages": chat("one")}},
            {"custom_id": "b", "body": {"messages": chat("two")}},
        ],
    )
    assert (stats.skipped, stats.succeeded) == (1, 1)
    assert [r["custom_id"] for r in out] == ["a", "b"], (
        "b is readable, not glued to the fragment"
    )


def test_rerun_does_not_duplicate_invalid_line_errors(tmp_path, server):
    rows = ["not json"]
    run(tmp_path, server, rows)
    stats, _, err = run(tmp_path, server, rows)
    assert stats.failed == 1
    assert len(err) == 1, "the line's old error row is replaced"


def test_skipped_earlier_failures_are_outstanding(tmp_path, server):
    rows = [{"custom_id": "bad", "body": {"messages": chat("bad-request")}}]
    run(tmp_path, server, rows)
    stats, _, _ = run(tmp_path, server, rows, "--skip-errors")
    assert (stats.failed, stats.outstanding) == (0, 1)


def test_chat_row_with_custom_id_resumes_by_request_id(tmp_path, server):
    rows = [{"request_id": "r1", "custom_id": "other", "messages": chat("q")}]
    run(tmp_path, server, rows)
    stats, out, _ = run(tmp_path, server, rows)
    assert stats.skipped == 1
    assert len(out) == 1


def test_unparseable_retry_after_falls_back_to_backoff():
    assert b2r.retry_after_seconds({"Retry-After": "soon"}) is None


def make_cfg(server, **overrides):
    values = dict(
        base_url=server.url,
        api_key="test-key",
        model="m",
        timeout=10,
        max_retries=30,
        initial_backoff=1,
        max_backoff=60,
        max_retry_time=1800,
    )
    values.update(overrides)
    return b2r.Config(**values)


def test_connection_dropped_mid_response_is_retried(tmp_path, server):
    stats, out, err = run(
        tmp_path, server, [{"custom_id": "d", "body": {"messages": chat("drop-once")}}]
    )
    assert (stats.succeeded, stats.failed, stats.retries) == (1, 0, 1)
    assert err == []


def test_deployment_scaling_up_then_recovering_succeeds(tmp_path, server):
    stats, out, _ = run(
        tmp_path,
        server,
        [{"custom_id": "p", "body": {"messages": chat("scaling-up-thrice")}}],
    )
    assert (stats.succeeded, stats.retries) == (1, 3)


def test_unlisted_5xx_is_retried(tmp_path, server):
    stats, _, _ = run(
        tmp_path,
        server,
        [{"custom_id": "x", "body": {"messages": chat("status-530-once")}}],
    )
    assert (stats.succeeded, stats.retries) == (1, 1)


def test_501_is_not_retried(tmp_path, server):
    stats, _, err = run(
        tmp_path, server, [{"custom_id": "x", "body": {"messages": chat("status-501")}}]
    )
    assert (stats.failed, stats.retries) == (1, 0)
    assert err[0]["error"]["code"] == "501"


def test_unauthorized_stops_the_run_without_recording_rows(tmp_path, server):
    rows = [
        {"custom_id": str(i), "body": {"messages": chat(f"unauthorized {i}")}}
        for i in range(20)
    ]
    stats, out, err = run(tmp_path, server, rows, "--concurrency", "1")
    assert stats.fatal and "401" in stats.fatal
    assert out == [] and err == [], "unsent rows stay unrecorded so a rerun sends them"
    assert len(server.requests) < 20


def test_unexpected_error_in_one_row_does_not_abort_the_run(
    tmp_path, server, monkeypatch
):
    real = b2r.format_success

    def flaky(job, result):
        if job.row_id == "boom":
            raise RuntimeError("bug")
        return real(job, result)

    monkeypatch.setattr(b2r, "format_success", flaky)
    stats, out, err = run(
        tmp_path,
        server,
        [
            {"custom_id": "boom", "body": {"messages": chat("a")}},
            {"custom_id": "fine", "body": {"messages": chat("b")}},
        ],
    )
    assert (stats.succeeded, stats.failed) == (1, 1)
    assert err[0]["error"]["code"] == "client_error"


def test_retry_time_budget_stops_retrying(server, monkeypatch):
    monkeypatch.setattr(
        b2r.time, "sleep", lambda s: pytest.fail("should give up before sleeping")
    )
    cfg = make_cfg(server, max_retry_time=1)
    with pytest.raises(b2r.RequestError) as e:
        b2r.post_json(cfg, "chat/completions", {"messages": chat("always-scaling-up")})
    assert e.value.code == "503"


def test_retry_after_is_honored_up_to_max_backoff(server):
    cfg = make_cfg(server)
    assert b2r.backoff_seconds(cfg, 0, 60.0) == 60, "jitter never exceeds --max-backoff"
    assert 10 <= b2r.backoff_seconds(cfg, 0, 10.0) <= 11
    assert 10 <= b2r.backoff_seconds(make_cfg(server, max_backoff=10), 0, 60.0) <= 11


def test_exponential_backoff_grows_with_a_floor(server):
    cfg = make_cfg(server)
    for attempt, step in [(0, 1), (3, 8), (10, 60)]:
        assert step / 2 <= b2r.backoff_seconds(cfg, attempt, None) <= step


def test_main_exits_2_when_the_run_stops(tmp_path, server, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "bad")
    write_jsonl(
        tmp_path / "in.jsonl",
        [{"custom_id": "a", "body": {"messages": chat("unauthorized")}}],
    )
    code = b2r.main(
        [
            "--input",
            str(tmp_path / "in.jsonl"),
            "--output",
            str(tmp_path / "o.jsonl"),
            "--model",
            "m",
            "--base-url",
            server.url,
            "--progress-every",
            "0",
        ]
    )
    assert code == 2


def test_chat_row_index_skips_blank_lines(tmp_path, server):
    # Batch numbers rows without a request_id by their index among non-blank
    # lines, so a blank line must not shift the ids.
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        json.dumps({"messages": chat("first")})
        + "\n\n"
        + json.dumps({"messages": chat("second")})
        + "\n"
    )
    _, out, errors = run(tmp_path, server, None)
    assert errors == []
    assert sorted(r["request_id"] for r in out) == ["0", "1"]


def test_ids_keep_their_input_type(tmp_path, server):
    # Batch writes the id as given; an int request_id stays an int so
    # downstream joins on it keep working.
    rows = [
        {"request_id": 7, "messages": chat("hi")},
        {"request_id": 8, "messages": chat("bad-request")},
        {"custom_id": 9, "body": {"messages": chat("hi")}},
    ]
    _, out, errors = run(tmp_path, server, rows)
    assert {(r.get("request_id"), r.get("custom_id")) for r in out} == {
        (None, 9),
        (7, None),
    }
    assert errors[0]["request_id"] == 8
    # And resume still recognizes them.
    stats, _, _ = run(tmp_path, server, None, "--skip-errors")
    assert (stats.succeeded, stats.skipped, stats.outstanding) == (0, 3, 1)


def test_row_with_custom_id_but_no_body_keeps_line_id(tmp_path, server):
    stats, _, errors = run(tmp_path, server, [{"custom_id": "x"}])
    assert errors == [
        {
            "custom_id": "line-1",
            "error": {
                "code": "validation_error",
                "message": "row is missing custom_id or body",
            },
        }
    ]
    # The rerun checks the row again and replaces its old error row.
    stats, _, errors = run(tmp_path, server, None)
    assert stats.failed == 1 and len(errors) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGINT")
@pytest.mark.parametrize("signame, code", [("SIGINT", 130), ("SIGTERM", 143)])
def test_stop_signal_stops_promptly_and_leaves_rows_unrecorded(
    tmp_path, server, signame, code
):
    # Ctrl-C, or SIGTERM from a job scheduler.
    import signal
    import subprocess
    import time

    rows = [
        {"custom_id": f"ok{i}", "body": {"messages": chat(f"ok{i}")}} for i in range(3)
    ]
    # These rows retry for minutes (503 with Retry-After: 60), like a
    # preempted deployment. Ctrl-C must not wait for them.
    rows += [
        {"custom_id": f"wait{i}", "body": {"messages": chat(f"always-scaling-up {i}")}}
        for i in range(20)
    ]
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(inp, rows)
    cmd = [
        sys.executable,
        b2r.__file__,
        "--input",
        str(inp),
        "--output",
        str(out),
        "--model",
        "m",
        "--base-url",
        server.url,
        "--concurrency",
        "4",
        "--progress-every",
        "0",
    ]
    env = {**os.environ, "FIREWORKS_API_KEY": "test-key"}
    p = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 20
    while len(read_jsonl(out)) < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)  # let the slow rows start retrying
    p.send_signal(getattr(signal, signame))
    started = time.monotonic()
    _, stderr = p.communicate(timeout=10)
    assert time.monotonic() - started < 5
    assert p.returncode == code
    assert "Rerun the same command" in stderr
    assert "Traceback" not in stderr
    assert sorted(r["custom_id"] for r in read_jsonl(out)) == ["ok0", "ok1", "ok2"]
    assert read_jsonl(str(out) + ".errors.jsonl") == []


def test_model_not_found_stops_the_run_without_recording_rows(
    tmp_path, server, monkeypatch
):
    # A deployment answers 404 until it is READY. Every row would fail the same
    # way, so the run stops and the rows stay unrecorded for the rerun.
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    rows = [{"custom_id": "a", "body": {"messages": chat("model-not-deployed")}}]
    rows += [{"custom_id": f"b{i}", "body": {"messages": chat("hi")}} for i in range(5)]
    write_jsonl(tmp_path / "in.jsonl", rows)
    out = tmp_path / "o.jsonl"
    code = b2r.main(
        [
            "--input",
            str(tmp_path / "in.jsonl"),
            "--output",
            str(out),
            "--model",
            "m",
            "--base-url",
            server.url,
            "--concurrency",
            "1",
            "--progress-every",
            "0",
        ]
    )
    assert code == 2
    assert read_jsonl(str(out) + ".errors.jsonl") == []


def test_403_is_a_row_error_and_the_run_continues(tmp_path, server):
    stats, out, err = run(
        tmp_path,
        server,
        [
            {"custom_id": "denied", "body": {"messages": chat("forbidden-endpoint")}},
            {"custom_id": "ok", "body": {"messages": chat("fine")}},
        ],
        "--concurrency",
        "1",
    )
    assert stats.fatal is None
    assert (stats.succeeded, stats.failed) == (1, 1)
    assert err[0]["error"]["code"] == "403"


def test_row_specific_404_is_a_row_error_and_the_run_continues(tmp_path, server):
    stats, out, err = run(
        tmp_path,
        server,
        [
            {"custom_id": "chat", "body": {"messages": chat("chat-on-classifier")}},
            {"custom_id": "emb", "url": "/v1/embeddings", "body": {"input": "x"}},
        ],
        "--concurrency",
        "1",
    )
    assert stats.fatal is None
    assert [r["custom_id"] for r in out] == ["emb"]
    assert err[0]["error"]["code"] == "404"


def test_is_model_unavailable_only_for_the_gateway_model_404():
    assert b2r.is_model_unavailable(
        404, '{"error": {"code": "NOT_FOUND", "param": "model"}}'
    )
    assert not b2r.is_model_unavailable(
        404, '{"error": {"message": "use /embeddings"}}'
    )
    assert not b2r.is_model_unavailable(404, "not json")
    assert not b2r.is_model_unavailable(403, '{"error": {"param": "model"}}')


def test_rerun_keeps_failures_for_rows_missing_from_the_input(tmp_path, server):
    both = [
        {"custom_id": "a", "body": {"messages": chat("bad-request a")}},
        {"custom_id": "b", "body": {"messages": chat("bad-request b")}},
    ]
    run(tmp_path, server, both)
    # The input is cut down to row a, then retried.
    stats, _, err = run(tmp_path, server, both[:1])
    assert stats.outstanding == 1, "b still has no result"
    assert sorted(e["custom_id"] for e in err) == ["a", "b"]
    assert len(err) == 2, "a's old error row was replaced, not duplicated"


def test_retry_timeout_is_cut_to_the_remaining_budget(server, monkeypatch):
    timeouts = []
    real = b2r.urllib.request.urlopen

    def recording(request, timeout):
        timeouts.append(timeout)
        return real(request, timeout=timeout)

    monkeypatch.setattr(b2r.urllib.request, "urlopen", recording)
    cfg = make_cfg(server, timeout=600, max_retry_time=5, initial_backoff=0)
    b2r.post_json(
        cfg, "chat/completions", {"model": "m", "messages": chat("scaling-up-thrice")}
    )
    assert len(timeouts) == 4
    assert timeouts[0] == 600
    assert all(t <= 5 for t in timeouts[1:])


def test_progress_counts_rows_rejected_before_sending(tmp_path, server, capsys):
    run(tmp_path, server, ["not json", "also not json"], "--progress-every", "1")
    assert "2 done (0 ok, 2 failed" in capsys.readouterr().err


def test_concurrency_is_capped_on_serverless():
    base = ["--input", "i", "--output", "o"]
    with pytest.raises(SystemExit):
        b2r.parse_args(
            [*base, "--model", "accounts/fireworks/models/m", "--concurrency", "33"]
        )
    assert (
        b2r.parse_args(
            [*base, "--model", "accounts/fireworks/models/m", "--concurrency", "32"]
        ).concurrency
        == 32
    )


def test_concurrency_is_capped_higher_on_a_deployment():
    base = ["--input", "i", "--output", "o"]
    deployment = "accounts/fireworks/models/m#accounts/acme/deployments/d1"
    assert (
        b2r.parse_args(
            [*base, "--model", deployment, "--concurrency", "256"]
        ).concurrency
        == 256
    )
    with pytest.raises(SystemExit):
        b2r.parse_args([*base, "--model", deployment, "--concurrency", "257"])


def test_a_429_pauses_every_worker(server, monkeypatch):
    throttle = b2r.Throttle()
    cfg = make_cfg(server, throttle=throttle)
    monkeypatch.setattr(b2r.time, "sleep", lambda s: None)
    b2r.post_json(
        cfg, "chat/completions", {"model": "m", "messages": chat("fail-twice-429")}
    )
    paused = []
    monkeypatch.setattr(throttle, "_until", b2r.time.monotonic() + 5)
    monkeypatch.setattr(b2r.time, "sleep", paused.append)
    monkeypatch.setattr(
        b2r.time,
        "monotonic",
        lambda: throttle._until if paused else throttle._until - 5,
    )
    throttle.wait()
    assert paused and paused[0] == pytest.approx(5)


def test_throttle_pause_only_extends():
    throttle = b2r.Throttle()
    throttle.pause(10)
    later = throttle._until
    throttle.pause(1)
    assert throttle._until == later


def test_direct_deployment_names_count_as_deployments():
    assert b2r.is_deployment("accounts/acme/deployments/d1")
    assert b2r.is_deployment("accounts/fireworks/models/m#accounts/acme/deployments/d1")
    assert b2r.is_deployment("accounts/acme/deployedModels/m-abc")
    assert not b2r.is_deployment("accounts/fireworks/models/gpt-oss-120b")


def test_stop_during_startup_exits_cleanly(tmp_path, monkeypatch, capsys):
    # A SIGTERM or Ctrl-C while earlier results are still being read must take
    # the normal interrupted exit, not escape as a traceback.
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

    def interrupted_run(args, cfg):
        raise KeyboardInterrupt

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(b2r, "run", interrupted_run)
    monkeypatch.setattr(b2r.os, "_exit", fake_exit)
    with pytest.raises(SystemExit) as e:
        b2r.main(["--input", "i", "--output", str(tmp_path / "o"), "--model", "m"])
    assert e.value.code == 130
    assert "Rerun the same command" in capsys.readouterr().err


def read_jsonl_strict(path):
    """Parses every nonblank line, failing on any invalid one."""
    with open(path, "rb") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("which", ["output", "errors"])
@pytest.mark.parametrize(
    "fragment",
    [b'{"custom_id": "b", "resp', '{"custom_id": "b", "response": "你'.encode()[:-1]],
    ids=["ascii", "utf8"],
)
def test_torn_last_line_is_repaired_before_resume(tmp_path, server, which, fragment):
    out = tmp_path / "out.jsonl"
    target = out if which == "output" else tmp_path / "out.jsonl.errors.jsonl"
    complete = json.dumps({"custom_id": "a", "response": {}}).encode() + b"\n"
    if which == "errors":
        out.write_bytes(complete)
        complete = (
            json.dumps(
                {"custom_id": "z", "error": {"code": "400", "message": "x"}}
            ).encode()
            + b"\n"
        )
    target.write_bytes(complete + fragment)
    rows = [
        {"custom_id": "a", "body": {"messages": chat("one")}},
        {"custom_id": "b", "body": {"messages": chat("two")}},
    ]
    stats, _, _ = run(tmp_path, server, rows)
    assert stats.succeeded == 1
    for path in (out, tmp_path / "out.jsonl.errors.jsonl"):
        read_jsonl_strict(path)  # every remaining line parses
    assert sorted(r["custom_id"] for r in read_jsonl_strict(out)) == ["a", "b"]


def test_complete_last_row_without_newline_is_kept(tmp_path, server):
    out = tmp_path / "out.jsonl"
    out.write_bytes(json.dumps({"custom_id": "a", "response": {}}).encode())
    stats, _, _ = run(
        tmp_path,
        server,
        [
            {"custom_id": "a", "body": {"messages": chat("one")}},
            {"custom_id": "b", "body": {"messages": chat("two")}},
        ],
    )
    assert (stats.skipped, stats.succeeded) == (1, 1)
    assert [r["custom_id"] for r in read_jsonl_strict(out)] == ["a", "b"]


def test_unreadable_middle_line_is_reported(tmp_path, server, capsys):
    out = tmp_path / "out.jsonl"
    out.write_bytes(
        b"garbage\n" + json.dumps({"custom_id": "a", "response": {}}).encode() + b"\n"
    )
    run(tmp_path, server, [{"custom_id": "a", "body": {"messages": chat("one")}}])
    assert "not valid JSON rows" in capsys.readouterr().err


def test_fatal_error_stops_rows_that_are_mid_retry(tmp_path, server):
    # One worker gets a 401 while the other is retrying a 503 with a long
    # Retry-After. The run must stop promptly, not wait out the retries.
    import time

    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(
        inp,
        [
            {"custom_id": "slow", "body": {"messages": chat("always-scaling-up")}},
            {"custom_id": "bad", "body": {"messages": chat("unauthorized")}},
        ],
    )
    args = b2r.parse_args(
        [
            "--input",
            str(inp),
            "--output",
            str(out),
            "--model",
            "m",
            "--concurrency",
            "2",
            "--progress-every",
            "0",
        ]
    )
    cfg = make_cfg(server, throttle=b2r.Throttle())
    started = time.monotonic()
    stats = b2r.run(args, cfg)
    assert time.monotonic() - started < 2
    assert stats.fatal and "401" in stats.fatal
    assert read_jsonl(out) == [] and read_jsonl(str(out) + ".errors.jsonl") == []
