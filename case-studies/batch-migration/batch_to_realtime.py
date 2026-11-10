#!/usr/bin/env python3
"""Run a Fireworks Batch input file against serverless or a deployment.

Reads a Batch input JSONL file, sends each row to the Fireworks inference API
with bounded concurrency and retries, and writes results as JSONL. Use it while
the Batch API is disabled, against a serverless model or your own (for example
preemptible) deployment.

Input rows use the OpenAI batch format: {"custom_id": "...", "body": {...}}
with an optional "url" such as "/v1/chat/completions" or "/v1/embeddings".

Output rows follow the Batch output format. Successful rows go to
--output and failed rows to --errors. Rows are written as they finish, so the
output order can differ from the input order; match rows by custom_id.

Reruns work like a Batch job continuation: rerun the same command and rows
whose id is already in --output are skipped, while failed rows (in --errors)
and rows a stopped run never finished are sent again. Pass --skip-errors to
leave failed rows alone.

Only the Python standard library is used. Set FIREWORKS_API_KEY first.

Examples:

    # Serverless
    python batch_to_realtime.py --input batch.jsonl --output results.jsonl \\
        --model accounts/fireworks/models/gpt-oss-120b

    # Your own deployment, such as a preemptible one
    python batch_to_realtime.py --input batch.jsonl --output results.jsonl \\
        --model "accounts/fireworks/models/gpt-oss-120b#accounts/<ACCOUNT>/deployments/<DEPLOYMENT_ID>" \\
        --concurrency 64

--concurrency is capped at 32 for serverless, which is shared with other
customers, and at 256 for your own deployment. When the API answers 429 or
503, every worker pauses, not only the row that got the error.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import email.utils
import http.client
import itertools
import json
import os
import random
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from collections.abc import Iterator

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Transient 4xx statuses. Every 5xx except these two is retried: they mean the
# request itself can never succeed.
RETRYABLE_4XX = frozenset({408, 409, 425, 429})
NON_RETRYABLE_5XX = frozenset({501, 505})
# A rejected API key (401) fails every row the same way, so the run stops.
# So does the gateway's 404 for a model or deployment that isn't available,
# which a deployment still being created answers until it is READY. That 404
# names the model as the bad parameter; any other 404 or 403 can be specific to
# one row (an endpoint the model or plan doesn't allow) and is recorded as that
# row's error.
FATAL_STATUS = frozenset({401})


def is_model_unavailable(status: int, body: str) -> bool:
    """True for the gateway's 404 that says the model or deployment can't serve."""
    if status != 404:
        return False
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return False
    return isinstance(error, dict) and error.get("param") == "model"


# A deployment with no running replica (for example a preemptible one that was
# just preempted) answers 503 DEPLOYMENT_SCALING_UP with Retry-After: 60 until
# a replica is back, which can take several minutes. The defaults keep a row
# retrying for up to 30 minutes so it survives that gap.
DEFAULT_MAX_RETRIES = 30
DEFAULT_MAX_RETRY_TIME = 1800.0
DEFAULT_MAX_BACKOFF = 60.0
RETRY_LOG_INTERVAL = 30.0

# Serverless capacity is shared with other customers, so keep a single run
# from flooding it; a dedicated deployment is only the customer's own traffic.
DEFAULT_CONCURRENCY = 16
MAX_SERVERLESS_CONCURRENCY = 32
MAX_DEPLOYMENT_CONCURRENCY = 256
# Statuses that mean the server as a whole is overloaded or not serving. They
# pause every worker, not only the row that got them.
SHARED_BACKOFF_STATUS = frozenset({429, 503})


def is_deployment(model: str) -> bool:
    """True when --model routes to the caller's own deployment, not serverless."""
    return "#" in model or "/deployedModels/" in model or "/deployments/" in model


class Throttle:
    """A pause shared by all workers. After a 429 or 503, no worker starts a
    request until the pause is over, so the run backs off as a whole."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until = 0.0

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._until = max(self._until, time.monotonic() + seconds)

    def wait(self, stop: threading.Event | None = None) -> None:
        while True:
            with self._lock:
                remaining = self._until - time.monotonic()
            if remaining <= 0:
                return
            if stop is None:
                time.sleep(remaining)
            elif stop.wait(remaining):
                raise Stopped


def is_retryable_status(code: int) -> bool:
    return code in RETRYABLE_4XX or (
        500 <= code < 600 and code not in NON_RETRYABLE_5XX
    )


ENDPOINTS = {
    "/v1/chat/completions": "chat/completions",
    "/v1/completions": "completions",
    "/v1/embeddings": "embeddings",
}


class RequestError(Exception):
    """A request that failed for good, with a Batch-style error code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class FatalError(Exception):
    """An error that every row would hit, such as a rejected API key."""


class Stopped(Exception):
    """The run is stopping; leave this row unrecorded so a rerun sends it."""


@dataclass
class Config:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_retries: int
    initial_backoff: float
    max_backoff: float
    max_retry_time: float = DEFAULT_MAX_RETRY_TIME
    max_tokens: int | None = None
    temperature: float | None = None
    throttle: Throttle | None = None


@dataclass
class Job:
    line_number: int
    row: dict[str, Any]
    row_id: str
    is_openai_format: bool
    endpoint: str
    payload: dict[str, Any]


def row_id_of(row: dict[str, Any], row_index: int) -> tuple[str, bool]:
    """Returns the row's id and whether it uses the OpenAI batch format.
    row_index is the zero-based index among non-blank lines."""
    if "custom_id" in row and "body" in row:
        return str(row["custom_id"]), True
    if "messages" in row:
        # Batch falls back to the row index (blank lines not counted) when
        # request_id is missing.
        return str(row.get("request_id", row_index)), False
    raise ValueError("row is missing custom_id or body")


def build_job(
    row: dict[str, Any], line_number: int, row_index: int, cfg: Config
) -> Job:
    row_id, is_openai = row_id_of(row, row_index)
    if is_openai:
        payload = dict(row["body"])
        url = row.get("url") or (
            "/v1/embeddings" if "input" in payload else "/v1/chat/completions"
        )
        endpoint = ENDPOINTS.get(url)
        if endpoint is None:
            raise ValueError(
                f"unsupported url {url!r}; expected one of {', '.join(ENDPOINTS)}"
            )
    else:
        payload = {"messages": row["messages"]}
        if "tools" in row:
            payload["tools"] = row["tools"]
        endpoint = "chat/completions"
    payload["model"] = cfg.model
    # Like Batch, job-level settings only fill in what the row leaves unset.
    if endpoint != "embeddings":
        if cfg.max_tokens is not None:
            payload.setdefault("max_tokens", cfg.max_tokens)
        if cfg.temperature is not None:
            payload.setdefault("temperature", cfg.temperature)
    payload.pop("stream", None)
    return Job(line_number, row, row_id, is_openai, endpoint, payload)


def retry_after_seconds(headers: Any) -> float | None:
    value = headers.get("Retry-After") if headers is not None else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None  # unparseable; fall back to exponential backoff
    return max(0.0, parsed.timestamp() - time.time()) if parsed else None


def read_error_body(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode(errors="replace")[:2000]
    except Exception:  # the connection can drop while reading the error body
        return ""


def backoff_seconds(cfg: Config, attempt: int, retry_after: float | None) -> float:
    """Wait before the next attempt: the server's Retry-After when given,
    otherwise exponential backoff. Jitter spreads out the in-flight rows so they
    don't retry in lockstep, and equal jitter keeps every wait at least half
    the backoff, so retries really do slow down."""
    if retry_after is not None:
        # Never sooner than the server asked, never longer than --max-backoff.
        return min(cfg.max_backoff, retry_after + random.uniform(0, retry_after * 0.1))
    step = min(cfg.max_backoff, cfg.initial_backoff * 2**attempt)
    return step / 2 + random.uniform(0, step / 2)


def post_json(
    cfg: Config,
    endpoint: str,
    payload: dict[str, Any],
    on_retry=None,
    stop: threading.Event | None = None,
) -> dict[str, Any]:
    """POSTs payload, retrying transient failures until --max-retries or
    --max-retry-time runs out. Raises RequestError when it gives up on the row
    and FatalError when no row can succeed. on_retry(error) runs before each
    retry. When stop is set, it raises Stopped instead of starting or waiting
    for another attempt."""
    url = f"{cfg.base_url.rstrip('/')}/{endpoint}"
    data = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
        "User-Agent": "fireworks-batch-migration/1.0",
    }
    started = time.monotonic()
    for attempt in itertools.count():
        retry_after: float | None = None
        # A retry gets only what is left of --max-retry-time, so a slow attempt
        # can't run past it. The first attempt always gets the full --timeout.
        timeout = cfg.timeout
        if attempt > 0:
            timeout = min(timeout, cfg.max_retry_time - (time.monotonic() - started))
            if timeout <= 0:
                break
        if stop is not None and stop.is_set():
            raise Stopped
        if cfg.throttle is not None:
            cfg.throttle.wait(stop)
        try:
            request = urllib.request.Request(
                url, data=data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            body = read_error_body(e)
            if e.code in FATAL_STATUS or is_model_unavailable(e.code, body):
                raise FatalError(f"HTTP {e.code}: {body or e.reason}") from None
            last = RequestError(str(e.code), body or str(e.reason))
            if not is_retryable_status(e.code):
                raise last from None
            retry_after = retry_after_seconds(e.headers)
            if cfg.throttle is not None and e.code in SHARED_BACKOFF_STATUS:
                cfg.throttle.pause(backoff_seconds(cfg, attempt, retry_after))
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            # Includes timeouts, resets and IncompleteRead: a replica that is
            # preempted mid-response drops the connection.
            last = RequestError(
                "connection_error", f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            )
        except ValueError as e:  # not JSON, or not UTF-8
            last = RequestError("invalid_response", f"response was not JSON: {e}")
        if attempt >= cfg.max_retries:
            break
        wait = backoff_seconds(cfg, attempt, retry_after)
        if time.monotonic() - started + wait > cfg.max_retry_time:
            break
        if on_retry is not None:
            on_retry(last)
        if stop is None:
            time.sleep(wait)
        elif stop.wait(wait):
            raise Stopped
    raise last


def raw_id(row: dict[str, Any], key: str, row_id: str) -> Any:
    """Returns row[key] as written in the input (Batch keeps an int id an
    int), or row_id when the row's id came from somewhere else."""
    value = row.get(key)
    return value if value is not None and str(value) == row_id else row_id


def format_success(job: Job, result: dict[str, Any]) -> dict[str, Any]:
    if job.is_openai_format:
        return {
            "custom_id": raw_id(job.row, "custom_id", job.row_id),
            "response": result,
        }
    out = {k: v for k, v in job.row.items() if k != "messages"}
    original = job.row["messages"]
    out["original_messages"] = original
    out["request_id"] = raw_id(job.row, "request_id", job.row_id)
    try:
        message = result["choices"][0]["message"]
        out["responses"] = message
        out["messages"] = original + [message]
    except (KeyError, IndexError, TypeError):
        out["responses"] = result
        out["messages"] = original
    # Match Batch's chat rows (finalize_chat_output_row in the Batch runner).
    usage = result.get("usage") or {}
    out["rollout_status"] = {"code": 100, "message": "Rollout finished", "details": []}
    out["usage"] = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "prompt_token_details": {
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0
            ),
        },
    }
    return out


def format_error(
    row: dict[str, Any], row_id: str, is_openai: bool, code: str, message: str
) -> dict[str, Any]:
    error = {"code": code, "message": message}
    if is_openai:
        return {"custom_id": raw_id(row, "custom_id", row_id), "error": error}
    out = {
        "request_id": raw_id(row, "request_id", row_id),
        "messages": row.get("messages", []),
        "error": error,
    }
    if "tools" in row:
        out["tools"] = row["tools"]
    return out


def output_row_id(row: dict[str, Any]) -> str | None:
    """Returns the id of an output or error row. Chat rows always carry
    request_id and may also copy a custom_id field from the input, so
    request_id wins."""
    row_id = row.get("request_id", row.get("custom_id"))
    return None if row_id is None else str(row_id)


def repair_tail(path: str) -> None:
    """Fixes the last line of a results file after an interrupted run, so the
    file stays valid JSONL for strict readers. A complete final row that only
    lacks its newline gets one; an incomplete fragment (including one that
    ends mid UTF-8 character) is cut off, and its row is sent again."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        size = f.seek(0, os.SEEK_END)
        if size == 0:
            return
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return
        # Find the start of the last line, reading backwards in chunks.
        start, pos = 0, size
        while pos > 0:
            step = min(pos, 1 << 16)
            pos -= step
            f.seek(pos)
            chunk = f.read(step)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                start = pos + newline + 1
                break
        f.seek(start)
        tail = f.read()
        try:
            json.loads(tail)
        except ValueError:
            f.truncate(start)
            print(
                f"Removed an incomplete last line from {path} (left by an interrupted run); "
                "its row will be sent again.",
                file=sys.stderr,
            )
            return
        f.seek(size)
        f.write(b"\n")


def read_ids(path: str) -> set[str]:
    """Returns the ids of the rows in a results file. A line that can't be
    read is reported, and its row is treated as having no result."""
    ids: set[str] = set()
    if not os.path.exists(path):
        return ids
    unreadable = 0
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row_id = output_row_id(row)
            except (ValueError, AttributeError):
                unreadable += 1
                continue
            if row_id is not None:
                ids.add(row_id)
    if unreadable:
        print(
            f"Warning: {unreadable} line(s) in {path} are not valid JSON rows and were "
            "ignored; rows without a readable result will be sent again.",
            file=sys.stderr,
        )
    return ids


def drop_old_ids(path: str, before: int, ids: set[str]) -> None:
    """Atomically rewrites path without the rows in its first `before` bytes
    whose id is in ids. Rows written after that offset are kept as they are."""
    if not ids or not os.path.exists(path):
        return
    with open(path, "rb") as f:
        old, new = f.read(before), f.read()
    kept: list[bytes] = []
    for line in old.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if output_row_id(row) not in ids:
            kept.append(line if line.endswith(b"\n") else line + b"\n")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.writelines(kept)
        f.write(new)
    os.replace(tmp, path)


def iter_rows(
    path: str,
) -> Iterator[tuple[int, int, dict[str, Any] | None, str | None]]:
    """Yields (line_number, row_index, row, parse_error) for each non-empty
    line. row_index counts non-empty lines from zero, as Batch does."""
    row_index = -1
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row_index += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not a JSON object")
                yield line_number, row_index, row, None
            except ValueError as e:
                yield line_number, row_index, None, str(e)


class Writer:
    """Appends JSONL rows from many threads, flushing each one so a crash loses at most the line in flight."""

    def __init__(self, path: str):
        # run() repairs a torn last line first, so appends start on a new line.
        self._f = open(path, "a")
        self._lock = threading.Lock()
        # Byte offset where this run's rows start, and the ids it wrote.
        self.start = self._f.tell()
        self.written_ids: set[str] = set()

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            if self._f.closed:
                return  # interrupted: the row stays unrecorded and is resent
            self._f.write(line)
            self._f.flush()
            row_id = output_row_id(row)
            if row_id is not None:
                self.written_ids.add(row_id)

    def close(self) -> None:
        with self._lock:
            self._f.close()


@dataclass
class Stats:
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    # Rows that failed in an earlier run and still have no result: skipped with
    # --skip-errors, or no longer in the input.
    outstanding: int = 0
    retries: int = 0
    # Set when the run stopped early because no row could succeed.
    fatal: str | None = None
    # Set when the run was stopped by Ctrl-C or SIGTERM.
    interrupted: bool = False


def run(args: argparse.Namespace, cfg: Config) -> Stats:
    errors_path = args.errors or args.output + ".errors.jsonl"
    repair_tail(args.output)
    repair_tail(errors_path)
    done = read_ids(args.output)
    failed_before = read_ids(errors_path)
    unresolved = failed_before - done
    # Like a Batch continuation, a rerun sends every row without a result:
    # failed rows and rows a stopped run never finished. An old error row is
    # removed only once its row has a new result (see the end of this
    # function), so failures whose rows aren't in this input stay recorded.
    if args.skip_errors:
        done |= failed_before

    out, err = Writer(args.output), Writer(errors_path)
    stats = Stats()
    lock = threading.Lock()
    started = time.time()
    stop = threading.Event()
    last_retry_log = [0.0]

    def count(ok: bool) -> None:
        """Counts a finished row and prints progress every --progress-every rows."""
        with lock:
            if ok:
                stats.succeeded += 1
            else:
                stats.failed += 1
            finished = stats.succeeded + stats.failed
        if args.progress_every and finished % args.progress_every == 0:
            rate = finished / max(time.time() - started, 1e-9)
            print(
                f"{finished} done ({stats.succeeded} ok, {stats.failed} failed, "
                f"{stats.retries} retries), {rate:.1f} rows/s",
                file=sys.stderr,
            )

    def on_retry(e: RequestError) -> None:
        with lock:
            stats.retries += 1
            now = time.time()
            if now - last_retry_log[0] < RETRY_LOG_INTERVAL:
                return
            last_retry_log[0] = now
        print(f"Retrying after {e.code}: {e.message[:200]}", file=sys.stderr)

    def process(job: Job) -> None:
        if stop.is_set():
            return  # left unrecorded, so a rerun sends it
        try:
            result = post_json(cfg, job.endpoint, job.payload, on_retry, stop)
            out.write(format_success(job, result))
            ok = True
        except Stopped:
            return  # left unrecorded, so a rerun sends it
        except FatalError as e:
            with lock:
                stats.fatal = stats.fatal or str(e)
            stop.set()
            return
        except RequestError as e:
            err.write(
                format_error(
                    job.row, job.row_id, job.is_openai_format, e.code, e.message
                )
            )
            ok = False
        except Exception as e:  # never let one row abort the whole run
            err.write(
                format_error(
                    job.row, job.row_id, job.is_openai_format, "client_error", repr(e)
                )
            )
            ok = False
        count(ok)

    # Ids of rows sent this run, for duplicate detection. This is the one
    # structure that grows with the input: a few dozen bytes per row.
    seen: set[str] = set()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        pending: set[concurrent.futures.Future] = set()
        for line_number, row_index, row, parse_error in iter_rows(args.input):
            if stop.is_set():
                break
            if row is None:
                if f"line-{line_number}" in done:
                    stats.skipped += 1
                    continue
                err.write(
                    format_error(
                        {},
                        f"line-{line_number}",
                        True,
                        "invalid_json",
                        parse_error or "",
                    )
                )
                count(False)
                continue
            try:
                job = build_job(row, line_number, row_index, cfg)
            except (ValueError, KeyError, TypeError) as e:
                try:
                    row_id, is_openai = row_id_of(row, row_index)
                except ValueError:
                    row_id, is_openai = f"line-{line_number}", True
                if row_id not in done:
                    err.write(
                        format_error(row, row_id, is_openai, "validation_error", str(e))
                    )
                    count(False)
                continue
            if job.row_id in done:
                stats.skipped += 1
                continue
            if job.row_id in seen:
                err.write(
                    format_error(
                        row,
                        job.row_id,
                        job.is_openai_format,
                        "duplicate_id",
                        f"id {job.row_id!r} already appears earlier in the input",
                    )
                )
                count(False)
                continue
            seen.add(job.row_id)
            # Bound in-flight rows (and their payloads) to a few per worker.
            if len(pending) >= args.concurrency * 4:
                finished, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in finished:
                    f.result()
            pending.add(pool.submit(process, job))
        for f in concurrent.futures.as_completed(pending):
            f.result()
    except KeyboardInterrupt:
        # Stop now instead of working through queued rows or waiting out
        # in-flight retries. Rows not yet written stay unrecorded, so a rerun
        # sends them.
        stop.set()
        stats.interrupted = True
    finally:
        pool.shutdown(wait=not stats.interrupted, cancel_futures=True)
        # Rows that finish after this are dropped, not written to a closed file.
        out.close()
        err.close()
        recorded = out.written_ids | err.written_ids
        if not args.skip_errors:
            drop_old_ids(errors_path, err.start, unresolved & recorded)
        # Earlier failures that still have no new result in this run.
        stats.outstanding = len(unresolved - recorded)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Send a Fireworks Batch input JSONL file to serverless or a deployment and write JSONL results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Batch input JSONL file.")
    p.add_argument(
        "--output",
        required=True,
        help="JSONL file for successful rows. Appended to on resume.",
    )
    p.add_argument(
        "--errors",
        help="JSONL file for failed rows. Defaults to <output>.errors.jsonl.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model to call, e.g. accounts/fireworks/models/gpt-oss-120b for serverless, or "
        "'<model>#accounts/<account>/deployments/<id>' for your own deployment.",
    )
    p.add_argument(
        "--base-url", default=os.environ.get("FIREWORKS_BASE_URL", DEFAULT_BASE_URL)
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Requests in flight at once. At most {MAX_SERVERLESS_CONCURRENCY} on "
        f"serverless and {MAX_DEPLOYMENT_CONCURRENCY} on your own deployment.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Most retries per row for 408, 409, 425, 429, 5xx (except 501 and 505), "
        "timeouts and dropped connections.",
    )
    p.add_argument(
        "--initial-backoff",
        type=float,
        default=1.0,
        help="First retry wait, in seconds.",
    )
    p.add_argument(
        "--max-backoff",
        type=float,
        default=DEFAULT_MAX_BACKOFF,
        help="Longest wait between retries, in seconds. A Retry-After from the server is used up to this.",
    )
    p.add_argument(
        "--max-retry-time",
        type=float,
        default=DEFAULT_MAX_RETRY_TIME,
        help="Longest time to keep retrying one row, in seconds. The default rides out a "
        "preemptible deployment rescheduling its replica.",
    )
    p.add_argument(
        "--timeout", type=float, default=600.0, help="Per-request timeout, in seconds."
    )
    p.add_argument(
        "--max-tokens", type=int, help="Default max_tokens for rows that don't set it."
    )
    p.add_argument(
        "--temperature",
        type=float,
        help="Default temperature for rows that don't set it.",
    )
    p.add_argument(
        "--skip-errors",
        action="store_true",
        help="Don't resend rows that failed in an earlier run. By default a rerun "
        "retries them, like a Batch job continuation.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N rows; 0 disables it.",
    )
    args = p.parse_args(argv)
    if args.concurrency < 1:
        p.error("--concurrency must be at least 1")
    limit = (
        MAX_DEPLOYMENT_CONCURRENCY
        if is_deployment(args.model)
        else MAX_SERVERLESS_CONCURRENCY
    )
    if args.concurrency > limit:
        target = (
            "your own deployment"
            if is_deployment(args.model)
            else "serverless, which is shared"
        )
        p.error(f"--concurrency is at most {limit} on {target}")
    if args.max_retries < 0:
        p.error("--max-retries must be 0 or more")
    if args.max_retry_time < 0:
        p.error("--max-retry-time must be 0 or more")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Set FIREWORKS_API_KEY first.", file=sys.stderr)
        return 2
    cfg = Config(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        initial_backoff=args.initial_backoff,
        max_backoff=args.max_backoff,
        max_retry_time=args.max_retry_time,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        throttle=Throttle(),
    )
    # A job scheduler stops the script with SIGTERM; treat it like Ctrl-C so
    # unfinished rows stay unrecorded and the rerun sends them.
    stopped_by = [signal.SIGINT]

    def on_sigterm(signum, frame):
        stopped_by[0] = signum
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        stats = run(args, cfg)
    except KeyboardInterrupt:
        # Stopped before run() reached its own handler, for example while
        # reading earlier results on a rerun. Nothing new was sent yet.
        stats = Stats(interrupted=True)
    errors_path = args.errors or args.output + ".errors.jsonl"
    if stats.interrupted:
        print(
            f"Interrupted: {stats.succeeded} succeeded, {stats.failed} failed this run. "
            "Rerun the same command to continue.",
            file=sys.stderr,
            flush=True,
        )
        # Don't wait for requests still in flight; their rows are resent on rerun.
        os._exit(128 + stopped_by[0])
    if stats.fatal:
        print(
            f"Stopped: {stats.fatal}. Check FIREWORKS_API_KEY and --model. If --model is a "
            "deployment you just created, wait until it is READY. Rows not yet sent "
            "are untouched; rerun the same command to continue.",
            file=sys.stderr,
        )
        return 2
    print(
        f"Finished: {stats.succeeded} succeeded, {stats.failed} failed, {stats.skipped} skipped (already done). "
        f"{stats.outstanding} rows still failed from earlier runs and were not resent. "
        f"{stats.retries} retries. "
        f"Results: {args.output}. Errors: {errors_path}.",
        file=sys.stderr,
    )
    # Exit 1 while any row lacks a result, including earlier failures.
    return 1 if stats.failed or stats.outstanding else 0


if __name__ == "__main__":
    sys.exit(main())
