# Run Batch input files on serverless or a deployment

The Fireworks Batch API is temporarily disabled while it is rebuilt. Until it is
back, [`batch_to_realtime.py`](batch_to_realtime.py) runs your existing Batch
input files against the regular inference API. You can point it at a
serverless model or at your own deployment, such as a
[preemptible deployment](https://docs.fireworks.ai/guides/ondemand-deployments#preemptible-deployments-eval-/-batch).

The script:

- reads a Batch input JSONL file in the OpenAI batch format;
- sends rows in parallel, retrying rate limits, server errors, timeouts and
  dropped connections for up to 30 minutes per row, so it rides out a
  preemptible deployment losing its replica;
- writes successful rows to one JSONL file and failed rows to another, in the
  matching Batch output format;
- reruns like a Batch job continuation: rerunning the same command sends every
  row that has no result yet, both failed rows and rows a stopped run never
  finished.

It uses only the Python standard library (Python 3.10 or newer), so there is
nothing to install.

## Quick start

```bash
export FIREWORKS_API_KEY=<your key>

# Serverless
python batch_to_realtime.py \
  --input batch_input.jsonl \
  --output results.jsonl \
  --model accounts/fireworks/models/gpt-oss-120b

# Your own deployment, for example a preemptible one
python batch_to_realtime.py \
  --input batch_input.jsonl \
  --output results.jsonl \
  --model "accounts/fireworks/models/gpt-oss-120b#accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID>" \
  --concurrency 64
```

Failed rows go to `results.jsonl.errors.jsonl` unless you pass `--errors`.

| Exit status | Meaning |
|---|---|
| 0 | Every row has a result. |
| 1 | Some rows still have no result: they failed in this run, or failed in an earlier run and were skipped with `--skip-errors`. Rerun the same command to try them again. |
| 2 | The run stopped early: `FIREWORKS_API_KEY` is missing, the API returned 401 (bad key), or it returned 404 because the model or deployment isn't available. Rows not yet sent are untouched, so rerun the same command once that's fixed. |
| 130 or 143 | You pressed Ctrl-C (130), or a job scheduler sent SIGTERM (143). The script stops at once without waiting for requests in flight; rerun the same command to continue. |

## Input and output formats

| Input row | Output row (success) | Output row (error) |
|---|---|---|
| OpenAI batch format: `{"custom_id": "...", "body": {...}}`, with an optional `"url"` (`/v1/chat/completions`, `/v1/completions` or `/v1/embeddings`) | `{"custom_id": "...", "response": <API response>}` | `{"custom_id": "...", "error": {"code": "...", "message": "..."}}` |

`--model` replaces the `model` in each row. `--max-tokens` and `--temperature`
only fill in rows that don't already set them, as Batch job settings do. Ids
are written back as they appear in the input, so a numeric `custom_id` stays a
number.

Rows are written as they finish, so the output order can differ from the
input order. Match rows on `custom_id`.

The `error.code` in a failed row is one of:

- an HTTP status such as `400` or `503`: the API's final answer, with its
  response body as the message;
- `connection_error`: the request timed out or the connection dropped on
  every attempt;
- `invalid_response`: the API answered with something that isn't JSON;
- `invalid_json`, `validation_error` or `duplicate_id`: the input line itself
  is malformed, in an unsupported format, or repeats an earlier id, so it was
  never sent;
- `client_error`: an unexpected error in the script while handling that row.

## Choosing settings

- **Progress:** every `--progress-every` rows (default 100) the script prints
  how many rows succeeded, failed and were retried. While retries happen, it
  also prints the latest reason at most every 30 seconds, for example a 503
  while a deployment scales up.
- **`--concurrency`** (default 16): the number of requests in flight at once.
  - On serverless, it is capped at 32 because serverless capacity is shared
    with other customers. Raise it until you see many 429 retries in the
    progress output, then back off.
  - On your own deployment, it is capped at 256. Start at about 8 to 16 per
    replica and raise it while latency stays acceptable.
  - When the API answers 429 (rate limited) or 503 (overloaded or scaling up),
    every worker pauses for the wait the server asks for, not only the row
    that got it, so the whole run backs off together.
- **Retries:** each row is retried until it succeeds or a limit runs out.
  - **What is retried:** 408, 409, 425, 429, every 5xx except 501 and 505,
    timeouts, and connections dropped mid-response. Other 4xx errors, such as
    a context that is too long or a 403 for an endpoint your plan doesn't
    include, go straight to the errors file.
  - **How long to wait:** the server's `Retry-After` when it sends one,
    otherwise exponential backoff from `--initial-backoff` (default 1s). Each
    wait is jittered and never longer than `--max-backoff` (default 60s).
  - **When to give up:** after `--max-retries` (default 30) or
    `--max-retry-time` (default 1800s, 30 minutes), whichever comes first. A
    retry's request timeout is cut to the time left, so a row never runs much
    past `--max-retry-time` after its first attempt.
  - **Preemption:** a preemptible deployment can lose its replica mid-run.
    Until the replica is rescheduled, requests fail with dropped connections or
    with 503 `DEPLOYMENT_SCALING_UP` and `Retry-After: 60`. The defaults ride
    out about 30 minutes of that. If rescheduling takes longer, raise
    `--max-retry-time`, or rerun the same command afterward.
  - **Bad key or model:** a 401 (bad key), or the gateway's 404 saying the
    model or deployment isn't available, stops the whole run with exit
    status 2 instead of failing every row. Rows not yet sent are left
    unrecorded, so rerunning the same command after fixing the key or model
    continues. A 404 that is specific to one row, such as a chat request to an
    embedding-only model, is recorded as that row's error instead.
  - **New deployments:** a deployment answers 404 until it is `READY`, so
    wait for `firectl deployment get <DEPLOYMENT_ID>` to show `State: READY`
    before you start.
- **`--timeout`** (default 600s): the per-request timeout. Raise it for very
  long generations.

## Resuming and retrying

This works like a Batch job continuation:

1. **Failed rows are collected** in the errors file, one row per failed input
   with its error, in the Batch error format.
2. **A stopped run leaves unfinished rows unrecorded.** Stop it with Ctrl-C,
   SIGTERM or anything else, and rows still in flight or queued are simply not
   written.
3. **A rerun of the same command sends every row without a result**: the
   failed rows and the unfinished ones. Rows already in the output file are
   skipped.

A failed row's old error row is removed only once that row has a new result or
a new error, so failures for rows that aren't in the input you pass stay
recorded and keep the exit status at 1. To rerun without resending rows that
already failed, for example ones that fail validation every time, add
`--skip-errors`.

Each row is written and flushed as soon as it finishes, so an interrupted run
loses at most the rows that were in flight. Those are sent again on the rerun.

If the process is killed in the middle of writing a row, the rerun first
repairs the file: it removes the incomplete last line (and sends that row
again), so both files stay valid JSONL for strict readers. Unreadable lines
anywhere else are reported as a warning, not removed.

## Tests

```bash
pip install pytest
python -m pytest test_batch_to_realtime.py
```

The tests run the script against a local fake server. They need no API key and
send no requests to Fireworks.
