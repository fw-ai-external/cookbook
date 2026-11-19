# Run Batch input files on serverless or a deployment

The Fireworks Batch API is temporarily disabled while it is rebuilt. Until it is
back, [`batch_to_realtime.py`](batch_to_realtime.py) runs your existing Batch
input files against the regular inference API. You can point it at a
serverless model or at your own deployment, such as a
[preemptible deployment](https://docs.fireworks.ai/guides/ondemand-deployments#preemptible-deployments-eval-/-batch).

The script:

- reads a Batch input JSONL file in the OpenAI batch format;
- sends rows in parallel, retrying rate limits, server errors, timeouts and
  dropped connections for up to a day per row, so it rides out a
  preemptible deployment losing its replica;
- writes successful rows to one JSONL file and failed rows to another, in the
  matching Batch output format;
- reruns like a Batch job continuation: rerunning the same command sends every
  row that has no result yet, both failed rows and rows a stopped run never
  finished;
- with `--delete-deployment-on-finish`, deletes your deployment once every row
  has a result, the way a Batch job cleaned up after itself.

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

# Your own deployment, for example a preemptible one, deleted when every row
# has a result
python batch_to_realtime.py \
  --input batch_input.jsonl \
  --output results.jsonl \
  --model "accounts/fireworks/models/gpt-oss-120b#accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID>" \
  --concurrency 64 \
  --delete-deployment-on-finish
```

Failed rows go to `results.jsonl.errors.jsonl` unless you pass `--errors`.

| Exit status | Meaning |
|---|---|
| 0 | Every row has a result. |
| 1 | Some rows still have no result: they failed in this run, or failed in an earlier run and were skipped with `--skip-errors`. Rerun the same command to try them again. |
| 2 | The run stopped early: `FIREWORKS_API_KEY` is missing, the API returned 401 (bad key), or it returned 404 because the model or deployment isn't available. Rows not yet sent are untouched, so rerun the same command once that's fixed. |
| 3 | Every row has a result, but `--delete-deployment-on-finish` couldn't delete the deployment. The message says why; delete it with `firectl deployment delete accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID> --ignore-checks`. |
| 130 or 143 | You pressed Ctrl-C (130), or a job scheduler sent SIGTERM (143). The script stops at once without waiting for requests in flight; rerun the same command to continue. |

## Running on a preemptible deployment

A preemptible deployment behaves differently from a Batch job in a few ways:

- **Leave the region unset.** Leave out `--region` unless you need a specific
  one for data residency. Then Fireworks can place the deployment, and move a
  waiting replica, wherever capacity is free. A deployment pinned to one region
  waits there until that region has room.
- **Expect waits at busy times.** A preemptible replica can wait a long time
  for capacity, both at create and after a preemption. By default each row
  keeps retrying for up to a day. If that's still not enough, raise
  `--max-retry-time`, or rerun the same command later; finished rows are
  skipped.
- **Delete the deployment when you're done.** Unlike a Batch job, a deployment
  keeps running, and billing, after your run finishes. Pass
  `--delete-deployment-on-finish` and the script deletes it once every row has
  a result. If any row failed or the run stopped, it keeps the deployment so a
  rerun can finish, and says so. It's a soft delete, so
  `firectl deployment undelete accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID>`
  brings it back for a while afterward.

  Use this option only with a deployment created for this run. To delete it
  right after the run, the script skips the API's safety checks: the one that
  refuses to delete a deployment that served requests in the last hour, and
  the one that protects a default deployment. It deletes the deployment even
  if other traffic or other models use it.

  To delete it yourself instead:

  ```bash
  firectl deployment delete accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID> --ignore-checks
  ```

  If you run files regularly, you can instead create it with
  `--min-replica-count 0 --scale-to-zero-window 30m`, so it scales to zero
  when idle. The first request after that waits for a replica to start, which
  the script retries through.

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
  - **How long to wait:** exponential backoff from `--initial-backoff`
    (default 1s), or the server's `Retry-After` when it asks for longer. Each
    wait is jittered and never longer than `--max-backoff` (default 60s).
  - **When to give up:** after `--max-retry-time` (default 86400s, a day).
    `--max-retries` also caps the number of retries; by default it has no
    limit. On serverless, lower `--max-retry-time` if you'd rather see a
    failing row in the errors file sooner. A retry's request timeout is cut to
    the time left, so a row never runs much past `--max-retry-time` after its
    first attempt.
  - **Preemption:** a preemptible deployment can lose its replica mid-run.
    Until the replica is rescheduled, requests fail with a 503 from the
    deployment's load balancer (`no healthy upstream`), a 502 or 504, or a
    dropped connection; none of them says the deployment was preempted. So when
    `--model` names your deployment, a row's next retry after one of these
    reads the deployment's state instead of resending, and keeps reading it
    while the state is `PREEMPTED` (on accounts that have that state). The
    row resends once the deployment leaves `PREEMPTED`. These retries use the
    same backoff and limits as any other, and a row that runs out records
    `deployment_preempted`. A deleted or failed deployment stops the run with
    exit status 2. The defaults ride out up to a day; if rescheduling takes
    longer, raise `--max-retry-time`, or rerun the same command afterward.
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

## Job log

Pass `--job-log job.log.jsonl` and the script appends the run's events to
that file, one JSON object per line. The last line says where the job stands;
the lines before it say how it got there.

```jsonl
{"time": "2026-11-15T09:10:00Z", "run": 2, "event": "start", "state": "JOB_STATE_RUNNING", "model": "accounts/<ACCOUNT_ID>/models/<MODEL>#accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID>", "input": "batch_input.jsonl", "output": "results.jsonl", "errors": "results.jsonl.errors.jsonl"}
{"time": "2026-11-15T09:11:00Z", "run": 2, "event": "progress", "state": "JOB_STATE_RUNNING", "rows": {"succeeded": 412, "failed": 3, "skipped": 1200, "outstanding": 0, "retries": 57}}
{"time": "2026-11-15T09:11:20Z", "run": 2, "event": "retry", "state": "JOB_STATE_RUNNING", "message": "Retrying after 429: {\"error\": ... rate limited ...}"}
{"time": "2026-11-15T13:40:05Z", "run": 2, "event": "end", "state": "JOB_STATE_COMPLETED", "exit_code": 0, "error": null, "rows": {"succeeded": 8800, "failed": 0, "skipped": 1200, "outstanding": 0, "retries": 230}}
```

| `event` | When | What it adds |
|---|---|---|
| `start` | The run starts | The model and the input, output and errors files |
| `progress` | Every 60 seconds | `rows`: this run's counts so far |
| `retry` | A row is retried, at most every 30 seconds | `message`: the latest reason, such as a 429 while rate limited or `deployment_preempted` while the deployment is `PREEMPTED` |
| `end` | The run ends | `exit_code`, `error` (why it failed or stopped, or `null`) and the final `rows` |

- **`state`** is `JOB_STATE_RUNNING` until the `end` line, which sets it from
  the exit status:

  | Exit status | `state` |
  |---|---|
  | 0 | `JOB_STATE_COMPLETED` |
  | 3 | `JOB_STATE_COMPLETED`; `error` says the deployment delete failed |
  | 1 | `JOB_STATE_FAILED`: some rows have no result, and a rerun retries them |
  | 2 | `JOB_STATE_FAILED`; `error` says why the run stopped early |
  | 130 or 143 | `JOB_STATE_CANCELLED` |

- **`rows`** counts this run. `skipped` is the rows that already had a result
  from an earlier run, and `outstanding` is the earlier failures that this run
  didn't resend.
- **`run`** counts how many times the job has been started. A rerun of the
  same command appends to the same log as the next run. Use one log per job.
- **A last line that isn't `end` and is more than a minute old** means the
  process died without a chance to log its end, for example when its machine
  was lost. Rerun the same command to continue.

To see where a job stands, read the last line:

```bash
tail -n 1 job.log.jsonl
```

A line cut off by a killed run is left as it is, and the next run starts on a
new line, so skip lines that aren't valid JSON when you read the log. The
script reports a log it can't write once, on stderr, and keeps running.

## Running it as a job

The script is one job run: start it, stop it, rerun it to continue. A job
platform such as a Cloud Run job, a Kubernetes Job, or a systemd service can
manage it like a Batch job:

- **Cancel:** stop the run. The platform sends SIGTERM, and the script exits
  at once with 143, leaving its unfinished rows unrecorded. For a Cloud Run
  job, that's `gcloud run jobs executions cancel <EXECUTION>`.
- **Retry a lost run:** let the platform start the same command again when a
  run exits with a status other than 0, for example a Cloud Run job's
  `--max-retries`. The rerun sends only the rows without a result. Exit 2
  repeats until you fix the key or the model, so watch for it rather than
  retrying it forever.
- **Deadline:** set the platform's timeout, for example a Cloud Run job's
  `--task-timeout`. Check the platform's documentation for its maximum. A run
  that hits it gets SIGTERM, like a cancel, and is recorded as
  `JOB_STATE_CANCELLED`.
- **How many run at once:** that's the platform's to limit. The script's
  `--concurrency` only limits requests within one run.
- **Keep the files across runs:** a retry continues only if it sees the
  earlier run's `--output`, errors file and `--job-log`. A container's own
  disk goes away with it, so keep them on storage that outlives the run. On
  Cloud Run, either mount a volume, or copy the files down from Cloud Storage
  before the run and back up after it. If you copy them, also copy them up
  every few minutes while the run goes on, since a run that's killed never
  reaches the copy at the end. The script appends to `--output` and
  `--job-log`, and rewrites the errors file in place on a rerun, so check how
  your volume handles appends and file replacement before you rely on a mount.
- **See every job:** keep each job's log under one prefix, and read the last
  line of each.

## Tests

```bash
pip install pytest
python -m pytest test_batch_to_realtime.py
```

The tests run the script against a local fake server. They need no API key and
send no requests to Fireworks.
