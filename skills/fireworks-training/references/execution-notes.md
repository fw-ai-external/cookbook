# Execution notes

Practical details that make a real run go smoothly. Each came from a live run.
Apply them while working; in replies, report what you checked and found
rather than restating these notes.

Environment and credential setup is in `sdk-setup.md` and `getting-started.md`.

## Model and data checks

- Confirm that the model supports your training method in the
  [fine-tuning model guide](https://docs.fireworks.ai/fine-tuning/finetuning-intro.md).
- Training API recipes read your input files directly. For a Managed migration,
  use the original training and evaluation files. Inspect one rendered example
  before launch.

## Resource IDs

Create calls can hang. Give the client a timeout
(`Fireworks(account_id=..., timeout=120)`); if a create times out, `get` the
planned ID before retrying, and retry with the same ID only if it does not
exist.

For a new run, choose a fresh log directory and output model ID. When
reporting losses, verify the log belongs to the intended recipe run.

On busy shared accounts, list with a `filter` (for example
`base_model="accounts/fireworks/models/muse-glimmer-30b"`) and a small
`page_size`; unfiltered lists can fail with a message-too-large error.

## Resolved configuration

The plan gives concrete values. Always set the LoRA rank explicitly (8 or 16
for a first run). When a value is resolved at run time, write "resolved at
run" rather than guessing, then read it back from the run's logged metrics and
report it.

## Monitoring a serverless run

- Serverless Training API sessions draw from a per-account, per-base-model
  concurrent-run quota. A create rejected with 429 "maximum concurrent
  serverless training runs" is waiting on that quota, not broken. Sessions do
  not appear in job lists and release their slot only when the run becomes
  terminal after the session expires (see the docs' serverless capacity
  section), so an open or recently closed session of yours can hold it. Quote
  the error, explain the wait, and do not guess whose run holds the slot.
  Never cancel another run to free the slot without approval.
- To follow a background run, write its output straight to a log file and read
  the file. Piping through `grep` or `tee` buffers output, so a healthy run can
  look silent; use `python -u` and `grep --line-buffered` if you must filter.

## Reporting what still costs money

Name only documented facts. The serverless docs say checkpoint storage is
currently included at no charge. No public page states a storage price for
promoted LoRA models, so say the model stays on the account and that no
storage charge is documented, rather than calling it free. The promotion
covers training, not sampling: report in-session evaluation sampling as its own
cost line (estimated or unverified), never inside a `$0` total. On-demand
deployments bill per GPU-hour while replicas are running.

## Evaluation

Read training and evaluation metrics from the recipe's run log or configured
Weights & Biases run before reporting. If eval loss
rose, say so plainly and treat it as a warning, then look at learning rate,
steps, data size, duplicates, and masking. Judge success against the approved
held-out task metric and baseline, since loss and accuracy can move apart.
Use the same decoding settings (`max_tokens`, temperature, stop sequences) for
base and tuned. Before reporting a surprising number (for example a 0% base),
read a few raw outputs and count unterminated samples (`termination.is_clean`
false): those ran out of `max_tokens` and measure the budget, not the model. Keep per-example raw outputs and predictions for base and tuned on the same holdout so you can
report a paired comparison (for example an exact McNemar test) and intervals,
not only two means. The starter's serverless path writes
`predictions_base.jsonl`, `predictions_tuned.jsonl`, and a `paired` block in
`summary.json`.
