# Muse Glimmer mini cookbook

Fine-tune Muse Glimmer 30B on a public Hugging Face example or your own data.
Muse Glimmer 30B LoRA SFT is free on Serverless Training until November 30,
2026 ([source](https://docs.fireworks.ai/fine-tuning/models)). The promotion
covers jobs started before `2026-12-01T00:00:00-08:00`; full-parameter SFT, RL,
deployments, inference, DPO, and Dedicated trainers are not covered. The starter
refuses to launch after the deadline unless you pass `--allow-paid`.

| | |
|---|---|
| Base model | `accounts/fireworks/models/muse-glimmer-30b` |
| Tokenizer | `meta-models/Muse-Glimmer-30B` |
| Renderer | `muse_glimmer` |
| Public examples | [Prompt routing](https://huggingface.co/datasets/SupraLabs/Prompt-Routing-Dataset) and [receipt extraction](https://huggingface.co/datasets/naver-clova-ix/cord-v2) |

## Setup

```bash
git clone --depth 1 https://github.com/fw-ai-external/cookbook && cd cookbook/training
python3 -m venv .venv && source .venv/bin/activate && pip install -e .  # Python 3.11+
read -s FIREWORKS_API_KEY && export FIREWORKS_API_KEY  # never paste it into chat
```

Create the key at <https://app.fireworks.ai/settings/users/api-keys>, or put
`FIREWORKS_API_KEY=...` in a gitignored `training/.env`.

## Download a public example at runtime

The helper downloads a public Hugging Face dataset, converts it to chat JSONL,
and caches it locally. It does not read or create a Fireworks account dataset.
Choose `prompt-router` for classification or `cord-receipts` for vision:

```bash
python -m examples.sft.example_datasets --example prompt-router
python -m examples.sft.example_datasets --example cord-receipts
```

The 600-row prompt-routing subset, the 300-row receipt subset, and the conversion
logic match the existing prompt-router and CORD-v2 case-study notebooks. Receipt
conversion resizes and embeds images as JPEG data URIs. An existing output file
is reused; replace it explicitly to fetch a fresh copy.

## SFT with a base-versus-tuned evaluation

The starter runs on Serverless Training and compares base and tuned adapters on
held-out rows. `plan` is read-only. `prepare` is local and free: it validates
the input without rewriting it, holds out 40 rows, writes the train and holdout
splits, and renders one example for inspection.

```bash
RUN=tickets-$(date +%Y%m%d-%H%M%S)  # use a fresh ID for every run
python -m examples.sft.example_datasets --example prompt-router
python -m examples.sft.muse_glimmer_starter prepare --file prompt-router.jsonl --run-id $RUN
python -m examples.sft.muse_glimmer_starter plan --surface serverless --file prompt-router.jsonl
python -m examples.sft.muse_glimmer_starter launch --surface serverless --file prompt-router.jsonl --run-id $RUN
```

Inspect the rendered example before launching. The script rejects rows longer
than `--max-seq-len` (2,048) and runs one epoch of cross-entropy training. It
writes metrics, base and tuned predictions, and a comparison summary to
`muse_starter_logs/$RUN/`. Per-key JSON accuracy and exact match are included.

In-session sampling does not require a deployment but is billed per token
separately from free training. The run reports its sampled token counts. The
tuned checkpoint is promoted to
`accounts/<account>/models/muse-starter-sl-$RUN`; pass `--no-promote` to skip.

## Your chat data

Supply a local JSONL file in this format instead of the public example:

```json
{"messages": [{"role": "system", "content": "Classify this request."}, {"role": "user", "content": "My invoice is wrong."}, {"role": "assistant", "content": "{\"category\":\"billing\"}"}]}
```

Do not silently drop or rewrite malformed rows. If corrections are needed, make
a separate copy with the data owner's approval. A few hundred clean rows are
enough for a first signal.

## RL on DeepMath (paid)

RL is not covered by the promotion. Price a run before launching it, and keep
its first run small:

```bash
python -m examples.rl.deepmath.train_deepmath_serverless \
  --base-model accounts/fireworks/models/muse-glimmer-30b \
  --tokenizer-model meta-models/Muse-Glimmer-30B --renderer-name muse_glimmer \
  --max-rows 32 --prompt-groups-per-step 8 --completions-per-prompt 4 \
  --max-completion-tokens 4096 --max-seq-len 8192 --lora-rank 8
```

Watch verifier reward, evaluation score, and gradient norms. RL mutations need
the same plan review and approval as any other paid training run.

## Cleanup

```bash
python -m examples.sft.muse_glimmer_starter cleanup --run-id $RUN
```

This deletes the saved serverless model if it exists. Serverless Training
sessions close when the script exits. Deleting a model or deploying a tuned
model is a separate action; deployments are billed per GPU-hour.

## With a coding agent

Install the cookbook skills ([Getting started](../../../skills/GETTING-STARTED.md))
and ask for the prompt-router or receipt-extraction example. The agent downloads
the chosen data locally with the helper above, shows the training plan and cost,
waits for your approval, then runs and reports the job.
