# Tools

Standalone utility scripts that complement the main cookbook recipes.

| Script | Description |
|--------|-------------|
| `migrate_managed_job.py` | Rerun a managed SFT, DPO or ORPO job on the Training API from its job name and your local copy of the data. See [Migrating a managed fine-tuning job](#migrating-a-managed-fine-tuning-job). |
| `create_trainer_then_deployment.py` | Provision a trainer, block until it is ready, then create and wait for its linked hotload deployment. Supports opting the trainer out of reservations. |
| `promote_checkpoint.py` | Promote a sampler checkpoint from the control plane checkpoint list to a deployable Fireworks model. |
| `merge_lora_and_promote.py` | Merge a LoRA/PEFT adapter into its base (`checkpoint_type="merged_base"`) and promote the result as a full `HF_BASE_MODEL`. Provisions a short-lived LoRA trainer, loads `--adapter-model` (PEFT model resource) explicitly, saves the merged base, and promotes. |
| `list_checkpoints.py` | List checkpoint rows known to the control plane for a trainer job. |
| `reconnect_and_adjust_lr.py` | Reconnect to an existing trainer job and adjust the learning rate mid-run. |
| `verify_logprobs.py` | Compare inference-time and training-time logprobs for a checkpoint. |

## Migrating a managed fine-tuning job

Managed supervised fine-tuning (SFT), DPO and ORPO no longer accept new jobs;
see the [migration guide](https://docs.fireworks.ai/fine-tuning/training-api/migrate-from-managed).
`migrate_managed_job.py` reruns a managed job on the Training API from its job
name and your local copy of the data. It reads the job's settings, builds the
matching recipe config (`recipes/sft_loop.py`, `recipes/dpo_loop.py` or
`recipes/orpo_loop.py`), and runs it. The run promotes its final checkpoint to
a new model that you can deploy.

### Before you start

- **Your data.** Fireworks-hosted datasets can't be downloaded, so you need
  your own copy of the training JSONL: a local file, a directory of `.jsonl`
  shards, or a `gs://` path you can read. Use the same file you uploaded for
  the managed job. If the job had a separate evaluation dataset, bring that
  too.
- **An API key for one account.** Create the key in the account that owns the
  job. The Training API rejects keys that can reach more than one account.
- **The cookbook from GitHub.** The tool runs from a clone of the cookbook
  repository; the `fireworks-training-cookbook` package on PyPI doesn't
  include it.

  ```bash
  git clone https://github.com/fw-ai-external/cookbook.git && cd cookbook/training
  uv venv --python 3.12 && source .venv/bin/activate
  uv pip install -e .
  ```

### Run it

Preview first. `--dry-run` reads the job, prints the chosen mode, any warnings
and the full recipe config, then exits without creating anything:

```bash
export FIREWORKS_API_KEY=...
python -m examples.tools.migrate_managed_job \
    --job accounts/<account>/supervisedFineTuningJobs/<job-id> \
    --dataset-path ./train.jsonl \
    --output-model-id <new-model-id> \
    --mode auto --dry-run
```

For DPO and ORPO jobs, pass `--job accounts/<account>/dpoJobs/<job-id>`.
When the preview looks right, run the same command without `--dry-run`. The
training loop runs on your machine for the whole job, so keep the process
running until it finishes. The new model is
`accounts/<account>/models/<new-model-id>`; the managed job's output model
already exists, so the tool asks for a new ID.

| Flag | Meaning |
|---|---|
| `--job` | The managed job's resource name. |
| `--dataset-path` | Your copy of the training data. Dataset resource names (`accounts/.../datasets/...`) don't work. |
| `--output-model-id` | ID of the new model. Must differ from the old job's output model. |
| `--mode` | `auto` (default), `serverless` or `dedicated`. See below. |
| `--eval-dataset-path` | Your copy of the job's evaluation dataset, if it had one. |
| `--tokenizer-model` | Hugging Face tokenizer, such as `Qwen/Qwen3-8B`. Defaults to the base model's Hugging Face repo, then the job's renderer model. For a base model you uploaded, pass the Hugging Face repo it came from. |
| `--max-seq-len` | Defaults to the job's `max_context_length`, then the base model's training context length. |
| `--log-path` | Where checkpoints and logs go. Defaults to `./migrated_runs/<job-id>-<time>`. Reuse it to continue a run. |
| `--init-from-checkpoint` | Continue from a checkpoint that an earlier run of this command saved. See [Continue an interrupted run](#continue-an-interrupted-run). |
| `--trainer-job-id` | Reattach to the dedicated trainer an interrupted run kept, to resume it. Forces dedicated mode. |
| `--dry-run` | Print the plan and config, then exit. |

### Serverless or dedicated

[Serverless training](https://docs.fireworks.ai/fine-tuning/training-api/serverless)
runs LoRA jobs on a shared trainer pool for supported models and bills per
token. Dedicated training provisions a trainer for your run and supports any
trainable model, including full-parameter training.

`--mode auto` picks serverless when all of these hold, and dedicated otherwise:

- the job trained a LoRA adapter (`lora_rank > 0`);
- the base model supports serverless training;
- the job didn't warm start from an adapter, use a reservation, tune the vision
  encoder, or use ORPO.

If serverless setup fails because the model isn't available for serverless
training or no shared trainer is free, `auto` retries on a dedicated trainer.
Both errors happen before any training step. `--mode serverless` never falls
back, and fails up front if the job can't run there.

Dedicated training uses your account's
[GPU quota](https://docs.fireworks.ai/fine-tuning/finetuning-intro#before-you-start-account-tier-and-gpu-quota).
By default an account with a payment method on file gets 32 each of H200,
B200 and B300; yours can differ. If creating the trainer fails with HTTP 403 `tier_required`, add a
payment method or credits; with `quota_exceeded`, request more GPU quota. Full-parameter
DPO also starts a separate reference trainer, so it needs quota for two
trainers. Dedicated trainers stop after 60 idle minutes by default, and the
recipe deletes its trainer when the command exits.

### How settings map

Fields the managed job left unset keep the recipe default, which is what the
managed job used too.

| Managed job field | Recipe `Config` field | Notes |
|---|---|---|
| `base_model` | `base_model` | A warm-start job uses the adapter's base model. |
| `warm_start_from` | `warm_start_from_adapter` | Dedicated only. Uses the job's LoRA rank. |
| `dataset`, `evaluation_dataset` | `dataset`, `evaluation_dataset` | From `--dataset-path` and `--eval-dataset-path`. A `gs://` path streams (`dataset_loading_mode="stream"`, SFT and DPO). |
| `eval_auto_carveout` | `eval_auto_carveout` | Off when you pass `--eval-dataset-path`. |
| `epochs`, `learning_rate` | `epochs`, `learning_rate` | Recipe defaults differ by method: SFT trains 3 epochs, DPO and ORPO 1. |
| `lr_scheduler`, `learning_rate_warmup_steps` | `lr_scheduler` | `ConstantSchedule`, `LinearSchedule` or `CosineSchedule`, with `warmup_steps`, `min_lr_ratio` and `decay_ratio`. |
| `batch_size_samples` | `batch_size` | Rows per step for SFT, preference pairs per step for DPO and ORPO. |
| `optimizer_weight_decay` | `weight_decay` | 0 means the default, 0.01. |
| `lora_rank`, `lora_alpha` | `lora_rank`, `lora_alpha` | Rank 0 is full-parameter training. Alpha defaults to 32. |
| `max_context_length` | `max_seq_len` | Always set; serverless requires it. |
| `output_model` | `output_model_id` | A new ID from `--output-model-id`. |
| `thinking_trace_history_mode` | `thinking_trace_history_mode` | `interleaved` or `preserved`. |
| `renderer_hugging_face_repo_id` | `renderer_name` | Picks the chat renderer, as it did for the managed job. The tokenizer is the base model's Hugging Face repo; the renderer repo stands in only when the base model has none. |
| `sampler_checkpoint_save_interval_steps` | `sampler_save_interval` | |
| `generation_eval` | `generation_eval` | Serverless SFT only. |
| `wandb_config` | `wandb` | Set `WANDB_API_KEY` to sync; without it W&B logs offline. |
| `use_reservation`, `reservation_target` | `trainer.use_reservation`, `trainer.reservation_target` | Always copied, so a job that didn't use a reservation still doesn't. |
| DPO `beta`, `ref_cache_concurrency`, `ref_cache_batch_size` | Same names | |
| ORPO `orpo.lambda` | `orpo_lambda` | Dedicated only. |
| Region, accelerator, training shape | Not set | The platform picks the trainer shape and placement. |

Full-parameter SFT and DPO runs and ORPO runs save a resumable checkpoint
every 50 steps (`dcp_save_interval=50`); managed jobs saved one every 20 steps.
LoRA SFT and DPO runs save a resumable, promotable checkpoint every 30 minutes
and at each epoch end (`checkpoint_interval_minutes=30`), as managed jobs did.
Both are the recipe defaults.

### When the tool stops or warns

The tool stops before creating anything when:

- the job belongs to an account that uses customer-managed encryption keys
  (CMEK), which the Training API can't write yet;
- the API key can reach more than one account, or belongs to a different
  account than the job;
- `--dataset-path` is missing (the job's Fireworks-hosted dataset can't be
  read or downloaded, so you need your own copy), is a dataset resource name,
  or doesn't exist;
- `--dataset-path` and `--eval-dataset-path` mix a local path and a `gs://`
  path (the recipe reads both with one loading mode);
- `--output-model-id` is invalid or is the old job's output model;
- `--mode serverless` is set for a job that can't run on serverless training.

It warns and continues when the job used something the Training API can't
reproduce:

- a legacy token-based `batch_size` (the recipe batches by rows instead);
- `lora_target_modules`, `lora_dropout` or `tune_vision_encoder`;
- a warm-start adapter that trained modules the Training API doesn't adapt by
  default, such as an MoE router (loading it may fail);
- a dataset in S3 or Azure Blob Storage (copy it yourself);
- Weights & Biases logging without `WANDB_API_KEY` set;
- a LoRA rank above 64, which serverless LoRA hosting doesn't accept;
- an evaluation dataset you didn't pass with `--eval-dataset-path`;
- dedicated mode, which needs GPU quota (see above).

### Continue an interrupted run

If a run stops early, for example on a crash or Ctrl-C, the tool prints how to
continue from its newest saved checkpoint:

- **Serverless runs resume where they stopped.** Rerun the same command with
  the printed `--log-path` and `--init-from-checkpoint <account>/<run-id>/<checkpoint>`.
  The log path holds the dataset position and, for DPO, the frozen reference
  the resumed run must score against.
- **Dedicated runs whose trainer was kept resume on it.** Rerun with the
  printed `--log-path` and `--trainer-job-id`. SFT and DPO resume where they
  stopped; ORPO resumes its step but reads the dataset from its first row. A
  kept trainer bills until it stops after 60 idle minutes, so delete it
  (`firectl rlor-trainer-job delete <id>`) if you won't resume. If no
  checkpoint was saved, the tool says how to start over instead.
- **Dedicated runs whose trainer was deleted restart the dataset.** The printed
  `--init-from-checkpoint <trainer-job-id>:<checkpoint>` starts a new run from
  that checkpoint's weights and optimizer state, but training reads the
  dataset from its first row again.

### After training

The new model appears in your account like any other fine-tuned model, and
you deploy it the usual way. Serverless LoRA hosting accepts adapters up to
rank 64; deploy a higher-rank adapter or a full-parameter model on a dedicated
deployment.
