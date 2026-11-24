# Training API (custom training loops)

*Source of truth: [Training API intro](https://docs.fireworks.ai/fine-tuning/training-api/introduction.md) · [current serverless comparison](https://docs.fireworks.ai/fine-tuning/training-api/serverless.md#serverless-vs-dedicated) · [public cookbook](https://github.com/fw-ai-external/cookbook/tree/main) — defer to the live docs/repo for current SDK + recipes, then record the checkout commit used by the run.*

Write or fork a Python training loop; Fireworks runs forward/backward on distributed GPUs. The SDK is **Tinker-compatible**, so Tinker code ports over with minimal changes. For standard supported jobs, prefer managed training.

Docs: https://docs.fireworks.ai/fine-tuning/training-api/introduction · Cookbook (start here): https://github.com/fw-ai-external/cookbook

> Status: Serverless Training is generally available. Dedicated Training needs access; request it at https://fireworks.ai/contact-training

## Access preflight

Before selecting a Training API path, confirm that the target account was enabled through onboarding or account settings. For dedicated work, read the account and shared shape inventories with `firectl training-shape list`; this proves shape visibility but does not by itself prove serverless entitlement. If no authoritative read-only entitlement signal is available, mark access unverified and ask the user to confirm enablement. Do not discover access by creating a trainer or serverless session.

## Managed training is retired

Managed SFT, DPO, and ORPO no longer accept new jobs; run all of them on the
**Training API** (serverless or dedicated). To rerun an old managed job, see
`training/examples/tools/README.md` and the
[migration guide](https://docs.fireworks.ai/fine-tuning/training-api/migrate-from-managed).

## Training API infrastructure

| Path | Use when | Cookbook |
|---|---|---|
| **Serverless** | LoRA SFT, DPO, ORPO, RL, distillation, and custom loops on models in the shared pool; per-token billing; no trainer or deployment to provision; session-based training and in-session sampling | `sft_loop` / `dpo_loop` with `serverless=True` (`training/examples/sft/train_sft_serverless.py`, `training/examples/dpo/train_dpo_serverless.py`), the [external SDFT cookbook](https://github.com/fw-ai-external/tinker-cookbook) for SDFT, RL via `training/examples/rl/*/train_*_serverless.py` on `training/recipes/experiment/async_rl_loop_serverless.py` |
| **Dedicated** | Full-parameter training, models not in the serverless pool, sustained workloads, explicit trainer, deployment, checkpoint, and teardown control | The matching `train_*.py` examples and `training/recipes/` with `TrainerConfig` / `DeployConfig` |

Default to Serverless when the model is in the pool and LoRA is enough; that
includes DPO, ORPO, distillation, and custom losses. Read the live
[serverless](https://docs.fireworks.ai/fine-tuning/training-api/serverless.md) and
[dedicated](https://docs.fireworks.ai/fine-tuning/training-api/dedicated.md) pages before choosing.

## How to run RL

RL runs on the Training API. Supply an inline `reward_fn(completion, row) -> float`. It may
read `ground_truth`, another declared reference field, tool outcomes, environment state, or a
judge result. There is no evaluator resource — same shape as Tinker's reward-in-the-loop. This
is fully agent-drivable.

- **Serverless:** start from `training/examples/rl/*/train_*_serverless.py` (on
  `recipes/experiment/async_rl_loop_serverless.py`). Rollouts sample from the session's
  sampler snapshots; nothing is provisioned.
- **Dedicated:** fork `training.recipes.rl_loop` / `async_rl_loop`; the recipe provisions a
  trainer plus a rollout deployment from `TrainerConfig` / `DeployConfig`.

Managed RFT (`firectl reinforcement-fine-tuning-job create`) has been removed. For questions
about jobs that already exist, see `managed-rft-operations.md`.

## Ways to supply the reward

The reward can come from two places:

1. **An inline `reward_fn`** — write the reward as Python in the forked recipe. The agent-drivable default; uses no evaluator resource.
2. **An LLM-judge rubric** — when correctness isn't a clean 0/1 (tone, quality, open-ended), score completions with a judge model against a rubric. Implement it inside `reward_fn` (call a judge model, map its verdict to a float). If the user gives no evaluator and no rubric, infer a data-grounded rubric from the training samples and confirm it with them before spending.

## Default workflow: fork a cookbook recipe

> **Cookbook implementation:** use the exact recipe and SDK references routed
> from `../../configure/SKILL.md`. This file explains serverless versus
> dedicated Training API choices; this reference directory and `training/` contain implementation
> depth.

**Approval gate still applies.** Training API cookbook recipes can create trainers, rollout deployments, checkpoints, promoted models, and paid inference. Before running a recipe or any create/promote/deploy action, show the account, recipe, dataset, full resolved trainer and deployment config, cost drivers, evaluation plan, and teardown plan; get explicit approval for this run. A deployment or materially expanded sweep gets its own approval.

Don't write a loop from scratch — fork a recipe in the `training/` tree of `fw-ai-external/cookbook`:
- `training/recipes/` — loop scripts (e.g. `async_rl_loop`)
- `training/recipes/experiment/` — experimental infrastructure variants, including async serverless RL and SAO/PPO
- `training/examples/` — worked RL / SFT / DPO / ORPO
- `training/utils/` — config, data loading, losses, metrics

Recipes cover SFT, DPO/ORPO, and RL. The RL loops train GRPO; DAPO, GSPO, CISPO, DRO, DPPO and others are objective modules in `training/utils/rl/algorithm/` that a forked loop calls (see `rl-custom-loss.md`). SAO/PPO and IGPO have their own recipes.

## Quick reference: Serverless SFT

This block ran live on Muse Glimmer 30B. When the cookbook checkout has the
Muse starter or `sft_loop`, run those and fork them for custom loops; use this
block when they are missing.

Use a configured Serverless `FiretitanServiceClient` named `service`. This
loop saves base and tuned snapshots (run from
`$COOKBOOK_DIR/training` so the `training.*` helpers import):

```python
from fireworks.training import tinker
from training.utils.supervised import build_renderer, render_messages_to_datum
from training.utils.tokenizers import load_tokenizer
from training.renderer import get_text_content

tok = load_tokenizer("meta-models/Muse-Glimmer-30B")
renderer = build_renderer(tok, "meta-models/Muse-Glimmer-30B", "muse_glimmer")
datums = [render_messages_to_datum(r["messages"], renderer=renderer, max_seq_len=2048).datum for r in rows]

try:
    tc = service.create_lora_training_client(base_model="accounts/fireworks/models/muse-glimmer-30b", rank=8)
    for batch in batches(datums):
        fb = tc.forward_backward(batch, "cross_entropy").result()          # fb.metrics["loss:sum"]
        opt = tc.optim_step(tinker.AdamParams(learning_rate=1e-4),
                            emit_grad_norm_metrics=True).result()          # opt.metrics grad norms
    path = tc.save_weights_for_sampler("tuned").result().path
    sampler = service.create_sampling_client(model_path=path, tokenizer=tok)
    out = sampler.sample(prompt=renderer.build_generation_prompt(msgs[:-1]), num_samples=1,
                         sampling_params=tinker.SamplingParams(max_tokens=256, temperature=0.0,
                                                               stop=renderer.get_stop_sequences())).result()
    msg, termination = renderer.parse_response(out.sequences[0].tokens)
    answer = get_text_content(msg)   # drops reasoning (thinking) parts
    sampler.close()
finally:
    service.close()
```

`sample` returns a future: for an eval set, submit every request first, then
collect results, instead of waiting on each one in turn.

Snapshot the base adapter the same way before the first step to compare base
and tuned on one holdout, and keep each example's raw output, parsed answer,
and `termination` for both. Muse Glimmer reasons before answering: a base
sample that ends `MALFORMED` usually ran out of `max_tokens` before the answer,
which is a budget problem, not a capability score. Raise the budget or report
those rows separately. Use the same `max_tokens`, temperature, and stop
sequences for base and tuned, and read a few raw outputs before reporting a
surprising number.

To keep the tuned adapter after the session, promote its sampler checkpoint
before closing; session checkpoints are unreachable once the session ends:

```python
# fw is a configured FireworksClient.
rows = [r for r in fw.list_training_session_checkpoints(service.training_session_name)
        if r.get("promotable") and "tuned" in r["checkpointName"]]   # ids are prefixed/suffixed; never match by equality
fw.promote_session_checkpoint(name=max(rows, key=lambda r: r["createTime"])["name"],
                              output_model_id="my-tuned-lora", base_model="accounts/fireworks/models/muse-glimmer-30b")
```

`service.close()` ends your use of the session, but the run keeps its slot in
the account's concurrent-run quota until the session expires. Do not tell the
user nothing is in use until then; a new serverless run on the
same model may queue or be rejected. More in
[`execution-notes.md`](execution-notes.md).

To release the slot immediately, call `service.delete_training_session()`
before `service.close()`, after promoting anything you want to keep. The
`sft_loop` / `dpo_loop` `close_session_on_exit=True` option does this at
teardown, including when the run fails. Either way the session's checkpoints
can no longer be resumed or promoted, so leave it off for runs that a later
process may resume.

## Core SDK primitives

- `forward_backward` — built-in losses by id (e.g. `"cross_entropy"`), no extra forward pass.
- `forward_backward_custom(datums, loss_fn)` — your Python loss; returns per-token logprobs with gradients. **Loss runs locally; forward/backward run on remote GPUs.**
- `forward` — forward-only (e.g. reference-model logprobs); optional `loss_fn_config={"top_k": K}` returns per-token top-K logprobs/indices — see below.
- `forward_projection(datums)` — forward-only raw projection-head outputs (`loss_fn_outputs[i]["projection"]`, shape `[tokens, projection_head_dim]`) for a trainer created with a positive `projection_head_dim`. Any compatible validated training shape can be used; the trainer setting configures the head dimension. Accumulates no gradients; train the head with `forward_backward_custom(..., output="projection")`. See `recipes/experiment/ppo_value_head_loop.py`.
- `optim_step(...)` — optimizer update after gradient accumulation.
- `save_weights_for_sampler()` and `create_sampling_client()` export a checkpoint and open a sampler for evaluation or rollouts. A sampling client can use an existing sampler checkpoint or a base model. To share its configuration across processes, use `get_sampler_handle()` and `FiretitanSamplingClient.from_sampler_handle(...)`.

Loss and datum routing: `training-api-losses.md`.

```python
def loss_fn(data, logprobs_list):   # logprobs_list: per-token, requires_grad
    # return (scalar differentiable loss, {metrics for logging})
    ...
```

### Gradient norm metrics (SFT and RL)

Telemetry defaults to off. Enable it on each optimizer update:

```python
result = training_client.optim_step(
    params, emit_grad_norm_metrics=True,
).result()
print(result.metrics)
```

Log `result.metrics` with the optimizer step. `True` emits global L2
`grad_norm` and `grad_norm_rms`; `"detailed"` adds parameter-bucket norms.
`grad_norm` is after accumulation normalization and before clipping; preserve
the run's normalization setting. Keys may have a `:last` suffix.
The RL loops always request `True`; a fork that needs bucket norms passes
`"detailed"`. `sft_loop` exposes `grad_norm_metrics` (`"off"`, `"basic"`, or
`"detailed"`) and defaults it to `"basic"`. Verify the first step's logs; missing metrics do not mean zero
gradients.

### Top-K logprobs from `forward` (entropy / distributional KL)

`forward` (forward-only, no gradient) accepts an optional `loss_fn_config={"top_k": K}`. When set, the trainer returns per-token top-K logprobs and their vocab ids alongside the requested-token `logprobs`, in the same pass — no separate full-vocab forward needed:

```python
result = training_client.forward(
    data,
    "cross_entropy",                        # loss_fn id; ignored in forward-only mode
    loss_fn_config={"top_k": 20},
).result()

for item in result.loss_fn_outputs:
    logprobs = item["logprobs"].data             # requested-token logprobs, shape [seq_len]
    top_k_logprobs = item["top_k_logprobs"].data  # shape [seq_len, 20], float32, sorted descending
    top_k_indices = item["top_k_indices"].data    # shape [seq_len, 20], int64 vocab ids, same order
```

Use it for per-token distribution statistics a single requested-id logprob can't give you — entropy (`-Σ p·log p`), collision (`Σ p²`), or an approximate KL against a reference model's own top-K. Fields are populated only when `top_k > 0` (`top_k` must be `>= 0`, else `ValueError`). They're for analysis — no built-in loss reads them.

## RL: async loop + rollouts

RL recipe: https://docs.fireworks.ai/fine-tuning/training-api/cookbook/rl

`training.recipes.rl_loop` (sync) and `async_rl_loop` (rollout/train overlap; superset, preferred for new work) are the fork-and-customise recipes. The one thing you customise is the **inline reward**, exactly like Tinker:
```python
def reward_fn(completion: str, row: dict) -> float:   # fork this
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    return 1.0 if predicted == truth else 0.0
```
The dataset is JSONL of prompts (`messages`) plus exactly the fields the selected reward reads. `ground_truth` is common for exact-match rewards, but it is not universal. Declare and validate the reward's required fields before launch. The recipe samples rollouts, scores each with `reward_fn`, and trains with the trainer's built-in GRPO loss, clipped against the rollout logprobs, with no reference KL. Key knobs: `completions_per_prompt`, `prompt_groups_per_step`, `eps_clip`/`eps_clip_high`, and `max_head_offpolicy_versions` (async; 0 = strict on-policy). To change the objective, fork the loop's train step (`rl-custom-loss.md`). On Dedicated, the recipe provisions the trainer + rollout deployment and tears them down on close; on Serverless, the serverless examples sample from session snapshots instead.

## Numerics alignment & MoE Router Replay (R3)

Trainer↔inference divergence silently wrecks RL. Required reading: https://docs.fireworks.ai/fine-tuning/rl-rollout-integration#numerics-alignment
- Match **precision/quantization** between trainer checkpoints and the deployment shape.
- Measure **logprob divergence** between trainer forward and rollout inference on the same tokens.
- **MoE → Router Replay (R3):** align the top-K experts the router picks. The RL loops request completion-only R3 automatically for MoE base models and attach the rollout's routing references to trainer datums.

### Migrate inline R3 to shared storage

**Inline R3 can be much slower**, especially for long rollouts or large
training batches. It puts a base64 routing matrix for every captured token
in the inference response and sends those matrices back in the training
request. That increases payload size and adds parsing and decoding on the
CPU. Shared-storage R3 writes the matrices to NFS as Parquet. The rollout
returns compact file and span references that the trainer reads directly.

1. Use a training shape and rollout deployment that support shared-storage
   R3. For dedicated runs, create the trainer first and attach the rollout to
   it. Confirm `training_client.r3_store_id` is set and matches the store
   reported by `sampler.r3_store_id`. See [trainer-first setup](rl-hotload.md#debug-a-trainer-first-deployment).
2. For a custom SDK loop, bind the deployment sampler to the policy's
   training client. The SDK selects the shared-storage transport.

   ```python
   sampler = service.create_deployment_sampler(training_client=training_client)
   try:
       (completion,) = await sampler.sample_with_prompt_tokens(
           prompt_token_ids,
           n=1,
           max_tokens=128,
           logprobs=True,
           include_routing_matrix=True,
           echo=False,
       )
   finally:
       sampler.close()
   ```

   For direct HTTP calls, send `routing_matrix_format: "parquet_v1"` and
   `logprobs: true` in the body and set `x-fireworks-r3-store-id` to the
   trainer's store ID. A request with only `include_routing_matrix: true`
   selects inline R3.
3. Check one rollout before starting the full run. The SDK's
   `completion.routing_matrices` should contain a `RoutingReferences` object
   with files in the trainer's store. A finished HTTP choice should contain
   `routing_references` and `routing_matrix_format: "parquet_v1"`. Preserve
   reference spans through rollout assembly and verify the trainer consumes
   them in a small training step. Recapture stored inline trajectories and
   expired references. If the rollout rejects Parquet or returns inline data,
   check that the trainer and rollout versions both support the shared store.

## Critical rules

- **Start from a cookbook recipe**, not a blank loop.
- **Align numerics** before trusting any RL signal; keep **R3** on for MoE.
- **Dedicated RL only:** use a separate rollout deployment (sized for sampling), not your prod endpoint. Serverless RL needs no deployment.
- **Dedicated RL only: set an explicit deployment shape for the rollout.** RFT provisions a rollout/sampling deployment, and the accelerator is owned by the *deployment shape*. With only a base model and `training_shape=auto`, creation fails: `ValueError: Cannot create a managed deployment without a deployment shape`. Pass a `deployment_shape` (or a `training_shape_id` whose shape references one) — e.g. via the recipe's `--training-shape`. The RFT preflight should check this before launch. (Observed live on qwen3-4b, 2026-07-15.)
- Use `forward_backward_custom` only for a genuinely custom objective; otherwise the built-in path is cheaper.
- Loss must be differentiable w.r.t. `logprobs_list`; the metrics dict is logging-only.
