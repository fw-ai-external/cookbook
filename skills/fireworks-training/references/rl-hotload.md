# RL: weight sync (hotload) during training

> Dedicated Training only: hotload publishes weights to a provisioned rollout
> deployment. Serverless RL samples from session sampler snapshots instead.

RL is the main consumer of hotload: the recipe publishes updated weights to the serving deployment so new rollouts come from the updated policy. Publication uses negotiated RDMA or a sampler checkpoint followed by FILE hotload. SFT / DPO / ORPO don't typically hotload — they save once at the end and call it a day.

`training/recipes/rl_loop.py` is always strict on-policy: it performs an
initial hotload and another hotload after every optimizer step. It intentionally
has no sync-cadence or conditional-initial-sync knob. Async off-policy admission
is controlled by `max_head_offpolicy_versions`, not by delaying weight sync.
`weight_sync_timeout` controls the hotload timeout.

For the current user-facing lifecycle, see [Dedicated Training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated.md). This file is the trainer-first debugging and recovery reference.

## Transport negotiation

The managed SDK selects RDMA automatically for eligible dedicated training
when the trainer and every inference replica advertise
`supports_rdma_weight_sync=true`. Dedicated LoRA also requires every replica to
advertise `supports_rdma_lora_weight_sync=true`. The create-model and
hotload-status responses carry these capabilities. Missing or unknown values
mean the corresponding capability is unavailable.

Deployments using gateway shard fan-out stay on FILE: its existing aggregated
status identifies that path, and its publication responses do not carry the
RDMA completion protocol. Do not choose a predecessor from its concatenated
shard ledgers.

Generic trainer/deployment overrides are opaque, opt-in testing interfaces;
leave them unset in recipes and provision configs.

RDMA resources are configured by the platform/shape. Do not add `extra_args`,
`extra_values`, or their deployment variants to enable it: these require a
superuser API key. The normal SDK/cookbook path sends none of these overrides;
unconfigured runtimes keep FILE. Experimental trainer topology belongs in the
training shape; public model-handle options do not add launch overrides.

Recipes use `make_weight_sync`, which reads the negotiated
`policy.supports_rdma_weight_sync` property. Create this callback once after attaching the
policy and reuse it before the first rollout and after optimizer steps. Its RDMA
branch calls `policy.weight_sync()`; its FILE branch calls
`saved = policy.save_weights_for_sampler(name)` followed by
`service.hotload_sampler_snapshot(saved.path)`. The cookbook wrapper waits for
completion. [`recipes/rl_loop.py`](../../../training/recipes/rl_loop.py) shows this usage.
The compatibility option
`weight_sync_transport="RDMA"` does not override capability checks. Do not
infer support from an image tag. The SDK's `weight_sync()` rechecks inference
support before each publication. Once RDMA is enabled or selected, capability
loss and publication failures surface as errors. Reused deployments keep their
existing RDMA enablement setting.

Each completed recipe publication logs `Weight sync completed: RDMA` or
`Weight sync completed: FILE`. Use this result to identify the transport that
actually completed; an initial RDMA capability advertisement alone does not
prove a later publication used RDMA.

RDMA publications are temporary and do not create promotable sampler
checkpoints. Keep explicit resumable/promotable checkpoint saves enabled when
needed. The trainer-owned bucket and incremental snapshot chain below describe FILE
publication and durable sampler checkpoints.

## Debug a trainer-first deployment

1. Start the trainer using a training shape. For manual SDK provisioning, keep
   the returned trainer endpoint and its full `job_name`. A backend placement
   decision may take time. The cookbook's `wait_for_trainer_before_deployment=True`
   waits for trainer readiness before allocating rollout capacity.
2. Create the deployment with `DeploymentConfig(hot_load_trainer_job=endpoint.job_name, ...)`.
   With cookbook-managed runs, `build_service_client` makes the attachment.
   Leave placement unset unless the caller needs an explicit trainer region.
   The control plane colocates the rollout deployment.
3. Compare the deployment's `hotLoadTrainerJob` to the trainer's full resource
   name. Verify the account, base model, deployment shape and ready state. On
   resume, pass the intended `DeployConfig(deployment_id=...)` and trainer
   identity. Allow the SDK to finish reattachment and readiness checks.
4. Publish once. For FILE, record the exact sampler snapshot returned and
   check the deployment reports that snapshot. For RDMA, record the version
   returned by `weight_sync()`. Sample only after publication completes.
5. When migrating a deployment-owned run, restore its durable checkpoint to
   a trainer-first run and create the rollout deployment against that trainer.
   Preserve the previous resources until the restored run publishes and samples.

| Gateway or hotload symptom | What to check |
|---|---|
| `hotload flow mismatch` or `hot_load_bucket_url ... conflicts with hot_load_trainer_job` | The existing trainer or rollout uses the deployment-owned layout. Restore from its durable checkpoint and create a trainer-first pair. |
| Snapshot not found or serving reports a different snapshot | Check that the deployment's trainer attachment matches the writer and its bucket. Wait for the rollout to finish reattaching before publishing again. |
| `FILE fallback is not allowed` | Check trainer and rollout readiness and compare their RDMA capabilities. For a dedicated LoRA run, verify RDMA LoRA support on each replica. If the trainer RDMA endpoint returns 404 or 405, use a compatible trainer image. |
| Replay references missing or a shared-store mismatch | Compare the trainer and sampler's `r3_store_id` and inspect the sampler request ID. Keep the validated trainer and deployment versions together and recapture expired references. |
| Serving rejects a replay mask or score-centering capture | Check the rollout image pinned by the exact deployment shape version and the trainer capability for the selected loss. Use a validated shape pair with shared-storage capture support. |

For an internal run, search Chronosphere by trainer name, deployment name and
sampler request ID. Capture the first create, attachment or publication error
along with the shape versions and snapshot identity.

## The knobs

| Field | Default | Meaning |
|---|---|---|
| `first_checkpoint_type` | `"base"` | First sampler save is a full snapshot; subsequent saves can be deltas. Do not change. |
| `weight_sync_timeout` | `600` | Per-hotload timeout in seconds. Bump if you see `Hotload did not complete within 600s` on large models. |
| `dcp_save_interval` | `0` | DCP (optimizer + weights) save cadence for **resume**. Orthogonal to sampler hotload. `0` = off; no intermediate resume points. |
| `dcp_timeout` | `2700` | 45 min default for `save_state` / `load_state_with_optimizer`. |
| `hot_load_transition_type` | unset (server picks `ASYNC`) | How the deployment treats requests that are mid-generation when the swap lands. See [Inference transition](#inference-transition-async-vs-sync). |

## Rollout speculator (SDK only)

The rollout deployment uses the base model's default speculator. `DeployConfig`
only exposes `disable_speculative_decoding` to turn it off. To pick a speculator,
pass `draft_model` (`"mtp"`, `"dspark"`, or a draft model resource name) together
with a positive `draft_token_count` to `FiretitanServiceClient.from_firetitan_config(...)`
directly. `draft_token_count` is required whenever `draft_model` is set. They
apply when the deployment is created, are not reconciled on reattach, and cannot
be combined with `disable_speculative_decoding`.

## On-policy vs off-policy (weight-sync timing)

- `rl_loop` collects one batch, applies one optimizer step, hotloads, and only
  then starts the next batch. Rollouts for step K+1 therefore use the policy
  produced by step K.
- `async_rl_loop` hotloads after every optimizer batch too, while its admission
  gate allows a bounded number of already-admitted behavior-policy versions.

## Prefix-cache policy across turns and weight updates

Set `DeployConfig.reset_prompt_cache` to choose how later, separate requests
reuse prefix KV after a successful sampler hotload:

| Value | Behavior after hotload |
|---|---|
| `new_session` | Existing sessions keep their prefix-cache namespace; new sessions use the new snapshot's namespace. |
| `all` | Every session moves to the new snapshot's namespace; the first later request cannot reuse the old prefix cache. |
| `none` | The cache namespace stays unchanged for existing and new sessions. |

```python
deployment = DeployConfig(reset_prompt_cache="new_session")
```

The shared deployment config defaults to `new_session`; `async_rl_loop` uses
that configurable policy. The strictly synchronous `rl_loop` always uses `all`,
including when a caller supplies a different deployment policy. It finishes its
rollout batch before training, waits for weight sync, then starts the next batch
with a fresh prefix namespace. It copies the supplied config rather than mutating
it. This loop ordering does not select the serving `SYNC` transition or drain
unrelated requests on a shared deployment.

Use at least the SDK version declared by the cookbook's package requirements;
older SDK versions do not provide the required cache-policy API.
The same setting reaches SDK-managed FILE and RDMA hotloads, including explicit
`all`. RDMA requires a trainer runtime that accepts the policy on weight-sync
requests; that runtime also defaults omitted policies to `new_session`.

Supported native serving backends implement these namespace policies. The
multi-LoRA serving backend currently accepts them as no-ops, so setting
`new_session` on that backend does not enable cross-update prefix reuse.

Use a stable `prompt_cache_key` (or `user`) across requests in a rollout. A hit
also needs matching prefix tokens, routing to an available cache, and retained
entries. Cache reuse may carry KV computed under previous weights; it does not
pin the session to those weights. This option does not reset the active KV of
an ongoing request that survives async hotload. The policy changes future
requests even when the weight update itself is asynchronous.

## Inference transition: ASYNC vs SYNC

Weight-sync *timing* decides which policy version a rollout starts on. The
inference transition decides what happens to a generation that is still
streaming when the swap lands. `DeployConfig.hot_load_transition_type` selects
it; leaving it unset lets the control plane choose, and for hot-load
deployments that resolves to `ASYNC`.

| | In-flight generation during a swap |
|---|---|
| `ASYNC` (default) | Paused, then resumed on the same connection. The turn keeps the KV cache built under the old weights and its remaining tokens decode against the new ones, so a single response can span two policy versions. |
| `SYNC` | Allowed to finish on the old weights before the swap; new requests queue. No response ever spans two versions, at the cost of a TTFT stall per hot load. |

`ASYNC` is what you want by default: it keeps the rollout fleet busy, and the
recorded `sampling_logprob` for each token still comes from the version that
actually sampled it, so the RL loops' rollout-logprob clipping anchor stays
correct.

Reach for `SYNC` when a mid-generation version boundary is itself suspect — for
example long multi-turn agentic rollouts where structured output or tool-call
formatting degrades over many steps and you want to rule out mixed-weight KV as
the cause. Expect lower rollout throughput while it is on.

```python
DeployConfig(
    deployment_id="my-sampler",
    hot_load_transition_type="SYNC",
)
```

Changing it on an existing deployment rolls the serving pod. The SDK-managed
path folds that change into the same PATCH as a trainer re-attach, so a resume
that also flips the transition type costs one pod roll rather than two.

Note that neither mode re-prefills: `ASYNC` reuses the old-weight KV and `SYNC`
avoids the overlap entirely. Fireworks has no mode that discards the prefix KV
and recomputes it under the new weights mid-generation.

## Incremental snapshots ARC2

For full-parameter FILE weight sync, the first sampler save is `base` (full weights, ~16 GB for 8B). Subsequent saves are `delta` (XOR diff, ~10× smaller). The SDK-managed sampler backend records this chain automatically — users don't pick per-step.

- LoRA always saves the full adapter regardless of `checkpoint_type` — every LoRA sampler checkpoint is promotable.
- Full-param `delta` saves are **not** promotable. Only `base` saves are. The cookbook's `TrainingCheckpoints.save(promotable=True)` always emits a `base` save, so periodic promotables stay promotable. Recipe FILE weight sync switches to `delta` after the first call — those rows are still visible in `list_checkpoints`, while `promote_latest` picks the most-recent **promotable** row.

## `dcp_save_interval` for resume

Separate from hotload: DCP saves persist the full train state (weights + optimizer) so you can resume training if the job dies. `0` (default) = off — if your run crashes mid-training, there is no intermediate resume point. Set this if your run is long enough that a crash is painful.

## Two deployments, one trainer

On-policy sampler + held-out eval deployment is a common pattern. Both copy the trainer's `hotLoadBucketUrl` at creation, and both can hotload snapshots saved by the same trainer.

```python
sampler = deploy_mgr.create_or_get(DeploymentConfig(
    deployment_id="my-sampler",
    base_model=...,
    hot_load_trainer_job=trainer_endpoint.job_name,
))
eval_dep = deploy_mgr.create_or_get(DeploymentConfig(
    deployment_id="my-eval",
    base_model=...,
    hot_load_trainer_job=trainer_endpoint.job_name,   # same trainer
    min_replica_count=0,                              # scale down when idle
))
```

## Trainer deletion and retention

`DeleteRlorTrainerJob` no longer hard-deletes the trainer row immediately — it transitions the row to `JOB_STATE_DELETED` and marks `DeletionTime`. A background GC hard-deletes rows after **30 days**, aligned with the RL-checkpoints GCS bucket lifecycle.

During the 30-day retention window:
- `list_checkpoints` on the deleted trainer keeps working.
- `promote_checkpoint` keeps working — just pass `name` as usual.
- `ListRlorTrainerJobs` / `ListDpoJobs` filter out `JOB_STATE_DELETED` so customer-facing listings don't surface tombstones.
- A second `DeleteRlorTrainerJob` / `DeleteDpoJob` on a tombstoned row is a no-op; it does **not** reset the retention clock.
- `CancelRlorTrainerJob` / `CancelDpoJob` on a tombstoned row returns `cannot cancel job in state: JOB_STATE_DELETED` — expected, no action needed.

After retention, both `list_checkpoints` and `promote_checkpoint` return `NOT_FOUND`. By that point checkpoints in GCS are also gone, so the behaviour is consistent.

**Practical implication:** you can safely delete a trainer as soon as a run finishes. You have a month to promote checkpoints via the API.

## Promoting a checkpoint

Pass the exact `name` from the checkpoint list while the trainer is alive or within retention.

```python
entries = client.list_checkpoints(job_id=trainer_endpoint.job_name)
entry = max((e for e in entries if e["promotable"]), key=lambda e: e["createTime"])
client.promote_checkpoint(
    name=entry["name"],   # 4-segment resource name, pass verbatim
    output_model_id="my-promoted-model",
    base_model=base_model,
)
```

The gateway resolves checkpoint storage from the trainer recorded in the checkpoint name.

## Runtime hotload mismatch or stale deployment attachment

Runtime hotload failures can still happen after create-time validation passes. The user-facing SDK signal is that the SDK asked the deployment to load snapshot `S`, but the deployment status still reports a different current snapshot or never becomes ready for `S`.

Typical causes:

- The caller created trainer A plus deployment D, then later attached trainer B to the same deployment outside the SDK-managed cookbook path.
- A full-parameter run sent a `delta` snapshot while the deployment still had an old base snapshot loaded.
- A LoRA run saved a full adapter under trainer B, but deployment D was still attached to trainer A. LoRA has no base/delta repair step; fix the deployment attachment.

Recovery:

1. Full-parameter training: force the next sampler sync to `checkpoint_type="base"` before collecting more rollouts. If the trainer process already stopped, resume from the latest DCP checkpoint first. In the SDK-managed recipes, use the normal recipe path; if calling the SDK directly, call `save_weights_for_sampler(..., checkpoint_type="base")` and hotload that returned snapshot identity.
2. LoRA training: `checkpoint_type` is not the fix. LoRA sampler saves are full adapter snapshots, so a snapshot mismatch means the deployment attachment is stale. Reattach the deployment through the SDK-managed cookbook path or create a fresh deployment tied to the current trainer.
3. If reusing a deployment intentionally, use `DeployConfig(deployment_id=...)` with the cookbook service path so the SDK patches `hot_load_trainer_job`, waits for the serving restart, and resets the sampler chain before the next rollout.
4. If there is no usable DCP checkpoint, there is no clean "continue from last step" recovery. Start a fresh run or work with Fireworks support to inspect server-side state.

Agent-assisted debugging: first search the Fireworks training cookbook skill for hotload recovery. If the cookbook is not available locally, clone `https://github.com/fw-ai-external/cookbook` and search `cookbook/skills`.

## See also

- [Dedicated Training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated.md) — user-facing lifecycle
- [`sdk-checkpoints.md`](sdk-checkpoints.md) — checkpoint and resume behavior
- Low-level `WeightSyncer` lifecycle: `fireworks.training.sdk.weight_syncer.WeightSyncer` (installed under `src/fireworks/training/sdk/weight_syncer.py`). Recipes use SDK-managed service hotload directly.
- `save_weights_for_sampler_ext`, `save_state`, `list_checkpoints`: `fireworks.training.sdk.client.FiretitanTrainingClient`.
- Trainer + deployment managers this flow depends on: `fireworks.training.sdk.trainer.TrainerJobManager` and `fireworks.training.sdk.deployment.DeploymentManager`.
