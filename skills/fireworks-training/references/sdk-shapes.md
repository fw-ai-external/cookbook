# Training and deployment shapes — always use a profile

> Dedicated Training and on-demand deployments only. Serverless Training
> attaches to the shared pool and takes no shape.

Shapes are the required entry point for both trainer and deployment. Cookbook
trainer config rejects `accelerator_type` and `accelerator_count`; clients
cannot select trainer accelerators, so set `training_shape_id` instead.
`node_count` and `custom_image_tag` remain advanced controls but should not be
hand-set when a shape is in use.

## Training shape

Whether `cfg.trainer.training_shape_id` can stay unset depends on whether the
run creates a deployment:

| Run | `training_shape_id` unset |
| --- | --- |
| Trainer only (SFT, DPO, ORPO, embedding) | Works: backend trainer creation picks a validated shape. |
| Creates a new dedicated rollout deployment (RL or distillation) | Works when `deployment.deployment_shape` explicitly supplies a compatible shape. Otherwise set a training shape whose validated profile links one. |
| Reattaches to an existing `deployment.deployment_id` | Works: the deployment already has its shape. |

For a new dedicated rollout deployment, prefer the validated trainer and
deployment shape pair. Set its training shape:

```python
cfg.trainer.training_shape_id = "accounts/fireworks/trainingShapes/qwen3-4b-minimum"
```

To find one, call the read-only lookup in `training.utils`, which returns
a compatible validated shape for the base model and tuning mode. This example
assumes a configured `TrainerJobManager` named `mgr`:

```python
from training.utils import auto_select_training_shape

shape_id = auto_select_training_shape(
    mgr, base_model="accounts/fireworks/models/qwen3-4b", lora_rank=0, public_only=True
)
profile = mgr.resolve_training_profile(shape_id)
if not profile.deployment_shape_version:
    raise ValueError("Choose a training shape with a linked deployment shape.")
cfg.trainer.training_shape_id = shape_id
```

Use the run's LoRA rank and context length for the lookup. If the result lacks a
linked deployment shape, choose a compatible shape explicitly.

The recipe then does:

```python
profile = rlor_mgr.resolve_training_profile(resolved_training_shape_id)
# profile.training_shape_version
# profile.deployment_shape_version
# profile.max_supported_context_length
# profile.accelerator_type, profile.node_count, ...  (read, do not copy to cfg)
```

See `training/recipes/async_rl_loop.py` and
`FiretitanServiceClient.from_firetitan_config(...)` for the refactored
trainer/deployment provisioning path.

## Deployment shape

**Do not create deployments without a [shape](https://docs.fireworks.ai/faq-new/deployment-infrastructure/what-is-a-deployment-shape).** Shapeless deployments are the most common cause of failed deployment creations, and the [shapeless path](https://docs.fireworks.ai/guides/ondemand-deployments#explicitly-creating-a-deployment-without-a-shape-advanced-users-only) will be deprecated. Find a deployable shape, or pass `default` to let the server pick one.

The SDK uses an explicit `cfg.deployment.deployment_shape` if supplied.
Otherwise it reads the deployment shape from the selected training profile.
Use a compatible versioned deployment shape for an explicit override, and read
the resolved value from the service:

```python
service = build_service_client(...)
training_client = service.create_training_client(...)
deployment_shape = service.deployment_shape
```

That is a **versioned** path (`accounts/fw/deploymentShapes/ds-x/versions/abc123`).
A new Cookbook deployment requires a shape from one of these two sources. Overriding individual fields on top of a shape — e.g. `replica_count` — is
supported; the shape owns accelerator selection, so manual accelerator fields
are never forwarded. If no shape fits a workload, [contact us](https://fireworks.ai/contact).

## Reference-model shape (DPO and KL forks)

A frozen reference is provisioned only for losses that consume reference
logprobs, such as DPO, IGPO and RL forks that add reference KL. `rl_loop` and
`async_rl_loop` train without one and reject reference settings.

For **full-parameter** training with a frozen reference, leave `cfg.trainer.reference_training_shape_id` unset to let backend trainer creation auto-select a compatible `LORA_TRAINER` shape. Set it explicitly only when you need an override; it should be a LoRA-capable shape and can share the same shape as the policy.

For **LoRA** (`lora_rank > 0`), two valid options:
- **Shared session (recommended, saves GPUs)**: leave `cfg.trainer.reference_training_shape_id` unset. `service.create_reference_client(...)` reuses the policy session with the adapter disabled for reference logprobs — no separate trainer, no extra GPUs.
- **Separate LoRA-capable ref trainer**: set `cfg.trainer.reference_training_shape_id` to a `LORA_TRAINER` shape (typically the same as the policy shape). The SDK provisions a frozen reference runtime on its own GPUs and requests `trainer_mode=LORA_TRAINER` shape matching.

Use the shared reference option for LoRA. For full-parameter training, select a compatible reference profile.

## Listing available shapes

Deployment shapes through the SDK:

```python
client.deployment_shape_versions.match_for_model("<MODEL>")   # deployable shape versions
```

Training shapes have no `Fireworks()` list call; cookbook recipes resolve them
through `FireworksClient.resolve_training_profile`. Listing them is a read-only
`firectl` call, fine for the agent when installed and otherwise for the user:

```bash
firectl training-shape list                       # alias: firectl ts list
firectl deployment-shape-version match --model <MODEL>
```

`match` returns the validated shapes the account can actually deploy the model on (server-side rules: exact-model matches over param-bucket siblings, embedding separation, FP4 filtering, PEFT addons resolved to their base model). Pass `--enable-addons` for multi-LoRA serving. Use `deployment-shape-version list --base-model <MODEL>` only to browse the raw inventory.

Always create deployments against a [shape](https://docs.fireworks.ai/faq-new/deployment-infrastructure/what-is-a-deployment-shape) returned by `match`.

Or programmatically via `FireworksClient` — see the SDK docs linked from the repo README.

## Do not pin a `/versions/<id>`

Pass the bare shape path `accounts/fireworks/trainingShapes/<shape>`. The platform auto-selects the latest validated version for you. Hand-picking a version is almost always wrong:

- The platform only serves validated versions — a versioned ref cannot force an unvalidated one.
- Pinning locks the run to a stale version and prevents the platform from rolling the shape forward when a better-validated image lands.

For the full list of shapes in your account, run `firectl training-shape list` (above).

## When `resolve_training_profile` raises `Failed to resolve latest validated training shape`

This means the shape currently has no validated version at all — usually a transient state right after a shape update. Pinning to an older `/versions/<id>` won't help. Retry after a short wait; if it persists, reach out to Fireworks support.
