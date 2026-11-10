# Tinker compatibility and adaptation

Use the installed SDK and the
[SDK compatibility guide](https://github.com/fw-ai-external/python-sdk/blob/main/src/fireworks/training/sdk/TINKER_COMPATIBILITY.md)
as the source of truth. The detailed differences below were checked with
Fireworks SDK **1.2.30**, Tinker **0.23.0** (the earlier compatibility baseline), and
Tinker **0.35.0**. Recheck signatures and backend capabilities on other versions.

## Types and application boundaries

Use native imports in application-owned code:

```python
from fireworks.training import sdk
from fireworks.training.sdk import types
```

`sdk.Datum` and `types.Datum` are the same native type. They are not
`tinker.Datum`. Returned objects and exceptions also belong to Fireworks;
catch `sdk.TrainingError` or the appropriate exported SDK API error instead of
Tinker errors. Replace foreign annotations, constructors, and `isinstance`
checks inside callbacks and imported helpers as well as at the entrypoint.

For encoded text and dense tensors, an explicit boundary conversion is possible:

```python
def to_fireworks_datum(datum):
    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints(datum.model_input.to_ints()),
        loss_fn_inputs={
            key: sdk.TensorData.from_numpy(value.to_numpy())
            for key, value in datum.loss_fn_inputs.items()
        },
    )
```

This is not a general adapter: it drops multimodal chunks, sparse padding
semantics, provenance, and routing/sampling references. Build native datums for
those cases instead of silently discarding fields.

| Boundary | Adaptation |
|---|---|
| Dense tensors | Public dtypes are float32 and int64. NumPy/torch conversion may change source precision; check shapes, empty tensors, and numeric values. |
| Sparse tensors | Native zero-padding CSR helpers do not accept Tinker 0.35's `pad_value`. Materialize the source with its intended pad value and use a dense native tensor to preserve nonzero padding. |
| Provenance | Tinker 0.35's `Datum.model_input_spans` and `loss_fn_input_spans` are absent from the native constructor. Preserve application metadata separately; a plain text adapter does not preserve provenance semantics. |
| Model input | Use native encoded text or inline image chunks. Opaque image-asset pointers and DMel audio datums are rejected. Generic type availability is not model execution support. |
| Routing | Preserve Fireworks routing matrices and sampling references when the rollout/trainer contract requires them. A text-only conversion loses them. |
| Sampled sequences | Common token, logprob, and stop-reason access exists. Tinker 0.35's required `sequence_id` and sampled-token top-k data are absent in the native type. |
| Sample responses | Newer `target_prompt_logprobs` and `prompt_alt_tokens` fields are absent. Adapt downstream consumers instead of inventing values. |

Preserve next-token alignment: inputs are `tokens[:-1]`, targets are
`tokens[1:]`, and every loss field aligns with the targets. For chat SFT,
context tokens are masked; for RL, keep behavior logprobs, advantages, and
required routing aligned with the sampled tokens. Retain the original loss
normalization, including across accumulated microbatches.

## Client setup and signatures

Use a Fireworks API key and a model resource such as
`accounts/fireworks/models/<model>`. The source model owner does not select the
account that owns the run; ownership comes from the authenticated key.

Serverless uses the public training endpoint:

```python
service = sdk.FiretitanServiceClient(
    api_key=api_key,
    base_url="https://api.fireworks.ai/training/v1/serverless",
)
client = service.create_lora_training_client(base_model=base_model, rank=rank)
```

For dedicated training, use
`sdk.FiretitanServiceClient.from_firetitan_config(...)` with a compatible
`training_shape_id`, matching tokenizer and LoRA settings, and explicit cleanup
policies. Set `create_deployment=False` for trainer-only work. RL/sampling
evaluation needs the corresponding deployment configuration. Follow the
[service reference](https://docs.fireworks.ai/fine-tuning/training-api/reference/service-client)
for the installed SDK; leave region unset unless the user supplied one and do
not infer placement from a shape.

Use keyword arguments when porting client creation. The positional argument
after `train_unembed` is `user_metadata` in Tinker, but Fireworks inserts
`alpha` there. A positional metadata dictionary can bind to the wrong field.

| Tinker call or option | Fireworks 1.2.30 boundary |
|---|---|
| `create_lora_training_client(..., optimizer=...)` | Newer optimizer-selection argument is absent. Resolve a supported optimizer setup explicitly. |
| `save_state(..., user_metadata=...)` | Argument is absent; keep application checkpoint metadata separately. |
| `sample(..., topk_sample_logprobs=..., target_prompt_logprobs=..., prompt_alt_tokens_k=..., prompt_logprobs_last_n=...)` | Newer arguments are absent from the native signature. |
| `compute_logprobs(..., prompt_logprobs_last_n=...)` | Native signature accepts the prompt only. |
| `service.close("success", detail=...).result()` | Native `service.close()` is synchronous and accepts neither argument. |
| `save_weights_and_get_sampling_client()` | Combined operation is absent; use explicit publication below. |
| `copy_weights`, `save_weights_external`, console-URL helpers | Newer methods are absent; use supported Fireworks lifecycle/checkpoint operations. |
| Tinker REST metadata, archive download, publication, deletion, TTL, project assignment | Do not pass these through. Use the documented Fireworks control plane where an equivalent exists. |

Training methods retain future-style behavior. A synchronous remote operation
typically finishes with `.result()`. For training `_async` methods, the first
await returns an API future that must also be awaited. Sampling `sample_async`
returns its result with one await. Check the installed signatures and exercise
the application's actual call pattern.

## Losses and token metrics

Keep the forward/backward → optimizer sequence. Custom callbacks receive
native datums and differentiable CPU logprob tensors and return a scalar loss
plus metrics. Do not detach the values used to compute the loss.

`optim_step(..., grad_accumulation_normalization=...)` can normalize gradients
server-side. Its default leaves them unchanged. Do not add normalization when
the source loss is already averaged. Follow
[loss inputs and normalization](../../fireworks-training/references/training-api-losses.md)
for objective-specific contracts.

Supported single-target cross-entropy backward paths can return per-token
`loss_fn_outputs[i]["logprobs"]`. Some older models may not support this feature;
[contact us](https://fireworks.ai/contact-training) if you need it. Check the
returned outputs for the model and loss configuration your loop uses.

Check the outputs consumed by the source loop before starting a long run. If
they are absent, contact us or use an explicit forward for token metrics.
A pre-update metric forward belongs before `optim_step` and
adds GPU cost. Never fabricate token logprobs from aggregate metrics. Compare
numerical tolerances separately from data-shape compatibility.

Fireworks also supports `precomputed_forward` on `forward_backward_custom`.
Reuse only identical data/order and unchanged policy weights. This does not
add a `precomputed_logprobs` argument to ordinary `forward_backward`.

## Publish weights for sampling

Replace the combined Tinker helper with two explicit operations:

```python
snapshot = client.save_weights_for_sampler("step-0001").result().path
sampler = service.create_sampling_client(model_path=snapshot, tokenizer=tokenizer)
try:
    response = sampler.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=sdk.SamplingParams(max_tokens=256, temperature=0.7),
    ).result()
finally:
    sampler.close()
```

The tokenizer and renderer must match the model. Preserve stop conditions and
completion budgets. `topk_prompt_logprobs` is unsupported on this sampler.

- Serverless binds the sampler to the returned session snapshot.
- Dedicated syncs the snapshot to its hot-load deployment. Saving alone does
  not update that deployment, and an older handle is not an independent
  historical deployment. Coordinate publication with in-flight rollouts.
- For full-parameter training, use the maintained recipe's weight-publication
  integration and verify support for the selected model and training setup.

## Checkpoints, references, and concurrency

| Purpose | Operation |
|---|---|
| Sampler snapshot | `save_weights_for_sampler(name).result().path` |
| Resumable weights/optimizer state | `save_state(name).result().path` |
| Weights-only warm start | `load_state(path).result()` |
| Restore weights and optimizer | `load_state_with_optimizer(path).result()` |
| New run from saved training state | Service `create_training_client_from_state` or its `with_optimizer` variant, with appropriate infrastructure setup |
| Deployable artifact | Promote a supported sampler checkpoint, then deploy separately |

Persist the returned reference and local data cursor, schedule, renderer, and
configuration. A Tinker checkpoint URI is not a Fireworks checkpoint reference.
Dedicated checkpoint names may be server-assigned `step-N`; keep logical-name
mappings separately. TTL, state overwrite, and weights access tokens are not
supported. Serverless checkpoint names must fit the documented promotion
length constraints. Follow
[checkpoint guidance](../../fireworks-training/references/sdk-checkpoints.md)
for the selected infrastructure.

Do not equate repeated client creation with independent adapters. Legacy
managed `lora_rank` configuration owns one policy; `max_lora_rank` configures
multi-model capacity where supported. Do not combine it with service-level
`lora_rank`/`lora_alpha`. Verify backend capacity and account limits before
porting a concurrent sweep.

A LoRA base reference can share the policy trainer through
`service.create_reference_client`. Use it only when the intended reference is
that frozen base. A fine-tuned reference, full-parameter policy, teacher, or
fixed opponent may need distinct resource setup. Never silently replace the
intended reference with base weights.

## Teardown and validation

On serverless SDK 1.2.30, closing the local service does not immediately release
the concurrent-run quota slot. Complete needed session-scoped checkpoint work,
close samplers, then explicitly release the session:

```python
try:
    # Run the migrated experiment.
    ...
finally:
    try:
        service.delete_training_session()
    finally:
        service.close()
```

Deletion can leave retained metadata in `EXPIRED` state; a successful cleanup
need not make subsequent reads return 404. Only release sessions owned by the
current experiment. For dedicated services, set trainer/deployment cleanup
policies explicitly and verify the created resources' final state.

Validate CPU data and signatures before remote work. For an approved live
smoke, check one real update with finite metrics, the token outputs the loop
needs, publication and stop behavior, checkpoint restoration, and cleanup.
Check restored model values separately from optimizer state and the local data
cursor. A successful restore API alone does not independently verify optimizer
buffers. Record unsupported features and untested paths; a short smoke does
not establish numerical or quality parity with the Thinking Machines service.
