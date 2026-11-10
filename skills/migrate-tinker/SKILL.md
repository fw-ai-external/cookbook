---
name: migrate-tinker
description: >-
  Port a Tinker SDK or Tinker Cookbook training loop to the native Fireworks
  Training SDK. Adapt clients, data types, loss inputs, sampling, checkpoints,
  and cleanup while preserving the experiment. Use configure to launch or
  resume training and debug to investigate a running Fireworks job.
---

# Migrate a Tinker training loop

Do the port in the user's existing application or fork. Preserve the dataset,
objective, tokenization, target shift, masks, normalization, rewards, schedule,
and evaluation unless a requested change requires otherwise.

The Fireworks SDK owns its classes, transport, futures, and errors. Familiar
method names are not a promise of drop-in compatibility with Tinker. Do not
replace `tinker` globally or send a current Tinker client to a Fireworks URL.

## Establish the migration boundary

Record the source Tinker/cookbook versions, target Fireworks SDK version, and
entrypoint. Inspect the installed target signatures and the SDK's
[compatibility guide](https://github.com/fw-ai-external/python-sdk/blob/main/src/fireworks/training/sdk/TINKER_COMPATIBILITY.md).
Use [compatibility and adaptation](references/compatibility.md) for the concrete
type, API, publication, and checkpoint differences. Its versioned observations
are a starting point; recheck them when targeting a different SDK release.

Find imports and constructors inside helpers as well as the main script:

```bash
rg -n 'import tinker|from tinker|ServiceClient|save_weights_and_get_sampling_client|create_rest_client|isinstance|except .*Tinker' .
```

Resolve the selected model and whether the run needs serverless or dedicated
training from the [current catalog](https://docs.fireworks.ai/fine-tuning/models).
Keep the tokenizer/renderer identity separate from the Fireworks model resource.
Do not substitute a model or reference policy merely to make a port execute.

## Implement the port

1. Construct native `sdk.Datum`, `sdk.ModelInput`, `sdk.TensorData`, optimizer
   parameters, and sampling parameters. Update exception handlers and checks
   that depend on Tinker class identity.
2. Replace service construction explicitly. Serverless attaches to the pool;
   dedicated uses SDK-managed provisioning. Leave region unset unless supplied.
3. Preserve training operation order and normalization. Adapt sampler
   publication, auxiliary references/teachers, resume, and teardown separately.
4. Update callback and checkpoint helpers that still create Tinker objects or
   call unsupported methods. Keep logical checkpoint names and local data
   cursors outside server-returned checkpoint references.

For a maintained replacement loop, choose the closest entry from
[cookbook code](../fireworks-training/references/cookbook-code.md). For the older
Fireworks `InfraConfig`/`setup_infra` migration, use
[SDK migration](../fireworks-training/references/sdk-migrate.md).

## Validate and report

Start on CPU: inspect native batch values, encoded shapes, masks, normalization,
and method binding. Exercise real local serializers/futures when the changed
boundary warrants it. A mock trainer proves request behavior, not model math.
Keep one-off audit scripts and run evidence local unless the user wants them
committed.

If the user requests live validation, hand off to
[configure](../configure/SKILL.md) with the selected model, infrastructure, tiny
dataset, operation/token bounds, timeout, and existing approval. Validate one
update, required token outputs, snapshot sampling, resume, and quota/resource
cleanup. Do not claim every recipe or full provider parity from a smoke test.

Report the changed integration points, remaining unsupported features,
validation evidence, and runtime-dependent checks. The public
[migration page](https://docs.fireworks.ai/fine-tuning/training-api/migrate-from-tinker)
is the high-level overview; keep implementation detail in this skill.
