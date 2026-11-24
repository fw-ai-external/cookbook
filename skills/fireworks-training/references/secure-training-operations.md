# Secure training operations

*Source of truth: live [Training overview](https://docs.fireworks.ai/fine-tuning/finetuning-intro.md), [Data handling](https://docs.fireworks.ai/guides/security_compliance/data_handling.md), and [Data security](https://docs.fireworks.ai/guides/security_compliance/data_security.md). Public policy, retention, and current product support stay in those docs.*

Use this reference for agent-assisted BYOB and CMEK setup. Always confirm the current cloud, method, and resource support in live docs before preparing commands.

## Bring your own bucket

BYOB lets a training run read a dataset from customer-managed object storage.
For a recipe that accepts GCS datasets, use the dataset path as input. See the
[migration guide](https://docs.fireworks.ai/fine-tuning/migrate-from-managed-training)
for current input formats and the [Data security guide](https://docs.fireworks.ai/guides/security_compliance/data_security.md)
for your cloud's access requirements.

## Customer-managed encryption keys

CMEK provides customer-managed encryption for supported training artifacts.
Confirm support for your training method and cloud before starting a run. Follow
the [Data security guide](https://docs.fireworks.ai/guides/security_compliance/data_security.md)
for setup and lifecycle guidance.

## Related

- Existing managed RFT jobs: `managed-rft-operations.md`
- Training API agent trajectories: `rl-agentic.md`
- Failure handling: `error-reference.md`
