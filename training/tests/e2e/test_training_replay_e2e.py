"""Exercise replay capture and trainer consumption on one explicitly selected small shape pair."""

import asyncio
import logging
import math
import os
import uuid
from contextlib import ExitStack

import pytest
from fireworks.training import sdk
from fireworks.training.sdk.routing import RoutingReferences

from training.utils import DeployConfig, ReconnectableClient, TrainerConfig, build_service_client
from training.utils.rl.algorithm import server_reinforce
from training.utils.rl.losses import combine_prompt_groups
from training.utils.rl.rollout import Rollout, rollout_to_prompt_group, sampled_completion_to_rollout_run
from training.utils.service import make_weight_sync
from training.utils.tokenizers import load_deployment_tokenizer

logger = logging.getLogger(__name__)


@pytest.mark.e2e
@pytest.mark.timeout(5400)
def test_training_replay_matrix(sdk_managers):
    """FIREWORKS_E2E_REPLAY=dense or moe opts in; require explicit matching shapes."""
    track = os.environ.get("FIREWORKS_E2E_REPLAY")
    if not track:
        pytest.skip("Set FIREWORKS_E2E_REPLAY=dense or moe to run live replay tests")
    if track not in {"dense", "moe"}:
        raise ValueError("FIREWORKS_E2E_REPLAY must be dense or moe")
    account = os.environ["FIREWORKS_E2E_ACCOUNT"]
    trainer_mgr, deploy_mgr = sdk_managers
    if trainer_mgr.account_id != account or deploy_mgr.account_id != account:
        raise ValueError("Test credential belongs to a different account")
    model = os.environ["FIREWORKS_E2E_MODEL"]
    tokenizer_model = os.environ["FIREWORKS_E2E_TOKENIZER_MODEL"]
    deployment = DeployConfig(
        deployment_id=f"replay-e2e-{uuid.uuid4().hex[:12]}",
        deployment_shape=os.environ["FIREWORKS_E2E_DEPLOYMENT_SHAPE"],
        tokenizer_model=tokenizer_model,
    )
    tokenizer = load_deployment_tokenizer(deployment)
    rank = int(os.environ.get("FIREWORKS_E2E_REPLAY_LORA_RANK", "8"))
    with ExitStack() as stack:
        service = build_service_client(
            api_key=trainer_mgr.api_key,
            base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"),
            inference_url=os.environ.get("FIREWORKS_INFERENCE_URL"),
            additional_headers=None,
            base_model=model,
            tokenizer_model=tokenizer_model,
            max_lora_rank=rank,
            max_context_length=512,
            learning_rate=1e-4,
            trainer=TrainerConfig(
                training_shape_id=os.environ["FIREWORKS_E2E_TRAINING_SHAPE"],
                region=os.environ.get("FIREWORKS_E2E_REGION"),
            ),
            deployment=deployment,
            cleanup_trainer_on_close=True,
            cleanup_deployment_on_close="delete",
        )
        stack.callback(service.close)
        handle = service.create_training_client(model, lora_rank=rank)
        job = trainer_mgr.get(service.trainer_job_id)
        assert job["name"].startswith(f"accounts/{account}/rlorTrainerJobs/")
        attached = deploy_mgr.get(service.deployment_id)
        assert attached is not None and attached.name.startswith(f"accounts/{account}/deployments/")
        assert attached.hot_load_trainer_job == job["name"]
        logger.info("Verified trainer-first attachment %s -> %s", attached.name, job["name"])
        assert handle.supports_router_replay is (track == "moe")
        assert handle.r3_store_id, "Sampler and trainer need a qualified shared store"
        policy = ReconnectableClient.from_training_client(
            handle,
            base_model=model,
            lora_rank=rank,
            job_id=service.trainer_job_id,
            service=service,
        )
        publish = make_weight_sync(policy, service, deployment)
        last_optimizer_version = -1

        def publish_and_check(name: str, **kwargs):
            nonlocal last_optimizer_version
            result = publish(name, **kwargs)
            if result is not None and result.optimizer_version is not None:
                assert result.optimizer_version > last_optimizer_version
                last_optimizer_version = result.optimizer_version
                logger.info("Published optimizer version %s", result.optimizer_version)
                return
            status = deploy_mgr.hotload_check_status(service.deployment_id, model)
            identities = set()
            for replica in status["replicas"]:
                if replica.get("current_snapshot_identity"):
                    identities.add(replica["current_snapshot_identity"])
                identities.update(
                    adapter["identity"]
                    for adapter in replica.get("loaded_adapters", [])
                    if adapter.get("status") == "loaded" and adapter.get("identity")
                )
            assert any(name in identity for identity in identities), (name, identities)
            logger.info("Verified FILE publication %s", name)

        publish_and_check("replay-initial", checkpoint_type="base")
        sampler = service.create_deployment_sampler(tokenizer=tokenizer, training_client=handle)
        stack.callback(sampler.close)
        assert sampler.r3_store_id == handle.r3_store_id
        prompt = tokenizer.encode("The next even number after eight is")

        with asyncio.Runner() as runner:
            for step, (mask, estimator) in enumerate(
                [
                    (False, None),
                    (True, None),
                    (False, "top_sampling"),
                    (True, "top_sampling"),
                    (False, "extra_sampling"),
                    (True, "extra_sampling"),
                ]
            ):
                logger.info("Validating %s capture step %s (mask=%s, estimator=%s)", track, step, mask, estimator)
                score_centering = {
                    "top_sampling": server_reinforce.TopSamplingSC(4),
                    "extra_sampling": server_reinforce.ExtraSamplingSC(4),
                }.get(estimator)
                objective = server_reinforce.ReinforceConfig(
                    score_centering=score_centering,
                    target_logprob_support="sampling_support" if mask else "full_vocabulary",
                )
                server_reinforce.check_trainer_support(handle, objective)
                captures = runner.run(
                    sampler.sample_with_prompt_tokens(
                        prompt,
                        n=2,
                        max_tokens=8,
                        temperature=1.0,
                        echo=False,
                        include_routing_matrix=track == "moe",
                        top_p=0.95,
                        top_k=32,
                        **server_reinforce.sampling_kwargs(objective),
                    )
                )
                assert len(captures) == 2
                for sample in captures:
                    assert sample.completion_len > 0
                    for channel, present in (
                        ("routing_matrices", track == "moe"),
                        ("sampling_mask_references", mask),
                        ("top_sampling_references", estimator == "top_sampling"),
                        ("sample_extra_references", estimator == "extra_sampling"),
                    ):
                        refs = getattr(sample, channel)
                        if not present:
                            assert refs is None, channel
                            continue
                        assert isinstance(refs, RoutingReferences) and refs.files, channel
                        assert all(f["store_id"] == handle.r3_store_id for f in refs.files), channel
                        assert len(refs) == sample.completion_len, channel
                group = rollout_to_prompt_group(
                    Rollout(
                        runs=[
                            sampled_completion_to_rollout_run(sample, reward=float(i))
                            for i, sample in enumerate(captures)
                        ]
                    )
                )
                assert group is not None
                batch = combine_prompt_groups([group])
                datums = server_reinforce.build_datums(batch, objective)
                if track == "moe":
                    for datum in datums:
                        assert datum.model_input.routing_references is not None
                        refs = RoutingReferences.from_dict(datum.model_input.routing_references)
                        assert refs.length == datum.model_input.length and refs.files
                check_support = mask and estimator is None
                if check_support:
                    # Score the same tokens and routes at the same weights to
                    # isolate the change in normalization due to the support.
                    unmasked = policy.forward(batch.data, "cross_entropy").loss_fn_outputs
                result = policy.forward_backward(
                    datums, server_reinforce.LOSS_FN, loss_fn_config=server_reinforce.loss_fn_config(objective)
                )
                assert len(result.loss_fn_outputs) == len(datums)
                shift_error = 0.0
                zero_shift_error = 0.0
                for i, (datum, sample, output) in enumerate(zip(datums, captures, result.loss_fn_outputs, strict=True)):
                    scores = output["logprobs"].data
                    assert len(scores) == datum.model_input.length
                    response_mask = datum.loss_fn_inputs["response_mask"].data
                    active = [value for value, used in zip(scores, response_mask, strict=True) if used]
                    assert len(active) == sample.completion_len and all(math.isfinite(value) for value in active)
                    if check_support:
                        raw_scores = unmasked[i]["logprobs"].data
                        active_raw = [value for value, used in zip(raw_scores, response_mask, strict=True) if used]
                        rollout_raw = sample.inference_logprobs
                        rollout_masked = sample.sampling_logprobs
                        assert rollout_raw is not None and rollout_masked is not None
                        trainer_shift = [masked - raw for masked, raw in zip(active, active_raw, strict=True)]
                        rollout_shift = [masked - raw for masked, raw in zip(rollout_masked, rollout_raw, strict=True)]
                        logger.info("Support normalization shift (trainer=%s, rollout=%s)", trainer_shift, rollout_shift)
                        assert trainer_shift == pytest.approx(rollout_shift, abs=0.15)
                        shift_error += sum(
                            abs(actual - expected) for actual, expected in zip(trainer_shift, rollout_shift, strict=True)
                        )
                        zero_shift_error += sum(map(abs, rollout_shift))
                if check_support:
                    # A no-op mask can fall within the absolute tolerance when
                    # only a small amount of probability mass is removed.
                    assert shift_error < 0.5 * zero_shift_error
                    k3 = server_reinforce.metrics(batch, result, objective)["reinforce_kl"]
                    logger.info("Masked trainer/rollout K3 %.6f", k3)
                    assert 0 <= k3 <= 0.03
                policy.optim_step(
                    sdk.AdamParams(learning_rate=1e-4), grad_accumulation_normalization="num_loss_tokens"
                )
                publish_and_check(f"replay-step-{step}")
                logger.info("Trained and published %s capture step %s", track, step)
            (updated,) = runner.run(sampler.sample_with_prompt_tokens(prompt, n=1, max_tokens=4, logprobs=True))
            assert updated.completion_len > 0
