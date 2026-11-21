"""Exercise the actual sampled-completion to server-REINFORCE datum path."""

import argparse

import pytest
from fireworks.training.sdk.routing import RoutingReferences
from fireworks.training.sdk.sampling import SampledCompletion
from training.utils.rl.algorithm import server_reinforce as objective
from training.utils.rl.rollout.renderer import sampled_completion_to_rollout_run
from training.utils.rl.rollout.types import Rollout, rollout_to_prompt_group
from training.utils.rl.losses import combine_prompt_groups


def test_cli_estimator_arguments_are_exclusive():
    parser = argparse.ArgumentParser()
    objective.add_cli_args(parser)
    for args in (
        ["--sc-num-samples", "4"],
        [
            "--loss-fn",
            "reinforce",
            "--score-centering-estimator",
            "extra_sampling",
            "--sc-top-sampling-size",
            "4",
        ],
    ):
        with pytest.raises(ValueError):
            objective.from_cli_args(parser.parse_args(args))
    config = objective.from_cli_args(
        parser.parse_args(
            [
                "--loss-fn",
                "reinforce",
                "--score-centering-estimator",
                "extra_sampling",
                "--weighting",
                "mis",
                "--sampler-kl-coeff",
                ".01",
            ]
        )
    )
    assert objective.sampling_kwargs(config) == {
        "logprobs": True,
        "sample_extra_tokens": 16,
        "sample_extra_format": "parquet_v1",
    }
    assert objective.loss_fn_config(config)["weighting"] == "mis"


@pytest.mark.parametrize("weighting", ["none", "tis", "mis"])
def test_config_has_independent_weighting_and_kl(weighting):
    config = objective.ReinforceConfig(
        objective.ExtraSamplingSC(4), weighting=weighting, sampler_kl_coeff=0.03
    )
    assert objective.loss_fn_config(config) == {
        "score_centering": True,
        "score_centering_estimator": "extra_sampling",
        "weighting": weighting,
        "sampler_kl_coeff": 0.03,
        "target_logprob_support": "full_vocabulary",
    }


@pytest.mark.parametrize("low,high", [(-1, 2), (2, 1), (0, 0), (0, float("nan"))])
def test_invalid_weight_bounds_fail_locally(low, high):
    with pytest.raises(ValueError):
        objective.ReinforceConfig(weight_low=low, weight_high=high)


def test_completion_to_datum_keeps_prefix_alignment_and_zero_advantage_mask():
    file = {"store_id": "test", "file_id": "extra_sampling", "format": "parquet_v1", "row_count": 2}
    completion = SampledCompletion(
        text="ok",
        full_tokens=[1, 2, 3, 4, 5],
        prompt_len=3,
        finish_reason="stop",
        sampling_logprobs=[-0.2, -0.4],
        sample_extra_references=RoutingReferences.from_dict(
            {
                "length": 2,
                "files": [file],
                "spans": [
                    {
                        "input_token_start": 0,
                        "count": 2,
                        "file_index": 0,
                        "file_row_start": 0,
                    }
                ],
            }
        ),
    )
    run = sampled_completion_to_rollout_run(completion, reward=0.0)
    group = rollout_to_prompt_group(
        Rollout(runs=[run]), advantage_fn=lambda rewards: rewards
    )
    batch = combine_prompt_groups([group])
    config = objective.ReinforceConfig(
        objective.ExtraSamplingSC(2), sampler_kl_coeff=0.01
    )
    datum = objective.build_datums(batch, config)[0]
    references = RoutingReferences.from_dict(datum.model_input.sample_extra_references)
    assert len(references) == 4
    assert [span.get("file_index") for span in references.spans] == [None, 0]
    assert references.spans[1]["input_token_start"] == 2
    assert datum.model_input.top_sampling_references is None
    batch.top_sampling_references = batch.sample_extra_references
    with pytest.raises(
        ValueError,
        match="top sampling and extra sampling SC candidates are mutually exclusive",
    ):
        objective.build_datums(batch, config)
    batch.top_sampling_references = []
    assert "sample_extra_token_ids" not in datum.loss_fn_inputs
    assert datum.loss_fn_inputs["response_mask"].data == [0, 0, 1, 1]
    assert datum.loss_fn_inputs["advantages"].data == [0.0, 0.0, 0.0, 0.0]
    assert datum.loss_fn_inputs["logprobs"].data == pytest.approx(
        [0.0, 0.0, -0.2, -0.4]
    )


def test_top_sampling_is_explicit_and_plain_reinforce_keeps_sc_disabled():
    plain = objective.loss_fn_config(objective.ReinforceConfig())
    top = objective.loss_fn_config(objective.ReinforceConfig(objective.TopSamplingSC()))
    assert plain["score_centering"] is False
    assert "score_centering_estimator" not in plain
    assert top["score_centering_estimator"] == "top_sampling"
    assert "sampler_kl_coeff" not in plain and "sampler_kl_coeff" not in top
    assert plain["weighting"] == top["weighting"] == "none"


def test_cli_sampler_kl_requires_score_centering():
    parser = argparse.ArgumentParser()
    objective.add_cli_args(parser)
    args = parser.parse_args(["--loss-fn", "reinforce", "--sampler-kl-coeff", ".01"])
    with pytest.raises(ValueError, match="sampler_kl_coeff requires score centering"):
        objective.from_cli_args(args)


@pytest.mark.parametrize("estimator", ["top_sampling", "extra_sampling"])
def test_reinforce_cli_selects_weighted_sc_with_kl_and_replay(estimator):
    parser = argparse.ArgumentParser()
    objective.add_cli_args(parser)
    config = objective.from_cli_args(
        parser.parse_args(
            [
                "--loss-fn",
                "reinforce",
                "--score-centering-estimator",
                estimator,
                "--weighting",
                "mis",
                "--sampler-kl-coeff",
                ".01",
                "--target-logprob-support",
                "sampling_support",
            ]
        )
    )
    request = objective.loss_fn_config(config)
    assert request["score_centering"] is True
    assert request["weighting"] == "mis" and request["sampler_kl_coeff"] == 0.01
    assert request["target_logprob_support"] == "sampling_support"
    assert (
        "anchor_logprob_source" not in request and "clip_low_threshold" not in request
    )
    sampling = objective.sampling_kwargs(config)
    assert sampling["sampling_mask"] == "parquet_v1"
    assert ("sample_extra_tokens" in sampling) == (
        estimator == "extra_sampling"
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--score-centering-estimator", "extra_sampling"],
        ["--score-centering-estimator", "top_sampling"],
        ["--weighting", "tis"],
        ["--sampler-kl-coeff", ".01"],
        ["--target-logprob-support", "sampling_support"],
    ],
)
def test_grpo_rejects_reinforce_options_instead_of_changing_its_loss(args):
    parser = argparse.ArgumentParser()
    objective.add_cli_args(parser)
    with pytest.raises(ValueError, match="require --loss-fn reinforce"):
        objective.from_cli_args(parser.parse_args(args))
    assert objective.from_cli_args(parser.parse_args([])) is None


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo", "dppo"])
def test_cli_offers_no_score_centering_alias_for_ratio_losses(loss_fn):
    parser = argparse.ArgumentParser()
    objective.add_cli_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--loss-fn", loss_fn])


def test_trainer_support_is_checked_with_the_submitted_loss_config():
    config = objective.ReinforceConfig(
        score_centering=objective.ExtraSamplingSC(4), sampler_kl_coeff=0.01
    )
    calls = []

    class Trainer:
        def validate_loss_capabilities(self, loss_fn, loss_fn_config):
            calls.append((loss_fn, loss_fn_config))
            raise ValueError(
                "Trainer does not advertise extra sampling score centering; upgrade the trainer"
            )

    with pytest.raises(ValueError, match="extra sampling score centering"):
        objective.check_trainer_support(Trainer(), config)
    assert calls == [("reinforce", objective.loss_fn_config(config))]
    with pytest.raises(ValueError, match="upgrade fireworks-ai"):
        objective.check_trainer_support(object(), config)


def test_recommended_extra_sampling_captures_only_sixteen_draws():
    config = objective.ReinforceConfig(objective.ExtraSamplingSC())
    assert objective.sampling_kwargs(config) == {
        "logprobs": True,
        "sample_extra_tokens": 16,
        "sample_extra_format": "parquet_v1",
    }
    assert (
        objective.loss_fn_config(config)["score_centering_estimator"]
        == "extra_sampling"
    )
    assert objective.ExtraSamplingSC(8).num_samples == 8


def test_unspecified_cli_estimator_uses_extra_sampling_when_sc_is_requested():
    parser = argparse.ArgumentParser()
    objective.add_cli_args(parser)
    config = objective.from_cli_args(
        parser.parse_args(["--loss-fn", "reinforce", "--sc-num-samples", "16"])
    )
    assert isinstance(config.score_centering, objective.ExtraSamplingSC)
    with pytest.raises(ValueError, match="requires top_sampling"):
        objective.from_cli_args(
            parser.parse_args(["--loss-fn", "reinforce", "--sc-top-sampling-size", "128"])
        )


@pytest.mark.parametrize("estimator", ["top_sampling", "extra_sampling"])
def test_cli_rejects_mixed_sc_sizes(estimator):
    parser = argparse.ArgumentParser()
    objective.add_cli_args(parser)
    with pytest.raises(ValueError, match="mutually exclusive"):
        objective.from_cli_args(
            parser.parse_args(
                [
                    "--loss-fn",
                    "reinforce",
                    "--score-centering-estimator",
                    estimator,
                    "--sc-num-samples",
                    "16",
                    "--sc-top-sampling-size",
                    "128",
                ]
            )
        )


@pytest.mark.parametrize("arguments", [
    ["--score-centering-estimator", "mc"],
    ["--score-centering-estimator", "topk"],
    ["--sc-head-size", "128"],
])
def test_cli_rejects_legacy_sc_names(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser()
    objective.add_cli_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--loss-fn", "reinforce", *arguments])


def test_preflight_requires_sdk_capability_validation():
    config = objective.ReinforceConfig(objective.ExtraSamplingSC())
    with pytest.raises(ValueError, match="needs an SDK with validate_loss_capabilities"):
        objective.check_trainer_support(object(), config)


def test_cli_rejects_two_explicit_estimators():
    parser = argparse.ArgumentParser()
    objective.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--loss-fn",
            "reinforce",
            "--score-centering-estimator",
            "top_sampling",
            "--score-centering-estimator",
            "extra_sampling",
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        objective.from_cli_args(args)
