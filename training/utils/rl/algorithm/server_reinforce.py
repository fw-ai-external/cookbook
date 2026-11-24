"""Built-in REINFORCE with optional score centering, TIS/MIS weighting and sampler KL.

Score centering replaces the sampler importance ratio, so the trainer defines it
only for ``loss_fn="reinforce"``. GRPO with score centering keeps the recipe's
group advantages and selects this objective instead of the PPO surrogate.
"""

import argparse
import math
from dataclasses import dataclass, replace
from typing import Any

from fireworks.training import sdk

from training.utils.client import GradAccNormalization
from training.utils.rl.common import policy_ratio_metrics
from training.utils.rl.losses import PolicyBatch, build_policy_datums
from training.utils.rl.sampling_support import (
    attach_sample_extra_references,
    attach_sampling_mask_references,
    attach_top_sampling_references,
    extra_sampling_kwargs,
    top_sampling_kwargs,
)

LOSS_FN = "reinforce"
NORMALIZATION = GradAccNormalization.NUM_LOSS_TOKENS


@dataclass(frozen=True)
class TopSamplingSC:
    """Center over the sampler probability head plus its modeled tail."""

    head_size: int = 128

    def __post_init__(self) -> None:
        if type(self.head_size) is not int or not 1 <= self.head_size <= 20000:
            raise ValueError("head_size must be an integer in 1..20000")


@dataclass(frozen=True)
class ExtraSamplingSC:
    """Center over independent draws with replacement; the recommended count is 16."""

    num_samples: int = 16

    def __post_init__(self) -> None:
        if type(self.num_samples) is not int or not 1 <= self.num_samples <= 64:
            raise ValueError("num_samples must be an integer in 1..64")


@dataclass(frozen=True)
class ReinforceConfig:
    score_centering: TopSamplingSC | ExtraSamplingSC | None = None
    weighting: str = "none"
    weight_low: float | None = None
    weight_high: float | None = None
    sampler_kl_coeff: float = 0.0
    target_logprob_support: str = "full_vocabulary"

    def __post_init__(self) -> None:
        if self.weighting not in ("none", "tis", "mis"):
            raise ValueError("weighting must be none, tis, or mis")
        if self.score_centering is not None and not isinstance(self.score_centering, (TopSamplingSC, ExtraSamplingSC)):
            raise ValueError("score_centering must be TopSamplingSC, ExtraSamplingSC, or None")
        if (
            isinstance(self.sampler_kl_coeff, bool)
            or not math.isfinite(self.sampler_kl_coeff)
            or self.sampler_kl_coeff < 0
        ):
            raise ValueError("sampler_kl_coeff must be finite and nonnegative")
        if self.sampler_kl_coeff and self.score_centering is None:
            raise ValueError("sampler_kl_coeff requires score centering")
        low, high = self.weight_bounds
        if (
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (low, high))
            or not 0 <= low <= high
            or high <= 0
        ):
            raise ValueError("weight bounds require finite 0 <= low <= high and high > 0")
        if self.target_logprob_support not in ("full_vocabulary", "sampling_support"):
            raise ValueError("unsupported target_logprob_support")

    @property
    def weight_bounds(self) -> tuple[float, float]:
        """Server defaults: ``[0.5, 5]`` for MIS, ``[0, 2]`` for TIS."""
        low = self.weight_low if self.weight_low is not None else (0.5 if self.weighting == "mis" else 0.0)
        high = self.weight_high if self.weight_high is not None else (5.0 if self.weighting == "mis" else 2.0)
        return low, high


def loss_fn_config(config: ReinforceConfig) -> dict[str, Any]:
    result = {
        "score_centering": config.score_centering is not None,
        "weighting": config.weighting,
        "target_logprob_support": config.target_logprob_support,
    }
    if isinstance(config.score_centering, ExtraSamplingSC):
        result["score_centering_estimator"] = "extra_sampling"
    elif isinstance(config.score_centering, TopSamplingSC):
        result["score_centering_estimator"] = "top_sampling"
    if config.sampler_kl_coeff:
        result["sampler_kl_coeff"] = config.sampler_kl_coeff
    if config.weight_low is not None:
        result["weight_low"] = config.weight_low
    if config.weight_high is not None:
        result["weight_high"] = config.weight_high
    return result


def check_trainer_support(training_client: Any, config: ReinforceConfig) -> None:
    """Fail at startup, before any rollout, when the trainer cannot run this objective."""
    validate = getattr(training_client, "validate_loss_capabilities", None)
    if validate is None:
        raise ValueError("Built-in REINFORCE needs an SDK with validate_loss_capabilities; upgrade fireworks-ai")
    validate(LOSS_FN, loss_fn_config(config))


def sampling_kwargs(config: ReinforceConfig) -> dict[str, Any]:
    result = {"logprobs": True}
    if isinstance(config.score_centering, ExtraSamplingSC):
        result.update(extra_sampling_kwargs(config.score_centering.num_samples))
    elif isinstance(config.score_centering, TopSamplingSC):
        result.update(top_sampling_kwargs(config.score_centering.head_size))
    if config.target_logprob_support == "sampling_support":
        result["sampling_mask"] = "parquet_v1"
    return result


def build_datums(batch: PolicyBatch, config: ReinforceConfig) -> list[sdk.Datum]:
    top_sampling = any(ref is not None for ref in batch.top_sampling_references)
    extra_sampling = any(ref is not None for ref in batch.sample_extra_references)
    if top_sampling and extra_sampling:
        raise ValueError("top sampling and extra sampling SC candidates are mutually exclusive")
    if top_sampling and not isinstance(config.score_centering, TopSamplingSC):
        raise ValueError("top_sampling_references require TopSamplingSC")
    if extra_sampling and not isinstance(config.score_centering, ExtraSamplingSC):
        raise ValueError("sample_extra_references require ExtraSamplingSC")
    # TIS/MIS weights and sampler KL use the sampler logprobs, never a recomputed anchor.
    result = []
    for datum, logq in zip(build_policy_datums(batch), batch.rollout_logprobs, strict=True):
        inputs = dict(datum.loss_fn_inputs)
        inputs["logprobs"] = sdk.TensorData(data=list(logq), dtype="float32", shape=[len(logq)])
        result.append(sdk.Datum(model_input=datum.model_input, loss_fn_inputs=inputs))
    if isinstance(config.score_centering, TopSamplingSC):
        result = attach_top_sampling_references(result, batch.top_sampling_references)
    elif isinstance(config.score_centering, ExtraSamplingSC):
        result = attach_sample_extra_references(result, batch.sample_extra_references)
    if config.target_logprob_support == "sampling_support":
        result = attach_sampling_mask_references(result, batch.sampling_mask_references)
    return result


def metrics(batch: PolicyBatch, result: Any, config: ReinforceConfig) -> dict[str, float]:
    low, high = config.weight_bounds
    outside = None if config.weighting == "none" else lambda ratio, *_: (ratio < low) | (ratio > high)
    result_metrics = policy_ratio_metrics(
        replace(batch, anchor_logprobs=batch.rollout_logprobs),
        result,
        loss_name="reinforce",
        prefix="reinforce",
        outside_trust_region=outside,
    )
    if isinstance(config.score_centering, ExtraSamplingSC) and result_metrics["active_tokens"] > 0:
        result_metrics["score_centering_mc"] = 1.0
    return result_metrics


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Shared opt-in objective arguments for single- and multi-turn recipes."""
    parser.add_argument(
        "--loss-fn",
        choices=("grpo", "reinforce"),
        default="grpo",
        help="grpo: PPO surrogate on group advantages; reinforce: built-in REINFORCE on the same advantages",
    )
    parser.add_argument(
        "--score-centering-estimator",
        choices=("none", "top_sampling", "extra_sampling"),
        action="append",
        default=None,
        help="SC mode; extra_sampling is recommended",
    )
    parser.add_argument(
        "--sc-num-samples",
        type=int,
        default=None,
        help="Extra sampling draws per emitted token, 1..64; default 16",
    )
    parser.add_argument(
        "--sc-top-sampling-size",
        type=int,
        default=None,
        help="Probability head size; requires explicit top_sampling",
    )
    parser.add_argument("--weighting", choices=("none", "tis", "mis"), default="none")
    parser.add_argument("--weight-low", type=float, default=None)
    parser.add_argument("--weight-high", type=float, default=None)
    parser.add_argument(
        "--sampler-kl-coeff",
        type=float,
        default=0.0,
        help="Sampler KL coefficient for score-centered REINFORCE",
    )
    parser.add_argument(
        "--target-logprob-support",
        choices=("full_vocabulary", "sampling_support"),
        default="full_vocabulary",
    )


def from_cli_args(args: argparse.Namespace) -> ReinforceConfig | None:
    estimators = args.score_centering_estimator or []
    if len(set(estimators)) > 1:
        raise ValueError("SC estimators are mutually exclusive; select one --score-centering-estimator")
    estimator = estimators[0] if estimators else None
    samples, head = args.sc_num_samples, args.sc_top_sampling_size
    if estimator is None:
        estimator = "extra_sampling" if samples is not None or head is not None else "none"
    if estimator not in ("none", "top_sampling", "extra_sampling"):
        raise ValueError("score_centering_estimator must be none, top_sampling, or extra_sampling")
    weighting = args.weighting
    beta = args.sampler_kl_coeff
    support = args.target_logprob_support
    low, high = args.weight_low, args.weight_high
    if args.loss_fn == "grpo":
        reinforce_options = (
            estimator != "none",
            samples is not None,
            head is not None,
            weighting != "none",
            bool(beta),
            low is not None,
            high is not None,
            support != "full_vocabulary",
        )
        if any(reinforce_options):
            raise ValueError(
                "Score centering, weighting, sampler KL and replay require --loss-fn reinforce; "
                "group advantages are unchanged"
            )
        return None
    if samples is not None and estimator != "extra_sampling":
        raise ValueError(
            "--sc-num-samples requires extra_sampling; top sampling and extra sampling are mutually exclusive"
        )
    if head is not None and estimator != "top_sampling":
        raise ValueError(
            "--sc-top-sampling-size requires top_sampling; top sampling and extra sampling are mutually exclusive"
        )
    if estimator == "extra_sampling":
        score_centering = ExtraSamplingSC(samples if samples is not None else 16)
    elif estimator == "top_sampling":
        score_centering = TopSamplingSC(head if head is not None else 128)
    else:
        score_centering = None
    return ReinforceConfig(
        score_centering=score_centering,
        weighting=weighting,
        weight_low=low,
        weight_high=high,
        sampler_kl_coeff=beta,
        target_logprob_support=support,
    )
