"""Independent replay-mask and score-centering sampling capture.

Choose one score-centering capture and optionally add the replay mask. Attach
the corresponding trainer-bound references to the datums before submitting the loss.
"""

from __future__ import annotations

from typing import Any, List

from fireworks.training import sdk
from fireworks.training.sdk import routing
from fireworks.training.sdk.routing import RoutingReferences

from training.utils.rl.router_replay import reference_gaps


def _attach_references(
    data: List[sdk.Datum], references_by_datum: List[RoutingReferences | None], field_name: str
) -> List[sdk.Datum]:
    if len(data) != len(references_by_datum):
        raise ValueError(f"{field_name} must provide one entry per datum")
    helper = getattr(routing, field_name.removesuffix("_references") + "_model_input_kwargs", None)
    if helper is None:
        raise ValueError(f"{field_name} requires trainer-shared replay support; upgrade fireworks-ai")
    result = []
    for datum, refs in zip(data, references_by_datum, strict=True):
        if field_name not in type(datum.model_input).model_fields:
            raise ValueError(f"{field_name} requires trainer-shared replay support; upgrade fireworks-ai")
        length = datum.model_input.length
        if refs is None:
            refs = reference_gaps(length)
        if not isinstance(refs, RoutingReferences) or len(refs) != length:
            raise ValueError(f"{field_name} must cover every model-input position")
        result.append(
            sdk.Datum(
                model_input=datum.model_input.model_copy(update=helper(refs, length)),
                loss_fn_inputs=datum.loss_fn_inputs,
            )
        )
    return result


def attach_top_sampling_references(
    data: List[sdk.Datum], top_sampling_references: List[RoutingReferences | None]
) -> List[sdk.Datum]:
    return _attach_references(data, top_sampling_references, "top_sampling_references")


def attach_sample_extra_references(
    data: List[sdk.Datum], sample_extra_references: List[RoutingReferences | None]
) -> List[sdk.Datum]:
    return _attach_references(data, sample_extra_references, "sample_extra_references")


def attach_sampling_mask_references(
    data: List[sdk.Datum], sampling_mask_references: List[RoutingReferences | None]
) -> List[sdk.Datum]:
    return _attach_references(data, sampling_mask_references, "sampling_mask_references")


def replay_mask_kwargs(*, top_p: float) -> dict[str, Any]:
    """Capture the complete ID-only replay support, without requesting an SC head."""
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1]; got {top_p!r}")
    return {"top_p": top_p, "sampling_mask": "parquet_v1"}


def top_sampling_kwargs(k: int) -> dict[str, Any]:
    """Capture only the probability head used by score centering."""
    if type(k) is not int or not 1 <= k <= 20000:
        raise ValueError("k must be an integer in [1, 20000]")
    return {"top_sampling_logprobs": k, "top_sampling_format": "parquet_v1"}


def extra_sampling_kwargs(k: int = 16) -> dict[str, Any]:
    """Capture independent stochastic draws for trainer-side score centering."""
    if type(k) is not int or not 1 <= k <= 64:
        raise ValueError("k must be an integer in [1, 64]")
    return {"sample_extra_tokens": k, "sample_extra_format": "parquet_v1"}


def sampling_support_loss_fn_config(loss_fn_config: dict[str, Any]) -> dict[str, Any]:
    """A built-in loss config that normalizes target logprobs over the recorded support."""
    return {**loss_fn_config, "target_logprob_support": "sampling_support"}
