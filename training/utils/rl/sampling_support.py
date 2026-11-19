"""Independent replay-mask and score-centering sampling capture.

Choose top_sampling_kwargs or extra_sampling_kwargs, optionally add
replay_mask_kwargs, then attach the selected channels' references to the datums. Loss selection remains with the
caller. These SC helpers explicitly select trainer-bound shared storage; plain
inference requests retain their existing transport defaults.
"""

from __future__ import annotations

from typing import Any, List

from fireworks.training import sdk
from fireworks.training.sdk.routing import RoutingReferences


def _attach_references(
    data: List[sdk.Datum],
    references: List[RoutingReferences | None],
    field: str,
) -> List[sdk.Datum]:
    """Return datums whose model inputs carry ``field``.

    A datum without references gets an all-gap reference; the trainer rejects
    it if a trained response position lacks a recorded row.
    """
    if len(data) != len(references):
        raise ValueError(f"{field} must provide one entry per datum")
    # Capture is opt-in; importing the recipe must also work with older SDKs.
    try:
        from fireworks.training.sdk import routing
    except ImportError as exc:
        raise ValueError(
            "Sampling support requires an SDK with Parquet sampling-reference "
            "transport; upgrade fireworks-ai before enabling sampling_support"
        ) from exc
    model_input_kwargs = getattr(
        routing, field.removesuffix("_references") + "_model_input_kwargs", None
    )
    if model_input_kwargs is None:
        raise ValueError(
            f"{field} requires a newer SDK; upgrade fireworks-ai before enabling it"
        )

    result: List[sdk.Datum] = []
    for datum, value in zip(data, references, strict=True):
        length = datum.model_input.length
        if value is None:
            value = RoutingReferences(
                length,
                (),
                ({"input_token_start": 0, "count": length},) if length else (),
            )
        result.append(
            sdk.Datum(
                model_input=datum.model_input.model_copy(
                    update=model_input_kwargs(value, length)
                ),
                loss_fn_inputs=datum.loss_fn_inputs,
            )
        )
    return result


def attach_top_sampling_references(
    data: List[sdk.Datum],
    top_sampling_references: List[RoutingReferences | None],
) -> List[sdk.Datum]:
    """Return datums whose model inputs carry ``top_sampling_references``."""
    return _attach_references(data, top_sampling_references, "top_sampling_references")


def attach_sampling_mask_references(
    data: List[sdk.Datum],
    sampling_mask_references: List[RoutingReferences | None],
) -> List[sdk.Datum]:
    """Return datums whose model inputs carry ``sampling_mask_references``."""
    return _attach_references(
        data, sampling_mask_references, "sampling_mask_references"
    )


def attach_sample_extra_references(
    data: List[sdk.Datum],
    sample_extra_references: List[RoutingReferences | None],
) -> List[sdk.Datum]:
    """Return datums whose model inputs carry ``sample_extra_references``."""
    return _attach_references(data, sample_extra_references, "sample_extra_references")


def replay_mask_kwargs(*, top_p: float) -> dict[str, Any]:
    """Capture the complete ID-only replay support, without requesting an SC head."""
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1]; got {top_p!r}")
    return {"top_p": top_p, "sampling_mask": "parquet_v1"}


def extra_sampling_kwargs(k: int = 16) -> dict[str, Any]:
    """Capture ``k`` draws with replacement per token; no probability head is needed."""
    if type(k) is not int or not 1 <= k <= 64:
        raise ValueError("k must be an integer in [1, 64]")
    return {"sample_extra_tokens": k, "sample_extra_format": "parquet_v1"}


def top_sampling_kwargs(k: int) -> dict[str, Any]:
    """Capture only the probability head used by score centering."""
    if type(k) is not int or not 1 <= k <= 20000:
        raise ValueError("k must be an integer in [1, 20000]")
    return {"top_sampling_logprobs": k, "top_sampling_format": "parquet_v1"}


def sampling_support_loss_fn_config(loss_fn_config: dict[str, Any]) -> dict[str, Any]:
    """A built-in loss config that normalizes target logprobs over the recorded support."""
    return {**loss_fn_config, "target_logprob_support": "sampling_support"}
