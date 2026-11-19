"""Align compact trainer-shared router replay references to model-input positions."""

from __future__ import annotations

from fireworks.training.sdk.routing import RoutingReferences, concat_routing


def require_references(value: RoutingReferences | None) -> RoutingReferences | None:
    if value is not None and not isinstance(value, RoutingReferences):
        raise ValueError("Replay requires trainer-shared routing references")
    return value


def reference_gaps(length: int) -> RoutingReferences:
    """A single contiguous gap for prompt, tool or masked positions."""
    return RoutingReferences(
        length, (), ({"input_token_start": 0, "count": length},) if length else ()
    )


def validate_r3_routing_matrices(
    routing_matrices: RoutingReferences | None,
    prompt_len: int,
    model_input_len: int,
) -> None:
    """Require a full-sequence or completion-only routing capture."""
    if not isinstance(routing_matrices, RoutingReferences) or not len(routing_matrices):
        raise ValueError("R3 requires nonempty trainer-shared routing references")
    expected = max(0, model_input_len - max(0, prompt_len - 1))
    if len(routing_matrices) not in {model_input_len, expected}:
        raise ValueError(
            f"R3 routing_matrices length mismatch: got {len(routing_matrices)}, expected "
            f"{model_input_len} full-sequence or {expected} completion-only values"
        )


def build_r3_routing_matrices(
    routing_matrices: RoutingReferences,
    prompt_len: int,
    model_input_len: int,
    completion_only: bool = True,
) -> RoutingReferences:
    """Align capture rows and represent prompt positions as a compact gap."""
    validate_r3_routing_matrices(routing_matrices, prompt_len, model_input_len)
    prefix_len = max(0, prompt_len - 1)
    if len(routing_matrices) != model_input_len:
        return concat_routing(reference_gaps(prefix_len), routing_matrices)
    if completion_only:
        return concat_routing(reference_gaps(prefix_len), routing_matrices[prefix_len:])
    return routing_matrices
