import pytest

from training.tests._replay_refs import replay_refs, replay_rows
from training.utils.rl.router_replay import build_r3_routing_matrices


@pytest.mark.parametrize("source", [["a", "b"], ["p", "q", "a", "b"]])
def test_completion_rows_are_aligned_with_compact_prompt_gap(source):
    result = build_r3_routing_matrices(replay_refs(source), prompt_len=3, model_input_len=4)
    assert replay_rows(result) == ["", "", "a", "b"]
    assert result.spans[0] == {"input_token_start": 0, "count": 2}


def test_full_sequence_preserves_prompt_routes():
    refs = replay_refs(["p", "q", "a", "b"])
    assert build_r3_routing_matrices(refs, 3, 4, completion_only=False) is refs


@pytest.mark.parametrize("source", [None, [], ["inline"], replay_refs([]), replay_refs(["x", "y", "z"])])
def test_invalid_capture_fails_before_training(source):
    with pytest.raises(ValueError, match="references|length mismatch"):
        build_r3_routing_matrices(source, prompt_len=3, model_input_len=4)
