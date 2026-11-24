from training.utils.rl.metrics import publish_tito_sidecar_metrics


def test_publishes_coverage_and_split_reasons() -> None:
    metrics: dict = {}
    publish_tito_sidecar_metrics(
        metrics,
        [
            {
                "tito/turn/completion_tokens_count": 4.0,
                "tito/turn/completion_tokens_sum": 4.0,
                "tito/turn/completion_tokens_min": 4.0,
                "tito/turn/completion_tokens_max": 4.0,
                "tito/trajectory/policy_turns_count": 1.0,
                "tito/trajectory/policy_turns_sum": 1.0,
                "tito/trajectory/policy_turns_min": 1.0,
                "tito/trajectory/policy_turns_max": 1.0,
                "tito/lineage/new_segment": 2.0,
                "tito/lineage/realign": 3.0,
                "tito/lineage/boundary_reason_history_rewrite": 1.0,
                "tito/lineage/history_rewrite_assistant_roundtrip": 1.0,
                "tito/lineage/boundary_reason_unbounded_or_ambiguous_drift": 1.0,
                "tito/coverage/sampled_completion_tokens": 100.0,
                "tito/coverage/trained_tokens": 80.0,
                "tito/coverage/lost_realign_overwritten": 20.0,
                "tito/coverage/lost_abandoned": 0.0,
                "tito/coverage/lost_fail_closed_masked": 4.0,
                "tito/coverage/lost_invisible": 1.0,
            }
        ],
    )

    assert metrics["tito/lineage/splits"] == 1.0
    assert metrics["tito/lineage/new_segment/history_rewrite"] == 1.0
    assert metrics["tito/lineage/new_segment/token_drift"] == 1.0
    assert metrics["tito/lineage/history_rewrite/assistant_roundtrip"] == 1.0
    assert metrics["tito/coverage/trained_fraction"] == 0.8
    assert metrics["tito/coverage/lost/realign_overwritten"] == 20.0
    assert metrics["tito/coverage/lost/abandoned"] == 0.0
    assert metrics["tito/coverage/lost/fail_closed_masked"] == 4.0
    assert metrics["tito/coverage/lost/invisible"] == 1.0


def test_coverage_fraction_omitted_without_samples() -> None:
    metrics: dict = {}
    publish_tito_sidecar_metrics(metrics, [{"tito/lineage/realign": 0.0}])
    assert "tito/coverage/trained_fraction" not in metrics


def test_tool_work_and_busy_seconds_merge_as_counters():
    result = {}
    common = {
        "tito/trajectory/policy_turns_count": 1,
        "tito/trajectory/policy_turns_sum": 0,
        "tito/trajectory/policy_turns_min": 0,
        "tito/trajectory/policy_turns_max": 0,
    }
    publish_tito_sidecar_metrics(
        result,
        [
            common
            | {
                "tito/tool/observed_rollouts": 1,
                "tito/tool/work_seconds": 3,
                "tito/tool/busy_seconds": 2,
            },
            common,
        ],
    )
    assert result["tito/tool/work_seconds_sum"] == 3
    assert result["tito/tool/busy_seconds_sum"] == 2
    assert result["tito/tool/rollout_coverage"] == 0.5


def test_failed_trial_timing_stays_bounded_between_publications():
    from training.utils.rl import trial_events

    trial_events.reset()
    try:
        for index in range(10_000):
            trial_events.record_failed_trial_wall(0.4 if index < 9_600 else 12.0)
        result = trial_events.drain()
        root = "tito/trial_failed/trial_wall_seconds"
        assert result[f"{root}_count"] == 10_000
        assert result[f"{root}_min"] == 0.4
        assert result[f"{root}_max"] == 12.0
        assert result[f"{root}_p95_upper_bound"] == 0.5
        assert result[f"{root}_p99_upper_bound"] == 12.0
        assert trial_events.drain() == {}
    finally:
        trial_events.reset()
