"""Sampling/trainer comparison, chunk reduction and logging contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fireworks.training import sdk

from training.utils import logging as logging_utils
from training.utils.metrics import forward_backward_metrics
from training.utils.rl.common import sample_train_metrics
from training.utils.rl.losses import PolicyBatch
from training.utils.rl.streaming import add_batch_diagnostics


def _batch(
    rows: list[list[float | None]],
    *,
    masks: list[list[float]] | None = None,
    prompt_lens: list[int] | None = None,
    **kwargs: Any,
) -> PolicyBatch:
    data = []
    for i, row in enumerate(rows):
        data.append(
            sdk.Datum(
                model_input=sdk.ModelInput.from_ints(list(range(len(row)))),
                loss_fn_inputs={
                    "target_tokens": sdk.TensorData(data=list(range(len(row))), dtype="int64", shape=[len(row)]),
                    "weights": sdk.TensorData(
                        data=masks[i] if masks is not None else [1.0] * len(row),
                        dtype="float32",
                        shape=[len(row)],
                    ),
                },
            )
        )
    return PolicyBatch(
        data=data,
        advantages=[0.0] * len(rows),
        prompt_lens=prompt_lens if prompt_lens is not None else [1] * len(rows),
        rollout_logprobs=rows,
        **kwargs,
    )


def _result(*rows: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        loss_fn_outputs=[{"logprobs": sdk.TensorData(data=row, dtype="float32", shape=[len(row)])} for row in rows],
        metrics={},
    )


def test_identical_logprobs_report_zero_even_with_zero_advantage() -> None:
    metrics = sample_train_metrics(_batch([[-1.0, -2.0]]), _result([-1.0, -2.0]))
    assert metrics == {
        "sample_train_max_abs_diff": 0.0,
        "sample_train_mismatch_frac": 0.0,
        "sample_train_mismatched_tokens": 0.0,
        "sample_train_compared_tokens": 2.0,
    }


def test_opposite_differences_do_not_cancel() -> None:
    metrics = sample_train_metrics(_batch([[-2.0, -2.0]]), _result([-1.0, -3.0]))
    assert metrics["sample_train_max_abs_diff"] == 1.0
    assert metrics["sample_train_mismatch_frac"] == 1.0


def test_counts_even_one_float32_step_without_a_tolerance() -> None:
    value = torch.nextafter(torch.tensor(-1.0), torch.tensor(0.0)).item()
    metrics = sample_train_metrics(_batch([[-1.0, -2.0]]), _result([value, -2.0]))
    assert 0 < metrics["sample_train_max_abs_diff"] < 1e-6
    assert metrics["sample_train_mismatch_frac"] == 0.5


def test_excludes_prompt_tool_and_suffix_tokens() -> None:
    batch = _batch(
        [[None, -2.0, None, -4.0, None]],
        masks=[[0.0, 1.0, 0.0, 1.0, 0.0]],
        prompt_lens=[2],
    )
    metrics = sample_train_metrics(batch, _result([float("nan"), -2.0, float("nan"), -3.5, float("nan")]))
    assert metrics["sample_train_compared_tokens"] == 2
    assert metrics["sample_train_mismatch_frac"] == 0.5
    assert metrics["sample_train_max_abs_diff"] == 0.5


def test_compares_original_sampling_not_raw_or_recomputed_anchor() -> None:
    batch = _batch(
        [[-1.0, -2.0]],
        anchor_logprobs=[[-5.0, -6.0]],
        raw_inference_logprobs=[[-7.0, -8.0]],
    )
    metrics = sample_train_metrics(batch, _result([-1.0, -2.0]))
    assert metrics["sample_train_max_abs_diff"] == 0.0


def test_unequal_rows_use_token_weighting_and_per_row_prompt_boundaries() -> None:
    batch = _batch([[-2.0], [None, -2.0, -2.0, -2.0]], prompt_lens=[1, 2])
    metrics = sample_train_metrics(batch, _result([-1.0], [100.0, -2.0, -2.0, -2.0]))
    assert metrics["sample_train_compared_tokens"] == 4
    assert metrics["sample_train_mismatch_frac"] == 0.25


@pytest.mark.parametrize("rows,masks", [([], None), ([[-1.0]], [[0.0]])])
def test_no_active_tokens_reports_no_comparison(rows: list, masks: list | None) -> None:
    assert sample_train_metrics(_batch(rows, masks=masks), _result(*rows)) == {}


@pytest.mark.parametrize("returned", [[], [[-1.0]], [[-1.0, -2.0, -3.0]]])
def test_rejects_misaligned_trainer_rows(returned: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        sample_train_metrics(_batch([[-1.0, -2.0]]), _result(*returned))


def test_rejects_missing_active_sampling_probability() -> None:
    with pytest.raises(ValueError, match="non-null"):
        sample_train_metrics(_batch([[-1.0, None]]), _result([-1.0, -2.0]))


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_active_logprobs(nonfinite: float) -> None:
    with pytest.raises(ValueError, match="finite active"):
        sample_train_metrics(_batch([[-1.0]]), _result([nonfinite]))


def test_unequal_chunks_preserve_global_max_and_token_weighted_fraction() -> None:
    short = _result([-0.5])
    short.metrics.update(sample_train_metrics(_batch([[-2.0]]), short))
    long = _result([-1.0, -1.0, -1.0])
    long.metrics.update(sample_train_metrics(_batch([[-1.0, -1.0, -1.0]]), long))

    metrics = forward_backward_metrics([short, long, SimpleNamespace(metrics={"loss:sum": 1.0})])

    assert metrics["train/sample_train_max_abs_diff"] == 1.5
    assert metrics["train/sample_train_mismatch_frac"] == 0.25
    assert metrics["train/sample_train_compared_tokens"] == 4
    assert metrics["train/sample_train_mismatched_tokens"] == 1


def test_async_diagnostics_compare_all_submissions_without_double_counting() -> None:
    submissions = [
        SimpleNamespace(batch=_batch([[-2.0]])),
        SimpleNamespace(batch=_batch([[-1.0, -1.0, -1.0]])),
    ]
    results = [_result([-0.5]), _result([-1.0, -1.0, -1.0])]
    for submission, result in zip(submissions, results, strict=True):
        result.metrics.update(sample_train_metrics(submission.batch, result))
    objective = SimpleNamespace(metrics=lambda *_args: {})

    add_batch_diagnostics(submissions, results, config=None, objective=objective)
    metrics = forward_backward_metrics(results)

    assert metrics["train/sample_train_max_abs_diff"] == 1.5
    assert metrics["train/sample_train_mismatch_frac"] == 0.25
    assert metrics["train/sample_train_compared_tokens"] == 4
    assert metrics["train/sample_train_mismatched_tokens"] == 1


def test_metrics_reach_wandb_payload_and_jsonl_without_network(tmp_path: Any, monkeypatch: Any) -> None:
    result = _result([-1.0, -2.5])
    result.metrics.update(sample_train_metrics(_batch([[-1.0, -2.0]]), result))
    metrics = forward_backward_metrics([result])
    metrics["train/step"] = 3
    sent = []
    monkeypatch.setattr(logging_utils, "wandb_log", lambda payload: sent.append(payload))
    path = tmp_path / "metrics.jsonl"

    logging_utils.log_metrics(metrics, step=3, metrics_file=str(path))

    assert json.loads(path.read_text()) == sent[0]
    assert sent[0]["train/sample_train_max_abs_diff"] == 0.5
    assert sent[0]["train/sample_train_mismatch_frac"] == 0.5
