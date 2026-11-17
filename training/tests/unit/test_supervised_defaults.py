"""Customer entrypoints preserve recipe save capabilities and explicit overrides."""

import importlib
import logging

import pytest

from training.utils.checkpoints import log_checkpoint_plan, resolve_checkpoint_cadence


@pytest.mark.parametrize("method", ["sft", "dpo"])
def test_default_and_explicit_checkpoint_plans(method):
    example = importlib.import_module(
        f"training.examples.{method}.train_{method}_serverless"
    )
    cfg = example.build_config(example.parse_args([]))
    cadence = resolve_checkpoint_cadence(
        dcp_save_interval=cfg.dcp_save_interval,
        checkpoint_interval_minutes=cfg.checkpoint_interval_minutes,
        lora_rank=cfg.lora_rank,
    )
    assert cadence.schedule_on and cadence.checkpoint_interval_minutes == 30
    cfg = example.build_config(
        example.parse_args(
            [
                "--dcp-save-interval",
                "5",
                "--sampler-save-interval",
                "5",
                "--checkpoint-interval-minutes",
                "0",
            ]
        )
    )
    cadence = resolve_checkpoint_cadence(
        dcp_save_interval=cfg.dcp_save_interval,
        checkpoint_interval_minutes=cfg.checkpoint_interval_minutes,
        lora_rank=cfg.lora_rank,
    )
    assert not cadence.schedule_on
    assert cadence.dcp_save_interval == cfg.sampler_save_interval == 5


def test_recovery_only_plan_warns_without_claiming_auto_promotion(caplog):
    cadence = resolve_checkpoint_cadence(
        dcp_save_interval=5, checkpoint_interval_minutes=0, lora_rank=8
    )
    with caplog.at_level(logging.INFO):
        log_checkpoint_plan(
            cadence,
            sampler_save_interval=0,
            save_final_checkpoint=True,
            output_model_id=None,
        )
    assert "resumable only" in caplog.text
    assert "promotion: disabled" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO):
        log_checkpoint_plan(
            cadence,
            sampler_save_interval=5,
            save_final_checkpoint=True,
            output_model_id="chosen-model",
        )
    assert "resumable only" not in caplog.text
    assert "promotion: chosen-model" in caplog.text
