"""Tests for the managed SFT/DPO/ORPO -> Training API migration tool."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from fireworks.training.sdk.training_spec import (
    ConstantSchedule,
    CosineSchedule,
    LinearSchedule,
)

import training.recipes.dpo_loop as dpo_loop
import training.recipes.orpo_loop as orpo_loop
import training.recipes.sft_loop as sft_loop
from training.examples.tools import migrate_managed_job as mmj
from training.utils.checkpoints import DEFAULT_DCP_SAVE_INTERVAL

ACCOUNT = "acme"
SFT_JOB = f"accounts/{ACCOUNT}/supervisedFineTuningJobs/sft-1"
DPO_JOB = f"accounts/{ACCOUNT}/dpoJobs/dpo-1"

POOL_MODEL = {
    "name": "accounts/fireworks/models/kimi-k3",
    "kind": "HF_BASE_MODEL",
    "huggingFaceUrl": "https://huggingface.co/moonshotai/Kimi-K3",
    "trainingContextLength": 65536,
    "supportsServerlessTraining": True,
}
SHAPE_MODEL = {
    "name": "accounts/fireworks/models/qwen3p5-9b",
    "kind": "HF_BASE_MODEL",
    "huggingFaceUrl": "https://huggingface.co/Qwen/Qwen3.5-9B",
    "trainingContextLength": 131072,
}
ADAPTER = {
    "name": f"accounts/{ACCOUNT}/models/old-adapter",
    "kind": "HF_PEFT_ADDON",
    "peftDetails": {
        "baseModel": "accounts/fireworks/models/kimi-k3",
        "r": 16,
        "targetModules": ["q_proj", "v_proj", "gate"],
    },
}

# Managed V2 LoRA SFT on a serverless-pool model, as the REST API returns it.
V2_LORA_SFT = {
    "name": SFT_JOB,
    "dataset": f"accounts/{ACCOUNT}/datasets/train",
    "baseModel": "accounts/fireworks/models/kimi-k3",
    "outputModel": f"accounts/{ACCOUNT}/models/old-sft",
    "epochs": 2,
    "learningRate": 0.0002,
    "maxContextLength": 4096,
    "loraRank": 8,
    "loraAlpha": 16,
    "batchSizeSamples": 8,
    "learningRateWarmupSteps": 3,
    "lrScheduler": {"cosine": {"minLrRatio": 0.1}},
    "optimizerWeightDecay": 0.05,
    "evalAutoCarveout": True,
    "generationEval": True,
    "thinkingTraceHistoryMode": "THINKING_TRACE_HISTORY_MODE_PRESERVED",
    "samplerCheckpointSaveIntervalSteps": 5,
    "region": "US_VIRGINIA_1",
    "encryptionState": "ENCRYPTION_STATE_PLAINTEXT",
    "state": "JOB_STATE_COMPLETED",
}

FULL_PARAM_SFT = {
    "name": SFT_JOB,
    "dataset": f"accounts/{ACCOUNT}/datasets/train",
    "baseModel": "accounts/fireworks/models/qwen3p5-9b",
    "outputModel": f"accounts/{ACCOUNT}/models/old-full",
    "learningRate": 0.00001,
    "batchSizeSamples": 16,
    "lrScheduler": {"linear": {"minLrRatio": 0.0, "decayRatio": 0.5}},
}

WARM_START_SFT = {
    "name": SFT_JOB,
    "dataset": f"accounts/{ACCOUNT}/datasets/train",
    "warmStartFrom": f"accounts/{ACCOUNT}/models/old-adapter",
    "outputModel": f"accounts/{ACCOUNT}/models/old-warm",
    "loraRank": 8,
    "useReservation": True,
    "reservationTarget": f"accounts/{ACCOUNT}/reservations/r1",
}

DPO_LORA = {
    "name": DPO_JOB,
    "dataset": f"accounts/{ACCOUNT}/datasets/prefs",
    "trainingConfig": {
        "baseModel": "accounts/fireworks/models/kimi-k3",
        "outputModel": f"accounts/{ACCOUNT}/models/old-dpo",
        "loraRank": 8,
        "maxContextLength": 8192,
        "batchSizeSamples": 4,
        "learningRate": 0.00001,
    },
    "lossConfig": {"method": "DPO", "dpo": {"beta": 0.2, "refCacheConcurrency": 8, "refCacheBatchSize": 2}},
    "evaluationDataset": f"accounts/{ACCOUNT}/datasets/held-out",
}

DPO_FULL = {
    "name": DPO_JOB,
    "dataset": f"accounts/{ACCOUNT}/datasets/prefs",
    "trainingConfig": {
        "baseModel": "accounts/fireworks/models/qwen3p5-9b",
        "outputModel": f"accounts/{ACCOUNT}/models/old-dpo-full",
        "epochs": 2,
    },
    "lossConfig": {"method": "DPO"},
}

ORPO = {
    "name": DPO_JOB,
    "dataset": f"accounts/{ACCOUNT}/datasets/prefs",
    "trainingConfig": {
        "baseModel": "accounts/fireworks/models/kimi-k3",
        "outputModel": f"accounts/{ACCOUNT}/models/old-orpo",
        "loraRank": 16,
        "loraTargetModules": ["q_proj"],
        "loraDropout": 0.1,
        "learningRateWarmupSteps": 4,
    },
    "lossConfig": {"method": "ORPO", "orpo": {"lambda": 0.5}},
    "evalAutoCarveout": True,
}

LEGACY_V1_SFT = {
    "name": SFT_JOB,
    "dataset": f"accounts/{ACCOUNT}/datasets/train",
    "baseModel": "accounts/fireworks/models/qwen3p5-9b",
    "outputModel": f"accounts/{ACCOUNT}/models/old-v1",
    "loraRank": 16,
    "batchSize": 65536,
    "maxContextLength": 8192,
    "earlyStop": True,
}


@pytest.fixture
def train_file(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("{}\n")
    return path


def _inputs(train_file, tmp_path, **overrides):
    values = {
        "dataset_path": str(train_file),
        "output_model_id": "new-model",
        "log_path": str(tmp_path / "logs"),
    }
    values.update(overrides)
    return mmj.MigrationInputs(**values)


def _plan(job, train_file, tmp_path, *, base_model, adapter=None, **overrides):
    ref = mmj.parse_job_name(job["name"])
    return mmj.build_plan(ref, job, _inputs(train_file, tmp_path, **overrides), base_model=base_model, adapter=adapter)


def test_v2_lora_sft_maps_to_serverless_sft_config(train_file, tmp_path):
    plan = _plan(V2_LORA_SFT, train_file, tmp_path, base_model=POOL_MODEL)

    assert plan.mode is mmj.Mode.SERVERLESS
    cfg = plan.config
    assert isinstance(cfg, sft_loop.Config)
    assert cfg.serverless is True
    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.tokenizer_model == "moonshotai/Kimi-K3"
    assert cfg.dataset == str(train_file)
    assert cfg.dataset_loading_mode == "stage"
    assert (cfg.epochs, cfg.learning_rate, cfg.batch_size) == (2, 0.0002, 8)
    assert (cfg.lora_rank, cfg.lora_alpha) == (8, 16)
    assert cfg.weight_decay == 0.05
    assert cfg.max_seq_len == 4096
    assert cfg.lr_scheduler == CosineSchedule(warmup_steps=3, min_lr_ratio=0.1)
    assert cfg.thinking_trace_history_mode == "preserved"
    assert cfg.eval_auto_carveout is True and cfg.evaluation_dataset == ""
    assert cfg.generation_eval is True
    assert cfg.sampler_save_interval == 5
    assert cfg.output_model_id == "new-model"
    assert cfg.log_path == str(tmp_path / "logs")
    # The recipes' step cadence, set explicitly; managed jobs saved every 20 steps.
    assert cfg.dcp_save_interval == mmj.MANAGED_DCP_SAVE_INTERVAL == DEFAULT_DCP_SAVE_INTERVAL == 50
    assert cfg.checkpoint_interval_minutes == 30.0
    # Explicit trainer intent; the region is never copied.
    assert cfg.trainer.use_reservation is False
    assert cfg.trainer.region is None
    # The dedicated fallback is the same run without the serverless-only bits.
    fallback = plan.fallback_config
    assert isinstance(fallback, sft_loop.Config)
    assert fallback.serverless is False and fallback.generation_eval is False
    assert fallback.lr_scheduler == cfg.lr_scheduler


def test_full_parameter_sft_runs_dedicated_without_lora_schedule(train_file, tmp_path):
    plan = _plan(FULL_PARAM_SFT, train_file, tmp_path, base_model=SHAPE_MODEL)

    assert plan.mode is mmj.Mode.DEDICATED
    assert any("full-parameter" in reason for reason in plan.mode_reasons)
    cfg = plan.config
    assert isinstance(cfg, sft_loop.Config)
    assert cfg.serverless is False
    assert cfg.lora_rank == 0
    assert cfg.learning_rate == 0.00001 and cfg.batch_size == 16
    # Unset epochs keep the recipe default, as managed V2 did.
    assert cfg.epochs == sft_loop.Config(log_path="x").epochs
    assert cfg.lr_scheduler == LinearSchedule(min_lr_ratio=0.0, decay_ratio=0.5)
    # No max_context_length on the job: use the model's training context length.
    assert cfg.max_seq_len == 131072
    assert cfg.tokenizer_model == "Qwen/Qwen3.5-9B"
    assert cfg.dcp_save_interval == 50
    assert cfg.checkpoint_interval_minutes == 0.0
    assert plan.fallback_config is None


def test_warm_start_sft_takes_base_model_from_adapter_and_runs_dedicated(train_file, tmp_path):
    plan = _plan(WARM_START_SFT, train_file, tmp_path, base_model=POOL_MODEL, adapter=ADAPTER)

    assert plan.mode is mmj.Mode.DEDICATED
    cfg = plan.config
    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.warm_start_from_adapter == f"accounts/{ACCOUNT}/models/old-adapter"
    assert cfg.lora_rank == 8
    assert cfg.trainer.use_reservation is True
    assert cfg.trainer.reservation_target == f"accounts/{ACCOUNT}/reservations/r1"
    warnings = "\n".join(plan.warnings)
    assert "differs from the warm-start adapter's rank 16" in warnings
    assert "gate" in warnings and "Adapter key mismatch" in warnings


def test_resuming_a_warm_start_run_drops_the_adapter(train_file, tmp_path):
    plan = _plan(
        WARM_START_SFT, train_file, tmp_path, base_model=POOL_MODEL, adapter=ADAPTER, init_from_checkpoint="job-1:step-20"
    )

    assert plan.config.init_from_checkpoint == "job-1:step-20"
    assert plan.config.warm_start_from_adapter is None


def test_rejected_api_key_stops_with_clear_message(monkeypatch):
    request = httpx.Request("GET", "https://api.fireworks.ai/v1/accounts")
    response = httpx.Response(401, request=request)

    class RejectedKeyClient:
        def __init__(self, **kwargs):
            pass

        @property
        def account_id(self):
            raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

        def close(self):
            pass

    monkeypatch.setattr(mmj, "FireworksClient", RejectedKeyClient)
    with pytest.raises(mmj.MigrationError, match="rejected \\(HTTP 401\\)"):
        mmj.FireworksReader("fw_test").key_account_id()


def test_dpo_lora_maps_to_serverless_dpo_config(train_file, tmp_path):
    plan = _plan(DPO_LORA, train_file, tmp_path, base_model=POOL_MODEL)

    assert plan.mode is mmj.Mode.SERVERLESS
    cfg = plan.config
    assert isinstance(cfg, dpo_loop.Config)
    assert cfg.serverless is True
    assert (cfg.beta, cfg.ref_cache_concurrency, cfg.ref_cache_batch_size) == (0.2, 8, 2)
    assert (cfg.lora_rank, cfg.batch_size, cfg.learning_rate, cfg.max_seq_len) == (8, 4, 0.00001, 8192)
    assert cfg.epochs == dpo_loop.Config(log_path="x").epochs
    assert cfg.checkpoint_interval_minutes == 30.0 and cfg.dcp_save_interval == 50
    assert cfg.runner.reference_checkpoint_file == str(tmp_path / "logs" / "reference_checkpoint.json")
    # The held-out dataset is a resource name: warn, don't guess.
    assert cfg.evaluation_dataset == ""
    assert any("--eval-dataset-path" in w for w in plan.warnings)
    assert plan.fallback_config.serverless is False


def test_dpo_eval_dataset_path_is_used_when_given(train_file, tmp_path):
    held_out = tmp_path / "held_out.jsonl"
    held_out.write_text("{}\n")
    plan = _plan(DPO_LORA, train_file, tmp_path, base_model=POOL_MODEL, eval_dataset_path=str(held_out))

    assert plan.config.evaluation_dataset == str(held_out)
    assert not any("--eval-dataset-path" in w for w in plan.warnings)


def test_dpo_full_parameter_runs_dedicated(train_file, tmp_path):
    plan = _plan(DPO_FULL, train_file, tmp_path, base_model=SHAPE_MODEL)

    assert plan.mode is mmj.Mode.DEDICATED
    cfg = plan.config
    assert isinstance(cfg, dpo_loop.Config)
    assert cfg.serverless is False and cfg.lora_rank == 0
    assert cfg.epochs == 2
    assert cfg.beta == dpo_loop.Config(log_path="x").beta
    assert cfg.checkpoint_interval_minutes == 0.0
    assert cfg.runner.reference_checkpoint_file is None


def test_orpo_maps_to_dedicated_orpo_config(train_file, tmp_path):
    plan = _plan(ORPO, train_file, tmp_path, base_model=POOL_MODEL)

    assert plan.mode is mmj.Mode.DEDICATED
    assert any("ORPO" in reason for reason in plan.mode_reasons)
    cfg = plan.config
    assert isinstance(cfg, orpo_loop.Config)
    assert cfg.orpo_lambda == 0.5
    assert cfg.lora_rank == 16
    assert cfg.lr_scheduler == ConstantSchedule(warmup_steps=4)
    assert cfg.dcp_save_interval == 50
    assert not hasattr(cfg, "checkpoint_interval_minutes")
    warnings = "\n".join(plan.warnings)
    assert "lora_target_modules" in warnings and "lora_dropout" in warnings
    assert "ORPO has no held-out evaluation" in warnings


def test_legacy_v1_batch_size_warns_and_keeps_recipe_batch(train_file, tmp_path):
    plan = _plan(LEGACY_V1_SFT, train_file, tmp_path, base_model=SHAPE_MODEL)

    assert plan.config.batch_size == sft_loop.Config(log_path="x").batch_size
    warnings = "\n".join(plan.warnings)
    assert "legacy (V1) token batch size 65536" in warnings
    assert "early stopping" in warnings


UNUSED_EXTERNAL_CONFIGS = {
    "awsS3Config": {"credentialsSecret": "", "iamRoleArn": ""},
    "azureBlobStorageConfig": {"credentialsSecret": "", "managedIdentityClientId": "", "tenantId": ""},
}


def test_unused_external_dataset_configs_do_not_warn(train_file, tmp_path):
    # The REST API returns both configs with empty strings on every job.
    plan = _plan({**V2_LORA_SFT, **UNUSED_EXTERNAL_CONFIGS}, train_file, tmp_path, base_model=POOL_MODEL)
    assert not any("S3" in w or "Azure" in w for w in plan.warnings)


def test_s3_dataset_config_warns(train_file, tmp_path):
    job = {**V2_LORA_SFT, **UNUSED_EXTERNAL_CONFIGS, "awsS3Config": {"iamRoleArn": "arn:aws:iam::1:role/r"}}
    plan = _plan(job, train_file, tmp_path, base_model=POOL_MODEL)
    assert any("from S3" in w for w in plan.warnings)


def test_cmek_job_stops(train_file, tmp_path):
    job = {**V2_LORA_SFT, "encryptionState": "ENCRYPTION_STATE_CMEK"}
    with pytest.raises(mmj.MigrationError, match="CMEK"):
        _plan(job, train_file, tmp_path, base_model=POOL_MODEL)


def test_serverless_mode_rejects_full_parameter(train_file, tmp_path):
    with pytest.raises(mmj.MigrationError, match="can't run on serverless"):
        _plan(FULL_PARAM_SFT, train_file, tmp_path, base_model=SHAPE_MODEL, mode="serverless")


def test_dedicated_mode_overrides_serverless_eligibility(train_file, tmp_path):
    plan = _plan(V2_LORA_SFT, train_file, tmp_path, base_model=POOL_MODEL, mode="dedicated")

    assert plan.mode is mmj.Mode.DEDICATED
    assert plan.config.serverless is False
    assert plan.fallback_config is None
    assert any("generation_eval" in w for w in plan.warnings)


def test_new_output_model_must_differ_from_managed_output(train_file, tmp_path):
    with pytest.raises(mmj.MigrationError, match="already exists"):
        _plan(V2_LORA_SFT, train_file, tmp_path, base_model=POOL_MODEL, output_model_id="old-sft")


def test_dataset_resource_names_are_rejected(train_file, tmp_path):
    with pytest.raises(mmj.MigrationError, match="resource names"):
        _plan(
            V2_LORA_SFT,
            train_file,
            tmp_path,
            base_model=POOL_MODEL,
            dataset_path=f"accounts/{ACCOUNT}/datasets/train",
        )


def test_gcs_dataset_streams(train_file, tmp_path):
    plan = _plan(V2_LORA_SFT, train_file, tmp_path, base_model=POOL_MODEL, dataset_path="gs://bucket/train.jsonl")
    assert plan.config.dataset_loading_mode == "stream"


def test_high_rank_and_wandb_warnings(train_file, tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    job = {**V2_LORA_SFT, "loraRank": 128, "wandbConfig": {"enabled": True, "entity": "team", "project": "ft"}}
    plan = _plan(job, train_file, tmp_path, base_model=POOL_MODEL)

    assert plan.config.wandb.entity == "team" and plan.config.wandb.project == "ft"
    warnings = "\n".join(plan.warnings)
    assert "above 64" in warnings
    assert "WANDB_API_KEY" in warnings


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://huggingface.co/Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
        ("https://huggingface.co/Qwen/Qwen3-8B/tree/main", "Qwen/Qwen3-8B"),
        ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        ("not-a-repo", ""),
    ],
)
def test_normalize_hf_repo_id(url, expected):
    assert mmj.normalize_hf_repo_id(url) == expected


def test_tokenizer_comes_from_the_base_model_and_renderer_from_the_renderer_repo(train_file, tmp_path):
    # Managed jobs tokenized with the base model and only chose the renderer from this repo.
    base = {**SHAPE_MODEL, "huggingFaceUrl": "https://huggingface.co/Qwen/Qwen3-8B"}
    job = {**FULL_PARAM_SFT, "rendererHuggingFaceRepoId": "meta-llama/Llama-3.1-8B-Instruct"}
    cfg = _plan(job, train_file, tmp_path, base_model=base).config
    assert cfg.tokenizer_model == "Qwen/Qwen3-8B"
    assert (cfg.renderer_name, cfg.renderer_name_is_resolved) == ("llama3", True)

    override = _plan(job, train_file, tmp_path, base_model=base, tokenizer_model="org/tok").config
    assert override.tokenizer_model == "org/tok"


def test_renderer_repo_matching_the_tokenizer_leaves_renderer_to_the_recipe(train_file, tmp_path):
    job = {**V2_LORA_SFT, "rendererHuggingFaceRepoId": "moonshotai/Kimi-K3"}
    cfg = _plan(job, train_file, tmp_path, base_model=POOL_MODEL).config
    assert cfg.tokenizer_model == "moonshotai/Kimi-K3"
    assert (cfg.renderer_name, cfg.renderer_name_is_resolved) == ("", False)


def test_unmappable_renderer_repo_warns_and_falls_back(train_file, tmp_path):
    job = {**V2_LORA_SFT, "rendererHuggingFaceRepoId": "not-a/real-model"}
    plan = _plan(job, train_file, tmp_path, base_model=POOL_MODEL)
    assert plan.config.renderer_name == ""
    assert any("can't map" in w for w in plan.warnings)


def test_renderer_repo_is_the_tokenizer_when_the_base_model_has_no_url(train_file, tmp_path):
    base = {**SHAPE_MODEL, "huggingFaceUrl": ""}
    job = {**FULL_PARAM_SFT, "rendererHuggingFaceRepoId": "Qwen/Qwen3.5-9B"}
    cfg = _plan(job, train_file, tmp_path, base_model=base).config
    assert (cfg.tokenizer_model, cfg.renderer_name) == ("Qwen/Qwen3.5-9B", "")


def test_mixed_local_and_gcs_dataset_paths_are_rejected(train_file, tmp_path):
    with pytest.raises(mmj.MigrationError, match="both be local or both gs://"):
        _plan(V2_LORA_SFT, train_file, tmp_path, base_model=POOL_MODEL, eval_dataset_path="gs://bucket/eval.jsonl")


def test_missing_tokenizer_source_stops(train_file, tmp_path):
    model = {**SHAPE_MODEL, "huggingFaceUrl": ""}
    with pytest.raises(mmj.MigrationError, match="--tokenizer-model"):
        _plan(FULL_PARAM_SFT, train_file, tmp_path, base_model=model)


@pytest.mark.parametrize(
    "name",
    [
        "accounts/acme/supervisedFineTuningJobs/sft-1 extra",
        "accounts/acme/rlorTrainerJobs/t-1",
        "supervisedFineTuningJobs/sft-1",
    ],
)
def test_bad_job_names_are_rejected(name):
    with pytest.raises(mmj.MigrationError, match="--job must look like"):
        mmj.parse_job_name(name)


def test_missing_dataset_path_explains_hosted_data(train_file, tmp_path):
    with pytest.raises(mmj.MigrationError, match=f"accounts/{ACCOUNT}/datasets/train.*your own copy"):
        _plan(V2_LORA_SFT, train_file, tmp_path, base_model=POOL_MODEL, dataset_path="")


def test_dedicated_runs_warn_about_gpu_quota(train_file, tmp_path):
    serverless = _plan(V2_LORA_SFT, train_file, tmp_path, base_model=POOL_MODEL)
    assert not any("quota" in w for w in serverless.warnings)

    lora = _plan(FULL_PARAM_SFT, train_file, tmp_path, base_model=SHAPE_MODEL)
    assert any("tier_required" in w and "quota_exceeded" in w and "can differ" in w for w in lora.warnings)

    full_dpo = _plan(DPO_FULL, train_file, tmp_path, base_model=SHAPE_MODEL)
    assert any("reference trainer" in w for w in full_dpo.warnings)


def test_unsupported_preference_method_stops(train_file, tmp_path):
    job = {**DPO_LORA, "lossConfig": {"method": "GRPO"}}
    with pytest.raises(mmj.MigrationError, match="GRPO"):
        _plan(job, train_file, tmp_path, base_model=POOL_MODEL)


# ---------------------------------------------------------------------------
# API reads and the command line
# ---------------------------------------------------------------------------


class FakeReader:
    def __init__(self, job, models, *, account=ACCOUNT, datasets=None):
        self.job = job
        self.models = models
        self.datasets = datasets or {}
        self.account = account

    def key_account_id(self):
        return self.account

    def get_job(self, ref):
        return self.job

    def get_model(self, name):
        return self.models.get(name)

    def get_dataset(self, name):
        return self.datasets.get(name)


def _models(*models):
    return {m["name"]: m for m in models}


def test_read_plan_downloads_example_instead_of_missing_fireworks_dataset(train_file, tmp_path, monkeypatch):
    downloaded = tmp_path / "logs" / "examples" / "prompt-router.jsonl"
    downloaded.parent.mkdir(parents=True)
    downloaded.write_text("{}\n")

    def download(example, output):
        assert example == "prompt-router"
        assert output == str(downloaded)
        return str(downloaded)

    monkeypatch.setattr(mmj, "download_example_dataset", download)
    reader = FakeReader({**V2_LORA_SFT, "dataset": f"accounts/{ACCOUNT}/datasets/onboarding-prompt-router"}, _models(POOL_MODEL))
    inputs = _inputs(train_file, tmp_path, dataset_path="", hf_example="prompt-router")

    plan = mmj.read_plan(reader, inputs, SFT_JOB)

    assert plan.config.dataset == str(downloaded)


def test_example_and_dataset_path_are_mutually_exclusive(train_file, tmp_path, monkeypatch):
    monkeypatch.setattr(mmj, "download_example_dataset", lambda *_: str(train_file))
    reader = FakeReader(V2_LORA_SFT, _models(POOL_MODEL))
    with pytest.raises(mmj.MigrationError, match="either --dataset-path or --hf-example"):
        mmj.read_plan(reader, _inputs(train_file, tmp_path, hf_example="prompt-router"), SFT_JOB)


def test_read_plan_rejects_key_from_another_account(train_file, tmp_path):
    reader = FakeReader(V2_LORA_SFT, _models(POOL_MODEL), account="other")
    with pytest.raises(mmj.MigrationError, match="belongs to account 'other'"):
        mmj.read_plan(reader, _inputs(train_file, tmp_path), SFT_JOB)


def test_read_plan_detects_cmek_dpo_job_from_output_model(train_file, tmp_path):
    output = {"name": f"accounts/{ACCOUNT}/models/old-dpo", "encryptionState": "ENCRYPTION_STATE_CMEK"}
    reader = FakeReader(DPO_LORA, _models(POOL_MODEL, output))
    with pytest.raises(mmj.MigrationError, match="CMEK"):
        mmj.read_plan(reader, _inputs(train_file, tmp_path), DPO_JOB)


def test_read_plan_continues_when_dpo_dataset_no_longer_exists(train_file, tmp_path):
    reader = FakeReader(DPO_LORA, _models(POOL_MODEL), datasets={})

    plan = mmj.read_plan(reader, _inputs(train_file, tmp_path), DPO_JOB)

    assert plan.config.dataset == str(train_file)


def test_read_plan_resolves_warm_start_base_model(train_file, tmp_path):
    reader = FakeReader(WARM_START_SFT, _models(POOL_MODEL, ADAPTER))
    plan = mmj.read_plan(reader, _inputs(train_file, tmp_path), SFT_JOB)
    assert plan.config.base_model == POOL_MODEL["name"]


def _copied_base(model_type):
    return {
        "name": f"accounts/{ACCOUNT}/models/qwen3p5-9b",
        "kind": "HF_BASE_MODEL",
        "trainingContextLength": 8192,
        "baseModelDetails": {"modelType": model_type},
    }


def test_read_plan_takes_tokenizer_from_catalog_model_for_a_copied_base(train_file, tmp_path):
    copied = _copied_base("qwen3_5")
    catalog = {**SHAPE_MODEL, "baseModelDetails": {"modelType": "qwen3_5"}}
    reader = FakeReader({**FULL_PARAM_SFT, "baseModel": copied["name"]}, _models(copied, catalog))
    plan = mmj.read_plan(reader, _inputs(train_file, tmp_path), SFT_JOB)
    assert plan.config.tokenizer_model == "Qwen/Qwen3.5-9B"
    assert any("same ID and model type" in w for w in plan.warnings)


def test_read_plan_ignores_a_catalog_model_of_another_architecture(train_file, tmp_path):
    copied = _copied_base("llama")
    catalog = {**SHAPE_MODEL, "baseModelDetails": {"modelType": "qwen3_5"}}
    reader = FakeReader({**FULL_PARAM_SFT, "baseModel": copied["name"]}, _models(copied, catalog))
    with pytest.raises(mmj.MigrationError, match="--tokenizer-model"):
        mmj.read_plan(reader, _inputs(train_file, tmp_path), SFT_JOB)


def test_multi_account_key_stops_with_clear_message(monkeypatch):
    class MultiAccountClient:
        def __init__(self, **kwargs):
            pass

        @property
        def account_id(self):
            raise ValueError("API key has access to multiple accounts: ['a', 'b']. This is not supported for firetitan training.")

        def close(self):
            pass

    monkeypatch.setattr(mmj, "FireworksClient", MultiAccountClient)
    reader = mmj.FireworksReader("fw_test")
    with pytest.raises(mmj.MigrationError, match="multiple accounts.*job's account only"):
        reader.key_account_id()


def _run_main(monkeypatch, reader, argv):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test")
    monkeypatch.setattr(mmj, "FireworksReader", lambda api_key, base_url=None: reader)
    return mmj.main(argv)


def _argv(train_file, tmp_path, job=SFT_JOB, *extra):
    return [
        "--job", job,
        "--dataset-path", str(train_file),
        "--output-model-id", "new-model",
        "--log-path", str(tmp_path / "logs"),
        *extra,
    ]  # fmt: skip


def test_dry_run_prints_config_and_does_not_train(monkeypatch, train_file, tmp_path, capsys):
    monkeypatch.setattr(sft_loop, "main", lambda cfg: pytest.fail("dry run must not train"))
    plan = _run_main(monkeypatch, FakeReader(V2_LORA_SFT, _models(POOL_MODEL)), _argv(train_file, tmp_path, SFT_JOB, "--dry-run"))

    out = capsys.readouterr().out
    assert "Mode:         serverless" in out
    assert "training.recipes.sft_loop" in out
    config_json = json.loads(out[out.index("{") : out.rindex("}") + 1])
    assert config_json["output_model_id"] == "new-model"
    assert config_json["lr_scheduler"]["type"] == "cosine"
    assert plan.mode is mmj.Mode.SERVERLESS


def test_run_passes_config_to_recipe(monkeypatch, train_file, tmp_path):
    received = []
    monkeypatch.setattr(dpo_loop, "main", lambda cfg: received.append(cfg) or {"steps": 1})
    _run_main(monkeypatch, FakeReader(DPO_FULL, _models(SHAPE_MODEL)), _argv(train_file, tmp_path, DPO_JOB))

    (cfg,) = received
    assert isinstance(cfg, dpo_loop.Config)
    assert cfg.serverless is False and cfg.output_model_id == "new-model"


def test_auto_mode_falls_back_to_dedicated_when_serverless_is_unavailable(monkeypatch, train_file, tmp_path):
    received = []

    def fake_main(cfg):
        received.append(cfg)
        if cfg.serverless:
            raise RuntimeError("no eligible shared trainer found for base model accounts/fireworks/models/kimi-k3")
        return {"steps": 1}

    monkeypatch.setattr(sft_loop, "main", fake_main)
    _run_main(monkeypatch, FakeReader(V2_LORA_SFT, _models(POOL_MODEL)), _argv(train_file, tmp_path))

    assert [cfg.serverless for cfg in received] == [True, False]


def test_no_fallback_after_training_started(monkeypatch, train_file, tmp_path):
    def fake_main(cfg):
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "logs" / "dataloader.json").write_text("{}")
        raise RuntimeError("no eligible shared trainer found")

    monkeypatch.setattr(sft_loop, "main", fake_main)
    with pytest.raises(RuntimeError, match="no eligible shared trainer"):
        _run_main(monkeypatch, FakeReader(V2_LORA_SFT, _models(POOL_MODEL)), _argv(train_file, tmp_path))


def test_no_fallback_for_other_errors_or_explicit_serverless(monkeypatch, train_file, tmp_path):
    calls = []

    def fake_main(cfg):
        calls.append(cfg)
        raise RuntimeError("base model accounts/fireworks/models/kimi-k3 is not available for serverless training")

    monkeypatch.setattr(sft_loop, "main", fake_main)
    with pytest.raises(RuntimeError):
        _run_main(monkeypatch, FakeReader(V2_LORA_SFT, _models(POOL_MODEL)), _argv(train_file, tmp_path, SFT_JOB, "--mode", "serverless"))
    assert len(calls) == 1

    def other_error(cfg):
        raise RuntimeError("dataset row 3 is malformed")

    monkeypatch.setattr(sft_loop, "main", other_error)
    with pytest.raises(RuntimeError, match="malformed"):
        _run_main(monkeypatch, FakeReader(V2_LORA_SFT, _models(POOL_MODEL)), _argv(train_file, tmp_path))


def test_main_reports_migration_errors_as_exit(monkeypatch, train_file, tmp_path):
    with pytest.raises(SystemExit, match="error: .*belongs to account 'other'"):
        _run_main(monkeypatch, FakeReader(V2_LORA_SFT, _models(POOL_MODEL), account="other"), _argv(train_file, tmp_path))


def test_main_requires_api_key(monkeypatch, train_file, tmp_path):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setattr(mmj, "load_dotenv", lambda: None)
    with pytest.raises(SystemExit, match="FIREWORKS_API_KEY"):
        mmj.main(_argv(train_file, tmp_path))


def _stop_after_checkpoint(log_line, tmp_path):
    def fake_main(cfg):
        logging.getLogger("fireworks.training.sdk.trainer").info(log_line)
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": 2, "checkpoints": {"step-20": {"step": 20, "data_consumed": 160}, "step-40": {"step": 40, "data_consumed": 320}}}
        (tmp_path / "logs" / "dataloader.json").write_text(json.dumps(manifest))
        raise KeyboardInterrupt

    return fake_main


def test_interrupted_dedicated_run_prints_resume_command(monkeypatch, train_file, tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(dpo_loop, "main", _stop_after_checkpoint("Created trainer job: tj-abc123", tmp_path))
    with pytest.raises(KeyboardInterrupt):
        _run_main(monkeypatch, FakeReader(DPO_FULL, _models(SHAPE_MODEL)), _argv(train_file, tmp_path, DPO_JOB))

    err = capsys.readouterr().err
    assert f"--log-path {tmp_path / 'logs'} --init-from-checkpoint tj-abc123:step-40" in err
    # A new trainer can't take over the deleted one's dataset position.
    assert "dataset restarts from its first row" in err


def test_failed_serverless_run_prints_run_scoped_resume_command(monkeypatch, train_file, tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO)
    line = "Serverless training session sess-1, run run-xyz. To continue from one of this run's checkpoints"
    monkeypatch.setattr(sft_loop, "main", _stop_after_checkpoint(line, tmp_path))
    with pytest.raises(KeyboardInterrupt):
        _run_main(monkeypatch, FakeReader(V2_LORA_SFT, _models(POOL_MODEL)), _argv(train_file, tmp_path))

    err = capsys.readouterr().err
    assert f"To resume, rerun this command with --log-path {tmp_path / 'logs'} --init-from-checkpoint {ACCOUNT}/run-xyz/step-40" in err


def test_run_without_checkpoint_says_rerun_starts_over(monkeypatch, train_file, tmp_path, capsys):
    def fail(cfg):
        raise RuntimeError("dataset row 3 is malformed")

    monkeypatch.setattr(dpo_loop, "main", fail)
    with pytest.raises(RuntimeError):
        _run_main(monkeypatch, FakeReader(DPO_FULL, _models(SHAPE_MODEL)), _argv(train_file, tmp_path, DPO_JOB))
    assert "starts training over" in capsys.readouterr().err


def test_trainer_job_id_reattaches_on_dedicated(train_file, tmp_path):
    plan = _plan(V2_LORA_SFT, train_file, tmp_path, base_model=POOL_MODEL, trainer_job_id="tj-keep")

    assert plan.mode is mmj.Mode.DEDICATED
    assert plan.config.trainer.job_id == "tj-keep"
    assert plan.fallback_config is None
    with pytest.raises(mmj.MigrationError, match="can't run on serverless"):
        _plan(V2_LORA_SFT, train_file, tmp_path, base_model=POOL_MODEL, trainer_job_id="tj-keep", mode="serverless")


KEPT = "The run stopped before it finished, so trainer tj-keep was kept. To resume from its last saved checkpoint, ..."


def _stop(log_line, tmp_path, *, checkpoint):
    def fake_main(cfg):
        logging.getLogger("training.recipes").warning(log_line)
        if checkpoint:
            (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
            manifest = {"schema_version": 2, "checkpoints": {"step-20": {"step": 20, "data_consumed": 160}}}
            (tmp_path / "logs" / "dataloader.json").write_text(json.dumps(manifest))
        raise KeyboardInterrupt

    return fake_main


@pytest.mark.parametrize(
    ("job", "models", "recipe", "checkpoint", "expected", "unexpected"),
    [
        # SFT/DPO with a checkpoint: reattach resumes step, weights and dataset position.
        (FULL_PARAM_SFT, (SHAPE_MODEL,), sft_loop, True, "--trainer-job-id tj-keep.", "first row"),
        # ORPO restores the step but replays the dataset.
        (ORPO, (POOL_MODEL,), orpo_loop, True, "reads the dataset from its first row", None),
        # LoRA without a checkpoint: start over on the kept trainer.
        (LEGACY_V1_SFT, (SHAPE_MODEL,), sft_loop, False, "To start over on the kept trainer", None),
        # Full-parameter without a checkpoint: the trainer holds stale state; delete it.
        (FULL_PARAM_SFT, (SHAPE_MODEL,), sft_loop, False, "firectl rlor-trainer-job delete tj-keep", "--trainer-job-id tj-keep."),
    ],
    ids=["sft-checkpoint", "orpo-checkpoint", "lora-no-checkpoint", "full-param-no-checkpoint"],
)
def test_kept_trainer_hints(monkeypatch, train_file, tmp_path, capsys, caplog, job, models, recipe, checkpoint, expected, unexpected):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(recipe, "main", _stop(KEPT, tmp_path, checkpoint=checkpoint))
    job_name = SFT_JOB if job["name"] == SFT_JOB else DPO_JOB
    with pytest.raises(KeyboardInterrupt):
        _run_main(monkeypatch, FakeReader(job, _models(*models)), _argv(train_file, tmp_path, job_name))

    err = capsys.readouterr().err
    assert expected in err
    if unexpected:
        assert unexpected not in err


def test_resume_hint_reads_gcs_log_paths(monkeypatch, train_file, tmp_path):
    manifest = json.dumps({"schema_version": 2, "checkpoints": {"step-20": {"step": 20, "data_consumed": 160}}})
    reads = []
    monkeypatch.setattr(mmj.fileio, "read_text", lambda path: reads.append(path) or manifest)
    plan = _plan(FULL_PARAM_SFT, train_file, tmp_path, base_model=SHAPE_MODEL, log_path="gs://bucket/run")
    recorder = mmj.ResumeRecorder()
    recorder.kept_trainer_job_id = "tj-keep"

    hint = mmj.resume_hint(plan, recorder)

    assert reads == ["gs://bucket/run/dataloader.json"]
    assert "--log-path gs://bucket/run --trainer-job-id tj-keep." in hint
    assert mmj._training_started("gs://bucket/run") is True


def test_unreadable_checkpoint_list_is_reported_not_guessed(monkeypatch, train_file, tmp_path):
    def denied(path):
        raise PermissionError("403 storage.objects.get")

    monkeypatch.setattr(mmj.fileio, "read_text", denied)
    plan = _plan(FULL_PARAM_SFT, train_file, tmp_path, base_model=SHAPE_MODEL, log_path="gs://bucket/run")
    recorder = mmj.ResumeRecorder()
    recorder.kept_trainer_job_id = "tj-keep"

    hint = mmj.resume_hint(plan, recorder)

    assert hint.startswith("Couldn't read gs://bucket/run/dataloader.json")
    assert "delete" not in hint
    # Unknown state never triggers the serverless-to-dedicated fallback.
    assert mmj._training_started("gs://bucket/run") is True


def test_serverless_concurrency_limit_explains_and_does_not_fall_back(monkeypatch, train_file, tmp_path, capsys):
    calls = []

    def at_limit(cfg):
        calls.append(cfg)
        raise RuntimeError(
            "create_model: maximum concurrent serverless training runs for base model "
            "accounts/fireworks/models/kimi-k3 exceeded: in use 1, limit 1"
        )

    monkeypatch.setattr(sft_loop, "main", at_limit)
    with pytest.raises(RuntimeError, match="maximum concurrent"):
        _run_main(monkeypatch, FakeReader(V2_LORA_SFT, _models(POOL_MODEL)), _argv(train_file, tmp_path))

    assert [cfg.serverless for cfg in calls] == [True]
    assert "Wait and rerun, or use --mode dedicated" in capsys.readouterr().err
