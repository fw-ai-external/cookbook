#!/usr/bin/env python3
# ruff: noqa: E402
"""Rerun a managed SFT, DPO or ORPO job on the Training API.

This tool reads a managed job's settings, maps them onto the matching cookbook
recipe ``Config`` (``sft_loop``, ``dpo_loop`` or ``orpo_loop``), and runs it on
serverless or dedicated training. The run ends by promoting the final
checkpoint to a new deployable model.

Fireworks-hosted datasets can't be downloaded, so bring a local copy of the
training data (or a ``gs://`` path you can read). The API key must belong to
the job's account and only that account.

Usage (from ``training/``)::

    export FIREWORKS_API_KEY=...
    python -m examples.tools.migrate_managed_job \\
        --job accounts/<account>/supervisedFineTuningJobs/<job-id> \\
        --dataset-path ./train.jsonl \\
        --output-model-id <new-model-id> \\
        --mode auto --dry-run

Drop ``--dry-run`` to start training. ``--mode auto`` picks serverless when the
job can run there and falls back to dedicated when serverless is unavailable.
If a run stops early, the tool prints how to continue it from its newest
checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import httpx
from dotenv import load_dotenv
from fireworks import APIStatusError, Fireworks, NotFoundError

_COOKBOOK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

from fireworks.training.sdk import FireworksClient
from fireworks.training.sdk.training_spec import (
    ConstantSchedule,
    CosineSchedule,
    LinearSchedule,
    LRSchedulerSpec,
)

import training.recipes.dpo_loop as dpo_loop
import training.recipes.orpo_loop as orpo_loop
import training.recipes.sft_loop as sft_loop
from training.utils import RunnerConfig, TrainerConfig, WandBConfig, fileio, resolve_renderer_snapshot
from training.utils.checkpoints import DATALOADER_BASE_NAME
from training.utils.validation import validate_output_model_id

logger = logging.getLogger(__name__)

# Managed SFT/DPO/ORPO saved a resumable checkpoint every 20 steps, and managed
# LoRA SFT/DPO also saved resumable + promotable checkpoints every 30 minutes
# and at each epoch end. The recipe defaults (0) turn both off, so copy them.
MANAGED_DCP_SAVE_INTERVAL = 20
MANAGED_CHECKPOINT_INTERVAL_MINUTES = 30.0

# Serverless LoRA hosting accepts adapters up to this rank.
SERVERLESS_LORA_MAX_RANK = 64

# Serverless setup fails with one of these before any training step when the
# pool can't take the run; --mode auto then retries on a dedicated trainer.
SERVERLESS_UNAVAILABLE_ERRORS = (
    "not available for serverless training",
    "no eligible shared trainer",
)

# Serverless returns this (HTTP 429) when the account already runs its limit of
# sessions on the base model; waiting frees the slot, so it isn't a fallback case.
SERVERLESS_CONCURRENCY_ERROR = "maximum concurrent serverless training runs"

CMEK_ENCRYPTION_STATE = "ENCRYPTION_STATE_CMEK"
FIREWORKS_ACCOUNT = "fireworks"
THINKING_MODE_PREFIX = "THINKING_TRACE_HISTORY_MODE_"

# Adapter target modules the Training API leaves out by default: MoE routers,
# the output head and vision towers. A warm-start adapter that trained them
# fails to load ("Adapter key mismatch").
_NON_DEFAULT_ADAPTER_LEAVES = frozenset({"gate", "lm_head", "output", "projection_head"})
_NON_DEFAULT_ADAPTER_SEGMENTS = ("router", "vision", "visual")

# Managed SFT fields that have no Training API equivalent.
_IGNORED_SFT_FIELDS = {
    "earlyStop": "early stopping",
    "isTurbo": "turbo mode",
    "jinjaTemplate": "a literal Jinja template",
    "mtpEnabled": "MTP training",
    "gradientAccumulationSteps": "gradient accumulation",
}

_JOB_NAME_RE = re.compile(
    r"^accounts/(?P<account>[a-z0-9-]+)/(?P<collection>supervisedFineTuningJobs|dpoJobs)/(?P<job>[a-z0-9-]+)$"
)
_HF_PATH_STOP_SEGMENTS = {"tree", "blob", "resolve"}

# Log lines that name what a resume needs: the SDK's dedicated trainer job, and
# the cookbook's serverless run.
_TRAINER_JOB_LOG_RE = re.compile(
    r"(?:Created|Resumed|Reusing) trainer job:? (?:accounts/[^/]+/rlorTrainerJobs/)?(?P<job>[a-z0-9-]+)"
)
# Recipes that keep a dedicated trainer when a run stops early log this warning.
_KEPT_TRAINER_LOG_RE = re.compile(r"trainer (?P<job>\S+) was kept")
_SERVERLESS_RUN_LOG_RE = re.compile(r"Serverless training session \S+, run (?P<run>[^\s.]+)\.")


class MigrationError(ValueError):
    """The job can't be migrated as requested; the message says why."""


class Method(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    ORPO = "orpo"


class Mode(str, Enum):
    SERVERLESS = "serverless"
    DEDICATED = "dedicated"


@dataclass(frozen=True)
class JobRef:
    account_id: str
    collection: str
    job_id: str

    @property
    def name(self) -> str:
        return f"accounts/{self.account_id}/{self.collection}/{self.job_id}"

    @property
    def is_sft(self) -> bool:
        return self.collection == "supervisedFineTuningJobs"


def parse_job_name(name: str) -> JobRef:
    match = _JOB_NAME_RE.match(name.strip())
    if not match:
        raise MigrationError(
            f"--job must look like accounts/<account>/supervisedFineTuningJobs/<id> or "
            f"accounts/<account>/dpoJobs/<id>; got {name!r}."
        )
    return JobRef(match["account"], match["collection"], match["job"])


@dataclass(frozen=True)
class ManagedSettings:
    """One managed job's training settings, flattened from the SFT or DPO shape."""

    method: Method
    base_model: str
    warm_start_from: str
    dataset: str
    evaluation_dataset: str
    eval_auto_carveout: bool
    generation_eval: bool
    epochs: int
    learning_rate: float
    lr_scheduler: dict[str, Any] | None
    warmup_steps: int
    batch_size_samples: int
    legacy_batch_size: int
    weight_decay: float
    lora_rank: int
    lora_alpha: int
    lora_target_modules: tuple[str, ...]
    lora_dropout: float
    tune_vision_encoder: bool
    max_context_length: int
    output_model: str
    thinking_trace_history_mode: str
    renderer_hf_repo_id: str
    sampler_save_interval: int
    wandb: dict[str, Any] | None
    use_reservation: bool
    reservation_target: str
    purpose: str
    encryption_state: str
    external_dataset_source: str
    beta: float | None = None
    ref_cache_concurrency: int | None = None
    ref_cache_batch_size: int | None = None
    orpo_lambda: float | None = None
    ignored_fields: tuple[str, ...] = ()


def _external_source(job: dict[str, Any]) -> str:
    # The API returns these configs with empty strings when they're unused.
    for key, label in (("awsS3Config", "S3"), ("azureBlobStorageConfig", "Azure Blob Storage")):
        if any((job.get(key) or {}).values()):
            return label
    return ""


def _purpose(job: dict[str, Any]) -> str:
    purpose = job.get("purpose") or ""
    return "" if purpose == "PURPOSE_UNSPECIFIED" else purpose


def settings_from_sft_job(job: dict[str, Any]) -> ManagedSettings:
    """Flatten a ``SupervisedFineTuningJob`` read with ``to_dict(mode="json")``."""
    ignored = tuple(label for key, label in _IGNORED_SFT_FIELDS.items() if job.get(key))
    return ManagedSettings(
        method=Method.SFT,
        base_model=job.get("baseModel") or "",
        warm_start_from=job.get("warmStartFrom") or "",
        dataset=job.get("dataset") or "",
        evaluation_dataset=job.get("evaluationDataset") or "",
        eval_auto_carveout=bool(job.get("evalAutoCarveout")),
        generation_eval=bool(job.get("generationEval")),
        epochs=int(job.get("epochs") or 0),
        learning_rate=float(job.get("learningRate") or 0.0),
        lr_scheduler=job.get("lrScheduler") or None,
        warmup_steps=int(job.get("learningRateWarmupSteps") or 0),
        batch_size_samples=int(job.get("batchSizeSamples") or 0),
        legacy_batch_size=int(job.get("batchSize") or 0),
        weight_decay=float(job.get("optimizerWeightDecay") or 0.0),
        lora_rank=int(job.get("loraRank") or 0),
        lora_alpha=int(job.get("loraAlpha") or 0),
        lora_target_modules=(),
        lora_dropout=0.0,
        tune_vision_encoder=bool(job.get("tuneVisionEncoder")),
        max_context_length=int(job.get("maxContextLength") or 0),
        output_model=job.get("outputModel") or "",
        thinking_trace_history_mode=job.get("thinkingTraceHistoryMode") or "",
        renderer_hf_repo_id=job.get("rendererHuggingFaceRepoId") or "",
        sampler_save_interval=int(job.get("samplerCheckpointSaveIntervalSteps") or 0),
        wandb=job.get("wandbConfig") or None,
        use_reservation=bool(job.get("useReservation")),
        reservation_target=job.get("reservationTarget") or "",
        purpose=_purpose(job),
        encryption_state=job.get("encryptionState") or "",
        external_dataset_source=_external_source(job),
        ignored_fields=ignored,
    )


def settings_from_dpo_job(job: dict[str, Any]) -> ManagedSettings:
    """Flatten a ``DpoJob`` (DPO or ORPO) read with ``to_dict(mode="json")``."""
    tc = job.get("trainingConfig") or {}
    loss = job.get("lossConfig") or {}
    method_name = loss.get("method") or "DPO"
    if method_name in ("DPO", "METHOD_UNSPECIFIED"):
        method = Method.DPO
    elif method_name == "ORPO":
        method = Method.ORPO
    else:
        raise MigrationError(f"Preference job loss method {method_name} has no cookbook migration; only DPO and ORPO do.")
    dpo = loss.get("dpo") or {}
    orpo = loss.get("orpo") or {}
    return ManagedSettings(
        method=method,
        base_model=tc.get("baseModel") or "",
        warm_start_from=tc.get("warmStartFrom") or "",
        dataset=job.get("dataset") or "",
        evaluation_dataset=job.get("evaluationDataset") or "",
        eval_auto_carveout=bool(job.get("evalAutoCarveout")),
        generation_eval=False,
        epochs=int(tc.get("epochs") or 0),
        learning_rate=float(tc.get("learningRate") or 0.0),
        lr_scheduler=tc.get("lrScheduler") or None,
        warmup_steps=int(tc.get("learningRateWarmupSteps") or 0),
        batch_size_samples=int(tc.get("batchSizeSamples") or 0),
        legacy_batch_size=int(tc.get("batchSize") or 0),
        weight_decay=float(tc.get("optimizerWeightDecay") or 0.0),
        lora_rank=int(tc.get("loraRank") or 0),
        lora_alpha=int(tc.get("loraAlpha") or 0),
        lora_target_modules=tuple(tc.get("loraTargetModules") or ()),
        lora_dropout=float(tc.get("loraDropout") or 0.0),
        tune_vision_encoder=False,
        max_context_length=int(tc.get("maxContextLength") or 0),
        output_model=tc.get("outputModel") or "",
        thinking_trace_history_mode=job.get("thinkingTraceHistoryMode") or "",
        renderer_hf_repo_id=job.get("rendererHuggingFaceRepoId") or "",
        sampler_save_interval=int(tc.get("samplerCheckpointSaveIntervalSteps") or 0),
        wandb=job.get("wandbConfig") or None,
        use_reservation=bool(job.get("useReservation")),
        reservation_target=job.get("reservationTarget") or "",
        purpose=_purpose(job),
        encryption_state=job.get("encryptionState") or "",
        external_dataset_source=_external_source(job),
        beta=dpo.get("beta"),
        ref_cache_concurrency=dpo.get("refCacheConcurrency"),
        ref_cache_batch_size=dpo.get("refCacheBatchSize"),
        orpo_lambda=orpo.get("lambda"),
    )


def normalize_hf_repo_id(value: str) -> str:
    """``https://huggingface.co/Qwen/Qwen3-8B/tree/main`` -> ``Qwen/Qwen3-8B``."""
    raw = value.strip()
    if "huggingface.co/" in raw:
        raw = raw.split("huggingface.co/", 1)[1]
    segments = []
    for segment in raw.strip("/").split("/"):
        if segment in _HF_PATH_STOP_SEGMENTS:
            break
        segments.append(segment)
    return "/".join(segments[:2]) if len(segments) >= 2 else ""


def lr_scheduler_from_managed(scheduler: dict[str, Any] | None, warmup_steps: int) -> LRSchedulerSpec | None:
    """Map the managed ``lrScheduler`` oneof plus warmup onto the recipe spec.

    Returns ``None`` when the job set neither, so the recipe default applies.
    """
    warmup = max(warmup_steps, 0)
    scheduler = scheduler or {}
    for key, schedule_cls in (("linear", LinearSchedule), ("cosine", CosineSchedule)):
        if key in scheduler:
            knobs = scheduler[key] or {}
            kwargs: dict[str, Any] = {"warmup_steps": warmup, "min_lr_ratio": float(knobs.get("minLrRatio") or 0.0)}
            if knobs.get("decayRatio"):
                kwargs["decay_ratio"] = float(knobs["decayRatio"])
            return schedule_cls(**kwargs)
    if "constant" in scheduler or warmup:
        return ConstantSchedule(warmup_steps=warmup)
    return None


def thinking_mode_from_managed(value: str) -> str:
    mode = value.removeprefix(THINKING_MODE_PREFIX).lower()
    return "" if mode in ("", "unspecified") else mode


def adapter_has_non_default_modules(target_modules: list[str] | tuple[str, ...]) -> list[str]:
    flagged = []
    for module in target_modules:
        parts = module.split(".")
        if parts[-1] in _NON_DEFAULT_ADAPTER_LEAVES or any(
            seg in part for part in parts for seg in _NON_DEFAULT_ADAPTER_SEGMENTS
        ):
            flagged.append(module)
    return flagged


@dataclass
class MigrationInputs:
    """What the user supplies on the command line."""

    dataset_path: str
    output_model_id: str
    log_path: str
    mode: str = "auto"
    eval_dataset_path: str = ""
    tokenizer_model: str = ""
    max_seq_len: int = 0
    init_from_checkpoint: str = ""
    trainer_job_id: str = ""


@dataclass
class MigrationPlan:
    job: JobRef
    settings: ManagedSettings
    mode: Mode
    mode_reasons: list[str]
    config: Any
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    fallback_config: Any = None

    @property
    def recipe(self) -> str:
        return {Method.SFT: "sft_loop", Method.DPO: "dpo_loop", Method.ORPO: "orpo_loop"}[self.settings.method]


def _dedicated_reasons(
    settings: ManagedSettings, base_model: dict[str, Any], lora_rank: int
) -> list[str]:
    reasons = []
    if lora_rank <= 0:
        reasons.append("full-parameter training (lora_rank=0) runs only on dedicated trainers")
    if settings.method is Method.ORPO:
        reasons.append("ORPO runs only on dedicated trainers")
    if settings.warm_start_from:
        reasons.append("warm starts from an adapter run only on dedicated trainers")
    if settings.use_reservation or settings.reservation_target:
        reasons.append("the job used a reservation, which only dedicated trainers consume")
    if settings.tune_vision_encoder:
        reasons.append("the job tuned the vision encoder")
    if not base_model.get("supportsServerlessTraining"):
        reasons.append(f"{base_model.get('name', 'the base model')} is not available for serverless training")
    return reasons


def choose_mode(requested: str, dedicated_reasons: list[str]) -> tuple[Mode, list[str]]:
    if requested == Mode.DEDICATED.value:
        return Mode.DEDICATED, ["--mode dedicated"]
    if requested == Mode.SERVERLESS.value:
        if dedicated_reasons:
            raise MigrationError("This job can't run on serverless training: " + "; ".join(dedicated_reasons) + ". Use --mode dedicated or --mode auto.")
        return Mode.SERVERLESS, ["--mode serverless"]
    if dedicated_reasons:
        return Mode.DEDICATED, dedicated_reasons
    return Mode.SERVERLESS, ["LoRA on a serverless-trainable base model, with no warm start, reservation, vision tuning or ORPO"]


def _resolve_tokenizer(inputs: MigrationInputs, settings: ManagedSettings, base_model: dict[str, Any]) -> str:
    """The training tokenizer: the base model's own repo, as managed jobs used.

    The job's renderer repo is a fallback only when the base model has none.
    """
    if inputs.tokenizer_model:
        return inputs.tokenizer_model
    for candidate in (base_model.get("huggingFaceUrl") or "", settings.renderer_hf_repo_id):
        repo_id = normalize_hf_repo_id(candidate) if candidate else ""
        if repo_id:
            return repo_id
    raise MigrationError(
        f"Can't tell which Hugging Face tokenizer {base_model.get('name', 'the base model')} uses: the job has "
        "no renderer repo and the model has no Hugging Face URL. Pass --tokenizer-model <org/name>; for a "
        "model you uploaded, use the Hugging Face repo it came from."
    )


def _resolve_renderer(settings: ManagedSettings, tokenizer_model: str, warnings: list[str]) -> str:
    """The renderer the job's renderer repo selects, when it differs from the tokenizer's.

    Managed jobs rendered chats with the renderer chosen from that repo while
    tokenizing with the base model; return it as a resolved renderer name, or
    "" to let the recipe pick from ``tokenizer_model``.
    """
    renderer_repo = normalize_hf_repo_id(settings.renderer_hf_repo_id) if settings.renderer_hf_repo_id else ""
    if not renderer_repo or renderer_repo == tokenizer_model:
        return ""
    try:
        return resolve_renderer_snapshot(
            tokenizer_model=renderer_repo,
            renderer_name="",
            thinking_trace_history_mode=thinking_mode_from_managed(settings.thinking_trace_history_mode),
        )
    except ValueError as exc:
        warnings.append(
            f"The job rendered chats with the {renderer_repo} renderer, which the cookbook can't map ({exc}). "
            f"The recipe picks the renderer from {tokenizer_model} instead."
        )
        return ""


def _resolve_max_seq_len(inputs: MigrationInputs, settings: ManagedSettings, base_model: dict[str, Any]) -> int:
    for value in (inputs.max_seq_len, settings.max_context_length, base_model.get("trainingContextLength")):
        if value and int(value) > 0:
            return int(value)
    raise MigrationError("The job has no max_context_length and the base model reports none. Pass --max-seq-len.")


def _check_dataset_path(path: str, flag: str) -> None:
    if path.startswith("accounts/"):
        raise MigrationError(
            f"{flag} must be a local file or directory, or a gs:// path you can read. Fireworks dataset "
            "resource names can't be read by the Training API, and hosted datasets can't be downloaded."
        )
    if not path.startswith("gs://") and not os.path.exists(path):
        raise MigrationError(f"{flag} {path!r} does not exist.")


def build_plan(
    job_ref: JobRef,
    job: dict[str, Any],
    inputs: MigrationInputs,
    *,
    base_model: dict[str, Any],
    adapter: dict[str, Any] | None = None,
) -> MigrationPlan:
    """Map a managed job onto a recipe config. Pure: all API reads are passed in.

    ``base_model`` and ``adapter`` are ``models.get(...).to_dict(mode="json")``
    payloads; ``adapter`` is the warm-start PEFT model, if any.
    """
    settings = settings_from_sft_job(job) if job_ref.is_sft else settings_from_dpo_job(job)
    warnings: list[str] = []
    notes: list[str] = []

    if settings.encryption_state == CMEK_ENCRYPTION_STATE:
        raise MigrationError(
            f"{job_ref.name} belongs to a CMEK (customer-managed encryption key) account. The Training API "
            "can't write CMEK-encrypted models yet, so this job can't be migrated. Contact your Fireworks "
            "account team."
        )

    if not inputs.dataset_path:
        raise MigrationError(
            f"{job_ref.name} trained on {settings.dataset or 'a Fireworks dataset'}, which Fireworks stores. "
            "The Training API can't read Fireworks datasets and they can't be downloaded, so migrating needs "
            "your own copy of the training data. Pass it with --dataset-path. If you no longer have a copy, "
            "contact your Fireworks account team."
        )
    _check_dataset_path(inputs.dataset_path, "--dataset-path")
    if inputs.eval_dataset_path:
        _check_dataset_path(inputs.eval_dataset_path, "--eval-dataset-path")
    if (
        settings.method is not Method.ORPO
        and inputs.eval_dataset_path
        and inputs.dataset_path.startswith("gs://") != inputs.eval_dataset_path.startswith("gs://")
    ):
        raise MigrationError(
            "--dataset-path and --eval-dataset-path must both be local or both gs:// paths: the recipe reads "
            "them with one loading mode."
        )
    if settings.method is Method.ORPO and inputs.dataset_path.startswith("gs://"):
        raise MigrationError("ORPO reads a local dataset only. Copy the data locally and pass --dataset-path.")

    errors = validate_output_model_id(inputs.output_model_id)
    if errors:
        raise MigrationError("\n".join(errors))
    old_output_id = settings.output_model.rsplit("/", 1)[-1]
    if inputs.output_model_id.rsplit("/", 1)[-1] == old_output_id:
        raise MigrationError(
            f"--output-model-id {inputs.output_model_id!r} is the managed job's output model, which already "
            "exists. Choose a new ID."
        )

    # Base model and LoRA rank. Warm starts take both from the adapter.
    base_model_name = settings.base_model
    lora_rank = settings.lora_rank
    if settings.warm_start_from:
        peft = (adapter or {}).get("peftDetails") or {}
        base_model_name = base_model_name or peft.get("baseModel") or ""
        adapter_rank = int(peft.get("r") or 0)
        if lora_rank <= 0:
            lora_rank = adapter_rank or 8
        elif adapter_rank and adapter_rank != lora_rank:
            warnings.append(f"The job's lora_rank {lora_rank} differs from the warm-start adapter's rank {adapter_rank}; using {lora_rank}.")
        flagged = adapter_has_non_default_modules(peft.get("targetModules") or [])
        if flagged:
            warnings.append(
                f"The warm-start adapter {settings.warm_start_from} trained modules the Training API doesn't "
                f"adapt by default ({', '.join(flagged)}). Loading it may fail with 'Adapter key mismatch'."
            )
    if not base_model_name:
        raise MigrationError(f"Can't find the base model of {job_ref.name}.")

    dedicated_reasons = _dedicated_reasons(settings, base_model, lora_rank)
    if inputs.trainer_job_id:
        dedicated_reasons.append(f"--trainer-job-id reattaches to dedicated trainer {inputs.trainer_job_id}")
    mode, mode_reasons = choose_mode(inputs.mode, dedicated_reasons)

    # Warnings for settings with no Training API equivalent.
    if settings.legacy_batch_size:
        batch_note = f"batch_size={settings.batch_size_samples}" if settings.batch_size_samples else "the recipe's default batch size"
        warnings.append(
            f"The job used the legacy (V1) token batch size {settings.legacy_batch_size}, which has no "
            f"Training API equivalent; using {batch_note} (rows per step). V1 jobs also used other trainer "
            "defaults, so results can differ."
        )
    if settings.lora_target_modules:
        warnings.append(f"lora_target_modules {list(settings.lora_target_modules)} is ignored; the Training API trains its default module set.")
    if settings.lora_dropout:
        warnings.append(f"lora_dropout {settings.lora_dropout} is ignored; the Training API trains without LoRA dropout.")
    if settings.tune_vision_encoder:
        warnings.append("tune_vision_encoder is ignored; the Training API does not tune the vision encoder.")
    for label in settings.ignored_fields:
        warnings.append(f"The job used {label}, which the Training API doesn't support; it is ignored.")
    if settings.external_dataset_source:
        warnings.append(f"The job read its dataset from {settings.external_dataset_source}. Copy the data yourself and pass it with --dataset-path.")
    if lora_rank > SERVERLESS_LORA_MAX_RANK:
        warnings.append(f"LoRA rank {lora_rank} is above {SERVERLESS_LORA_MAX_RANK}, so the new model can't be hosted as a serverless LoRA; deploy it on a dedicated deployment.")

    eval_dataset = inputs.eval_dataset_path
    eval_auto_carveout = settings.eval_auto_carveout and not eval_dataset
    if settings.evaluation_dataset and not eval_dataset:
        warnings.append(
            f"The job evaluated on {settings.evaluation_dataset}. Pass --eval-dataset-path with a local copy; "
            "this run has no held-out evaluation set."
        )
    if settings.method is Method.ORPO and (eval_dataset or eval_auto_carveout):
        warnings.append("ORPO has no held-out evaluation in the cookbook; the evaluation settings are ignored.")

    wandb = WandBConfig()
    if settings.wandb and settings.wandb.get("enabled"):
        wandb = WandBConfig(
            entity=settings.wandb.get("entity") or None,
            project=settings.wandb.get("project") or None,
            run_name=inputs.output_model_id.rsplit("/", 1)[-1],
        )
        if not os.environ.get("WANDB_API_KEY"):
            warnings.append("The job logged to Weights & Biases. Set WANDB_API_KEY to sync this run; without it W&B runs offline.")

    if settings.purpose:
        notes.append(f"trainer purpose {settings.purpose} copied from the job (dedicated only).")
    notes.append("Region left unset; the control plane places the trainer.")

    tokenizer_model = _resolve_tokenizer(inputs, settings, base_model)
    renderer_name = _resolve_renderer(settings, tokenizer_model, warnings)
    max_seq_len = _resolve_max_seq_len(inputs, settings, base_model)

    def config_for(target: Mode) -> Any:
        return _build_recipe_config(
            settings,
            inputs,
            mode=target,
            base_model=base_model_name,
            lora_rank=lora_rank,
            tokenizer_model=tokenizer_model,
            renderer_name=renderer_name,
            max_seq_len=max_seq_len,
            eval_dataset=eval_dataset,
            eval_auto_carveout=eval_auto_carveout,
            wandb=wandb,
        )

    config = config_for(mode)
    fallback = None
    if mode is Mode.SERVERLESS and inputs.mode == "auto":
        fallback = config_for(Mode.DEDICATED)
    if settings.generation_eval and mode is Mode.DEDICATED:
        warnings.append("generation_eval runs only on serverless SFT; it is off for this dedicated run.")
    if mode is Mode.DEDICATED:
        quota = (
            "Dedicated training uses your account's GPU quota. By default an account with a payment method "
            "gets 32 each of H200, B200 and B300; yours can differ. If creating the trainer fails with HTTP 403 "
            "tier_required, add a payment method or credits; if it fails with quota_exceeded, request more GPU "
            "quota."
        )
        if settings.method is Method.DPO and lora_rank <= 0:
            quota += " Full-parameter DPO also starts a separate reference trainer, so it needs quota for two."
        warnings.append(quota)

    return MigrationPlan(
        job=job_ref,
        settings=settings,
        mode=mode,
        mode_reasons=mode_reasons,
        config=config,
        warnings=warnings,
        notes=notes,
        fallback_config=fallback,
    )


def _build_recipe_config(
    settings: ManagedSettings,
    inputs: MigrationInputs,
    *,
    mode: Mode,
    base_model: str,
    lora_rank: int,
    tokenizer_model: str,
    renderer_name: str,
    max_seq_len: int,
    eval_dataset: str,
    eval_auto_carveout: bool,
    wandb: WandBConfig,
) -> Any:
    """Build the recipe ``Config``, mirroring the managed V2 orchestrator.

    Fields the job left at 0 keep the recipe default, which is what managed V2
    used. The trainer region is never set.
    """
    serverless = mode is Mode.SERVERLESS
    trainer = TrainerConfig(
        job_id=inputs.trainer_job_id or None,
        use_reservation=settings.use_reservation,
        reservation_target=settings.reservation_target or None,
        purpose=None if serverless else (settings.purpose or None),
    )
    common: dict[str, Any] = {
        "log_path": inputs.log_path,
        "base_model": base_model,
        "dataset": inputs.dataset_path,
        "tokenizer_model": tokenizer_model,
        # A renderer chosen from the job's renderer repo is final, like managed's materialized one.
        "renderer_name": renderer_name,
        "renderer_name_is_resolved": bool(renderer_name),
        "thinking_trace_history_mode": thinking_mode_from_managed(settings.thinking_trace_history_mode),
        "lora_rank": lora_rank,
        "max_seq_len": max_seq_len,
        "output_model_id": inputs.output_model_id,
        "dcp_save_interval": MANAGED_DCP_SAVE_INTERVAL,
        "trainer": trainer,
        "wandb": wandb,
        "init_from_checkpoint": inputs.init_from_checkpoint or None,
        # A resumed run's checkpoint already holds the warm-started weights.
        "warm_start_from_adapter": None if inputs.init_from_checkpoint else (settings.warm_start_from or None),
    }
    if settings.learning_rate > 0:
        common["learning_rate"] = settings.learning_rate
    if settings.epochs > 0:
        common["epochs"] = settings.epochs
    if settings.batch_size_samples > 0:
        common["batch_size"] = settings.batch_size_samples
    if settings.weight_decay > 0:
        common["weight_decay"] = settings.weight_decay
    if settings.lora_alpha > 0:
        common["lora_alpha"] = settings.lora_alpha
    if settings.sampler_save_interval > 0:
        common["sampler_save_interval"] = settings.sampler_save_interval
    scheduler = lr_scheduler_from_managed(settings.lr_scheduler, settings.warmup_steps)
    if scheduler is not None:
        common["lr_scheduler"] = scheduler

    if settings.method is Method.ORPO:
        if settings.orpo_lambda is not None:
            common["orpo_lambda"] = float(settings.orpo_lambda)
        return orpo_loop.Config(**common)

    # SFT and DPO: stream gs:// data, keep held-out evaluation, and save the
    # managed LoRA checkpoint schedule (full-parameter jobs had only the step cadence).
    common["dataset_loading_mode"] = "stream" if inputs.dataset_path.startswith("gs://") else "stage"
    common["evaluation_dataset"] = eval_dataset
    common["eval_auto_carveout"] = eval_auto_carveout
    common["checkpoint_interval_minutes"] = MANAGED_CHECKPOINT_INTERVAL_MINUTES if lora_rank > 0 else 0.0
    common["serverless"] = serverless

    if settings.method is Method.SFT:
        common["generation_eval"] = settings.generation_eval and serverless
        return sft_loop.Config(**common)

    if settings.beta is not None:
        common["beta"] = float(settings.beta)
    if settings.ref_cache_concurrency is not None:
        common["ref_cache_concurrency"] = int(settings.ref_cache_concurrency)
    if settings.ref_cache_batch_size is not None:
        common["ref_cache_batch_size"] = int(settings.ref_cache_batch_size)
    if serverless:
        # A resumed serverless DPO run must score against the same frozen reference.
        common["runner"] = RunnerConfig(reference_checkpoint_file=fileio.join(inputs.log_path, "reference_checkpoint.json"))
    return dpo_loop.Config(**common)


# ---------------------------------------------------------------------------
# API reads
# ---------------------------------------------------------------------------


class FireworksReader:
    """The few control-plane reads the migration needs."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = Fireworks(**kwargs)

    def key_account_id(self) -> str:
        """Resolve the key's account the way the training SDK does."""
        client = FireworksClient(api_key=self._api_key, base_url=self._base_url or "https://api.fireworks.ai")
        try:
            return client.account_id
        except httpx.HTTPStatusError as exc:
            raise MigrationError(
                f"FIREWORKS_API_KEY was rejected (HTTP {exc.response.status_code}). Check the key and try again."
            ) from exc
        except ValueError as exc:
            if "multiple accounts" in str(exc):
                raise MigrationError(
                    f"{exc} Create an API key in the job's account only, then set FIREWORKS_API_KEY to it."
                ) from exc
            raise MigrationError(str(exc)) from exc
        finally:
            client.close()

    def get_job(self, ref: JobRef) -> dict[str, Any]:
        jobs = self._client.supervised_fine_tuning_jobs if ref.is_sft else self._client.dpo_jobs
        return jobs.get(ref.job_id, account_id=ref.account_id).to_dict(mode="json")

    def get_model(self, name: str) -> dict[str, Any] | None:
        """Return the model, or ``None`` if it doesn't exist."""
        account_id, model_id = _split_resource(name, "models")
        try:
            return self._client.models.get(model_id, account_id=account_id).to_dict(mode="json")
        except NotFoundError:
            return None

    def get_dataset(self, name: str) -> dict[str, Any] | None:
        account_id, dataset_id = _split_resource(name, "datasets")
        try:
            return self._client.datasets.get(dataset_id, account_id=account_id).to_dict(mode="json")
        except NotFoundError:
            return None


def _split_resource(name: str, collection: str) -> tuple[str, str]:
    parts = name.split("/")
    if len(parts) != 4 or parts[0] != "accounts" or parts[2] != collection:
        raise MigrationError(f"Expected accounts/<account>/{collection}/<id>; got {name!r}.")
    return parts[1], parts[3]


def _is_cmek(resource: dict[str, Any] | None) -> bool:
    return bool(resource) and resource.get("encryptionState") == CMEK_ENCRYPTION_STATE


def read_plan(reader: FireworksReader, inputs: MigrationInputs, job_name: str) -> MigrationPlan:
    job_ref = parse_job_name(job_name)
    key_account_id = reader.key_account_id()
    if key_account_id != job_ref.account_id:
        raise MigrationError(
            f"FIREWORKS_API_KEY belongs to account {key_account_id!r}, but the job is in "
            f"{job_ref.account_id!r}. Use a key created in {job_ref.account_id!r}."
        )
    job = reader.get_job(job_ref)

    # DPO jobs carry no encryption state; their output model and dataset do.
    if not job_ref.is_sft and not job.get("encryptionState"):
        tc = job.get("trainingConfig") or {}
        for name, getter in ((tc.get("outputModel"), reader.get_model), (job.get("dataset"), reader.get_dataset)):
            resource = getter(name) if name else None
            if resource is not None:
                if _is_cmek(resource):
                    job = {**job, "encryptionState": CMEK_ENCRYPTION_STATE}
                break

    settings = settings_from_sft_job(job) if job_ref.is_sft else settings_from_dpo_job(job)
    adapter = None
    if settings.warm_start_from:
        adapter = reader.get_model(settings.warm_start_from)
        if adapter is None:
            raise MigrationError(f"The warm-start adapter {settings.warm_start_from} no longer exists.")
    base_model_name = settings.base_model or ((adapter or {}).get("peftDetails") or {}).get("baseModel") or ""
    base_model = reader.get_model(base_model_name) if base_model_name else None
    if base_model is None:
        raise MigrationError(f"The base model {base_model_name or '(unset)'} of {job_ref.name} no longer exists.")
    # A copy of a Fireworks model in the customer's account may lack the Hugging Face URL;
    # the catalog model with the same ID has it.
    needs_tokenizer = not (inputs.tokenizer_model or base_model.get("huggingFaceUrl"))
    model_account, model_id = _split_resource(base_model_name, "models")
    catalog_note = ""
    if needs_tokenizer and model_account != FIREWORKS_ACCOUNT:
        catalog_name = f"accounts/{FIREWORKS_ACCOUNT}/models/{model_id}"
        catalog_model = reader.get_model(catalog_name)
        if catalog_model and catalog_model.get("huggingFaceUrl") and _same_architecture(base_model, catalog_model):
            base_model = {**base_model, "huggingFaceUrl": catalog_model["huggingFaceUrl"]}
            catalog_note = (
                f"{base_model_name} has no Hugging Face URL; using the tokenizer of {catalog_name}, which has "
                "the same ID and model type. Pass --tokenizer-model if that's wrong."
            )
    plan = build_plan(job_ref, job, inputs, base_model=base_model, adapter=adapter)
    if catalog_note and not inputs.tokenizer_model:
        plan.warnings.append(catalog_note)
    return plan


def _same_architecture(model: dict[str, Any], other: dict[str, Any]) -> bool:
    model_type = (model.get("baseModelDetails") or {}).get("modelType")
    return bool(model_type) and model_type == (other.get("baseModelDetails") or {}).get("modelType")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def describe_plan(plan: MigrationPlan) -> str:
    lines = [
        f"Managed job:  {plan.job.name} ({plan.settings.method.value})",
        f"Recipe:       training.recipes.{plan.recipe}",
        f"Mode:         {plan.mode.value}",
    ]
    lines += [f"  - {reason}" for reason in plan.mode_reasons]
    if plan.fallback_config is not None:
        lines.append("  - falls back to dedicated if serverless is unavailable")
    for warning in plan.warnings:
        lines.append(f"WARNING: {warning}")
    for note in plan.notes:
        lines.append(f"Note: {note}")
    lines.append("Config:")
    lines.append(json.dumps(_jsonable(plan.config), indent=2, sort_keys=True))
    return "\n".join(lines)


def _recipe_main(plan: MigrationPlan) -> Callable[[Any], Any]:
    module = {Method.SFT: sft_loop, Method.DPO: dpo_loop, Method.ORPO: orpo_loop}[plan.settings.method]
    return module.main


def serverless_unavailable(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(text in message for text in SERVERLESS_UNAVAILABLE_ERRORS)


def _training_started(log_path: str) -> bool:
    try:
        return _latest_checkpoint(log_path) is not None
    except Exception:  # noqa: BLE001 - unreadable state: assume a checkpoint may exist
        return True


class ResumeRecorder(logging.Handler):
    """Notes the trainer job (dedicated) or run (serverless) that a resume needs, from the run's logs."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.trainer_job_id: str | None = None
        self.kept_trainer_job_id: str | None = None
        self.serverless_run_id: str | None = None

    def reset(self) -> None:
        self.trainer_job_id = None
        self.kept_trainer_job_id = None
        self.serverless_run_id = None

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        trainer = _TRAINER_JOB_LOG_RE.search(message)
        if trainer and self.trainer_job_id is None:
            # The policy trainer starts first; a DPO reference trainer comes later.
            self.trainer_job_id = trainer["job"]
        kept = _KEPT_TRAINER_LOG_RE.search(message)
        if kept:
            self.kept_trainer_job_id = kept["job"].rsplit("/", 1)[-1].rstrip(".,;")
        run = _SERVERLESS_RUN_LOG_RE.search(message)
        if run:
            self.serverless_run_id = run["run"]


def _latest_checkpoint(log_path: str) -> str | None:
    """Newest resumable checkpoint recorded in the run's ``dataloader.json``.

    Reads local and ``gs://`` log paths, as the recipes write both. Raises if
    the file exists but can't be read.
    """
    raw = fileio.read_text(fileio.join(log_path, DATALOADER_BASE_NAME))
    if not raw:
        return None
    try:
        manifest = json.loads(raw)
    except ValueError:
        return None
    checkpoints = manifest.get("checkpoints", manifest) if isinstance(manifest, dict) else {}
    names = [name for name in checkpoints if name != "schema_version"]
    return names[-1] if names else None


def _kept_trainer_hint(plan: MigrationPlan, job_id: str, checkpoint: str | None) -> str:
    log_path = plan.config.log_path
    delete = f"firectl rlor-trainer-job delete {job_id}"
    billing = f"The kept trainer bills until it stops after 60 idle minutes; delete it with `{delete}` if you won't resume."
    if checkpoint is not None:
        hint = f"To resume, rerun this command with --log-path {log_path} --trainer-job-id {job_id}."
        if plan.settings.method is Method.ORPO:
            hint += " ORPO restores the step but reads the dataset from its first row again."
        return f"{hint} {billing}"
    if plan.config.lora_rank > 0:
        return (
            f"No checkpoint was saved, so there is nothing to resume. To start over on the kept trainer, rerun "
            f"this command with --log-path {log_path} --trainer-job-id {job_id}. {billing}"
        )
    return (
        f"No checkpoint was saved, and the kept full-parameter trainer still holds the stopped run's weights. "
        f"Delete it with `{delete}`, then rerun this command without --trainer-job-id."
    )


def resume_hint(plan: MigrationPlan, recorder: ResumeRecorder) -> str | None:
    log_path = plan.config.log_path
    try:
        checkpoint = _latest_checkpoint(log_path)
    except Exception as exc:  # noqa: BLE001 - don't mask the error that stopped the run
        return f"Couldn't read {fileio.join(log_path, DATALOADER_BASE_NAME)} to find the newest checkpoint ({exc})."
    if recorder.kept_trainer_job_id:
        return _kept_trainer_hint(plan, recorder.kept_trainer_job_id, checkpoint)
    if checkpoint is None:
        return "No resumable checkpoint was saved, so rerunning the command starts training over."
    if recorder.serverless_run_id:
        reference = f"{plan.job.account_id}/{recorder.serverless_run_id}/{checkpoint}"
        return f"To resume, rerun this command with --log-path {log_path} --init-from-checkpoint {reference}"
    if recorder.trainer_job_id:
        # The recipe deleted its dedicated trainer on exit (it logged no kept trainer). A
        # checkpoint from another trainer restores weights and optimizer state, not the
        # dataset position.
        reference = f"{recorder.trainer_job_id}:{checkpoint}"
        return (
            f"The dedicated trainer was deleted when the run stopped, so the run can't resume where it left off. "
            f"To start a new run from checkpoint {reference} (weights and optimizer state), rerun this command "
            f"with --log-path {log_path} --init-from-checkpoint {reference}. The dataset restarts from its first row."
        )
    return None


def run_plan(plan: MigrationPlan) -> Any:
    """Run the recipe; with ``--mode auto``, retry on dedicated if serverless can't start.

    If the run stops early, print how to continue it.
    """
    main_fn = _recipe_main(plan)
    recorder = ResumeRecorder()
    root_logger = logging.getLogger()
    root_logger.addHandler(recorder)
    try:
        try:
            return main_fn(plan.config)
        except Exception as exc:
            if (
                plan.fallback_config is None
                or not serverless_unavailable(exc)
                or _training_started(plan.config.log_path)
            ):
                raise
            logger.warning("Serverless training is unavailable (%s). Retrying on a dedicated trainer.", exc)
            recorder.reset()
            return main_fn(plan.fallback_config)
    except BaseException as exc:
        if SERVERLESS_CONCURRENCY_ERROR in str(exc):
            print(
                "\nYour account is at its limit of concurrent serverless training runs for this model. A run that "
                "just finished keeps its session open for a while. Wait and rerun, or use --mode dedicated.",
                file=sys.stderr,
            )
        hint = resume_hint(plan, recorder)
        if hint:
            print(f"\n{hint}", file=sys.stderr)
        raise
    finally:
        root_logger.removeHandler(recorder)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rerun a managed SFT, DPO or ORPO job on the Training API.")
    p.add_argument("--job", required=True, help="accounts/<account>/supervisedFineTuningJobs/<id> or accounts/<account>/dpoJobs/<id>")
    p.add_argument("--dataset-path", default="", help="Your copy of the job's training data: a local JSONL file or directory, or a gs:// path.")
    p.add_argument("--output-model-id", required=True, help="ID for the new model. The managed job's output model already exists.")
    p.add_argument("--mode", choices=["auto", "serverless", "dedicated"], default="auto")
    p.add_argument("--eval-dataset-path", default="", help="Local copy of the job's evaluation dataset, if it had one.")
    p.add_argument("--tokenizer-model", default="", help="Hugging Face tokenizer, e.g. Qwen/Qwen3-8B. Default: from the job and base model.")
    p.add_argument("--max-seq-len", type=int, default=0, help="Default: the job's max_context_length, else the model's training context length.")
    p.add_argument("--log-path", default="", help="Checkpoint and log directory. Reuse it to continue a run. Default: ./migrated_runs/<job-id>-<time>.")
    p.add_argument("--init-from-checkpoint", default="", help="Continue from a checkpoint an earlier run of this command saved (same --log-path).")
    p.add_argument("--trainer-job-id", default="", help="Reattach to the dedicated trainer an interrupted run kept, to resume it (same --log-path).")
    p.add_argument("--dry-run", action="store_true", help="Print the config and chosen mode, then exit.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Any:
    load_dotenv()
    args = parse_args(argv)
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        raise SystemExit("Set FIREWORKS_API_KEY to an API key created in the job's account.")
    try:
        job_ref = parse_job_name(args.job)
        log_path = args.log_path or os.path.join("migrated_runs", f"{job_ref.job_id}-{time.strftime('%Y%m%d-%H%M%S')}")
        inputs = MigrationInputs(
            dataset_path=args.dataset_path,
            output_model_id=args.output_model_id,
            log_path=log_path,
            mode=args.mode,
            eval_dataset_path=args.eval_dataset_path,
            tokenizer_model=args.tokenizer_model,
            max_seq_len=args.max_seq_len,
            init_from_checkpoint=args.init_from_checkpoint,
            trainer_job_id=args.trainer_job_id,
        )
        reader = FireworksReader(api_key, os.environ.get("FIREWORKS_BASE_URL") or None)
        plan = read_plan(reader, inputs, args.job)
    except MigrationError as exc:
        raise SystemExit(f"error: {exc}") from exc
    except APIStatusError as exc:
        raise SystemExit(f"error: reading {args.job} failed with HTTP {exc.status_code}: {exc.message}") from exc

    print(describe_plan(plan))
    if args.dry_run:
        print("\nDry run: nothing was created. Drop --dry-run to start training.")
        return plan
    print(f"\nLogs and checkpoints: {log_path}. If the run stops early, the tool prints how to continue it.")
    return run_plan(plan)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
