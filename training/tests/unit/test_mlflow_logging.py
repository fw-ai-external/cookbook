"""Cookbook metrics reach the SDK's MLflow transport without external HTTP."""

from __future__ import annotations

import asyncio
import builtins
import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fireworks.training.sdk import _mlflow

from training.utils import logging as logs
from training.utils.config import WandBConfig


@pytest.fixture(autouse=True)
def _isolated_logger(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(logs, "_mlflow_client", None)
    monkeypatch.setattr(logs, "_mlflow_initialized", False)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=None))
    monkeypatch.setattr(_mlflow, "client_from_env", lambda **_kwargs: None)
    yield
    logs._finish_mlflow("FINISHED")


@pytest.mark.parametrize("failed", [False, True])
def test_metrics_use_one_run_correct_steps_and_terminal_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failed: bool
) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/ingest/experiments":
            return httpx.Response(200, json={"experiment_id": "test-experiment"})
        if request.url.path.endswith("/runs/create"):
            return httpx.Response(200, json={"run": {"info": {"run_id": "test-run"}}})
        return httpx.Response(200, json={})

    def factory(**kwargs: str) -> _mlflow.MlflowIngestClient:
        assert kwargs["project"] == "test-project"
        return _mlflow.MlflowIngestClient(
            "https://mlflow-ingest.invalid",
            "test-only",
            kwargs["project"],
            kwargs["run_name"],
            transport=httpx.MockTransport(handle),
        )

    monkeypatch.setattr(_mlflow, "client_from_env", factory)
    wb = WandBConfig(project="test-project", run_name="test-run")
    assert logs.setup_wandb(wb, {}) is False  # W&B off does not disable MLflow.
    logs.setup_wandb(wb, {})
    record = {"train/step": 3, "train/sample_train_max_abs_diff": 0.3927, "train/sample_train_mismatch_frac": 29 / 64}
    path = tmp_path / "metrics.jsonl"
    logs.log_metrics(record, step=99, metrics_file=str(path))
    logs.wandb_log({"train/step": 4, "train/sample_train_max_abs_diff": 0.0}, step=100)
    logs.log_run_resources(SimpleNamespace(trainer_job_id="test-trainer"))
    try:
        with logs.finish_metrics_on_exit():
            if failed:
                raise ValueError("training failed")
    except ValueError:
        pass
    logs.wandb_finish()

    points = [m for url, body in requests if url.endswith("/runs/log-batch") for m in body["metrics"]]
    assert [(m["value"], m["step"]) for m in points if m["key"] == "train/sample_train_max_abs_diff"] == [
        (0.3927, 3),
        (0.0, 4),
    ]
    assert [(m["value"], m["step"]) for m in points if m["key"] == "train/sample_train_mismatch_frac"] == [
        (29 / 64, 3)
    ]
    assert json.loads(path.read_text()) == {**record, "step": 99}
    assert sum(url.endswith("/runs/create") for url, _body in requests) == 1
    assert requests[-1][1]["status"] == ("FAILED" if failed else "FINISHED")
    assert any(
        t["key"] == "fireworks.trainer_job_id"
        for url, body in requests
        if url.endswith("/runs/log-batch")
        for t in body["tags"]
    )


def test_configured_startup_failure_is_not_silently_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**_kwargs: str) -> None:
        raise RuntimeError("private authentication details")

    monkeypatch.setattr(_mlflow, "client_from_env", fail)
    with pytest.raises(RuntimeError, match="MLflow initialization failed") as error:
        logs.setup_wandb(WandBConfig(), {})
    assert "private authentication" not in str(error.value)


@pytest.mark.parametrize("recipe", ["rl_loop", "async_rl_loop"])
@pytest.mark.parametrize("error_type,status", [(RuntimeError, "FAILED"), (asyncio.CancelledError, "KILLED")])
def test_failed_recipe_retry_uses_separate_mlflow_runs(
    monkeypatch: pytest.MonkeyPatch, recipe: str, error_type: type[BaseException], status: str
) -> None:
    module = importlib.import_module(f"training.recipes.{recipe}")
    requests = []
    clients = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("/experiments"):
            return httpx.Response(200, json={"experiment_id": "test-experiment"})
        if request.url.path.endswith("/runs/create"):
            return httpx.Response(200, json={"run": {"info": {"run_id": body["run_name"]}}})
        return httpx.Response(200, json={})

    def factory(**kwargs: str) -> _mlflow.MlflowIngestClient:
        client = _mlflow.MlflowIngestClient(
            "https://mlflow-ingest.invalid",
            "test-only",
            kwargs["project"],
            kwargs["run_name"],
            transport=httpx.MockTransport(handle),
        )
        clients.append(client)
        return client

    def fail_tokenizer(*_args: object, **_kwargs: object) -> None:
        logs.log_metrics({"train/sample_train_max_abs_diff": len(clients)}, step=1)
        logs.log_run_resources(SimpleNamespace(trainer_job_id=f"trainer-{len(clients)}"))
        raise error_type("stop before provisioning")

    monkeypatch.setattr(_mlflow, "client_from_env", factory)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-only")
    monkeypatch.setattr(module, "validate_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_deployment_tokenizer", fail_tokenizer)
    for run_name in ("attempt-a", "attempt-b"):
        cfg = module.Config(
            log_path="/tmp/mlflow-retry-test",
            deployment=module.DeployConfig(tokenizer_model="test-tokenizer"),
            wandb=WandBConfig(project="test-project", run_name=run_name),
        )
        with pytest.raises(error_type, match="stop before provisioning"):
            module.main(cfg, rows=[], rollout_fn_factory=lambda _setup: None)
        assert logs._mlflow_client is None
        assert logs._mlflow_initialized is False

    assert len(clients) == 2
    updates = [body for url, body in requests if url.endswith("/runs/update")]
    assert [(body["run_id"], body["status"]) for body in updates] == [
        ("attempt-a", status),
        ("attempt-b", status),
    ]
    points = [
        (body["run_id"], metric["value"])
        for url, body in requests
        if url.endswith("/runs/log-batch")
        for metric in body["metrics"]
        if metric["key"] == "train/sample_train_max_abs_diff"
    ]
    assert points == [("attempt-a", 1), ("attempt-b", 2)]
    tags = [
        (body["run_id"], tag["value"])
        for url, body in requests
        if url.endswith("/runs/log-batch")
        for tag in body["tags"]
        if tag["key"] == "fireworks.trainer_job_id"
    ]
    assert tags == [("attempt-a", "trainer-1"), ("attempt-b", "trainer-2")]


@pytest.mark.parametrize("configured", [False, True])
def test_older_sdk_still_works_unless_mlflow_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured: bool
) -> None:
    original = builtins.__import__

    def without_mlflow(name: str, *args: object, **kwargs: object) -> object:
        if name == "fireworks.training.sdk" and "_mlflow" in args[2]:
            raise ImportError("old SDK")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mlflow)
    for name in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_TOKEN",
        "FIREWORKS_REQUIRE_MLFLOW",
        "FIREWORKS_SKIP_MLFLOW_REASON",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FIREWORKS_MLFLOW_CONFIG_DIR", str(tmp_path))
    if configured:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow-ingest.invalid")
        with pytest.raises(RuntimeError, match="SDK with MLflow ingest support"):
            logs.setup_wandb(WandBConfig(), {})
    else:
        assert logs.setup_wandb(WandBConfig(), {}) is False
