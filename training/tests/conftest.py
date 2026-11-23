from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_trainer_cancel_grace_period(monkeypatch):
    monkeypatch.setenv("FW_TRAINER_CANCEL_GRACE_PERIOD_S", "0")


@pytest.fixture(autouse=True)
def _isolate_mlflow(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    # CPU tests must not discover the developer's real MLflow configuration.
    if "e2e" not in request.keywords:
        monkeypatch.setenv("FIREWORKS_SKIP_MLFLOW_REASON", "cookbook unit test")


@pytest.fixture(params=[0, 16], ids=["full", "lora"], scope="session")
def port_lora_rank(request) -> int:
    """Precision track parameter for live SDK-managed e2e tests."""
    return int(request.param)
