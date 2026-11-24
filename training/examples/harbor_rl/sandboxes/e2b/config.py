"""Configure Harbor's E2B environment and HTTP transport."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

from training.utils.errors import UserConfigError

logger = logging.getLogger(__name__)


def configure_environment(
    harbor: Any,
    environment: dict[str, Any],
    *,
    uses_compose: bool,
    frozen_manifest: str | Path | None,
) -> None:
    # Harbor checks this too. Fail before creating an unsupported template.
    if uses_compose or environment.get("extra_docker_compose"):
        raise ValueError("Harbor E2B does not support Docker Compose tasks")
    environment["type"] = harbor.EnvironmentType.E2B
    if frozen_manifest is not None:
        environment["type"] = None
        environment["import_path"] = "training.examples.harbor_rl.sandboxes.e2b.frozen:FrozenE2BEnvironment"
        environment["kwargs"] = {**environment.get("kwargs", {}), "frozen_manifest": str(frozen_manifest)}


def configure_http_transport() -> str:
    version = os.environ.get("E2B_HTTP_VERSION") or "1.1"
    if version not in {"1.1", "2"}:
        raise UserConfigError("E2B_HTTP_VERSION must be '1.1' or '2'")
    try:
        connection_config = importlib.import_module("e2b.connection_config")
    except ImportError as error:
        raise UserConfigError("E2B HTTP selection requires a supported e2b SDK (use the validated e2b==2.55.0)") from error
    resolver = getattr(connection_config, "resolve_http_version", None)
    if not callable(resolver):
        raise UserConfigError("Installed e2b SDK does not support E2B_HTTP_VERSION; use the validated e2b==2.55.0")
    os.environ["E2B_HTTP_VERSION"] = version
    if resolver() != version:
        raise UserConfigError("Installed e2b SDK did not honor E2B_HTTP_VERSION")
    logger.info("Harbor E2B HTTP transport version=%s", version)
    return version
