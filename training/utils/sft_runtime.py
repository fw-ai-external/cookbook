"""Caller-owned execution hooks for a serverless SFT invocation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fireworks.training.sdk import FireworksClient, FiretitanServiceClient


class SFTRuntime(Protocol):
    """Keep resource ownership and durable recovery outside the training loop.

    The caller owns both clients and their credentials. SFT never closes these
    clients or deletes their session. Hooks may raise to stop new work; already
    submitted operations can finish. No hook is serialized into recipe config.
    """

    service: FiretitanServiceClient
    checkpoint_client: FireworksClient

    def check_active(self) -> None:
        """Raise if cancellation or lost ownership forbids new remote work."""
        ...

    def checkpoint_saved(self, reference: str, cursor_path: Path) -> None:
        """Durably publish the exact DCP reference with its committed cursor.

        Called before a subsequent sampler export. Failure aborts training;
        retaining model weights without this cursor is not a recovery point.
        """
        ...
