"""Preserve the trainer's R3 binding when launching Harbor sidecars."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace

from fireworks.training.sdk.sampling import DeploymentSampler

from training.examples.harbor_rl.tito.sidecar import (
    TITOSidecarBundle,
    TITOSidecarLaunchSpec,
    TITOSidecarSettings,
    build_launch_spec,
)

from training.recipes.experiment.async_rl_loop_serverless import (
    Config,
    ServerlessSampler,
    _rollout_setup,
)
from training.utils.rl.rollout.setup import RolloutSetup


def sampling_client(model: str = "snapshot", **routing: object) -> SimpleNamespace:
    return SimpleNamespace(
        deployment_sampler=SimpleNamespace(
            model=model, base_url="https://example.invalid", tokenizer=None,
            **routing,
        ),
        close=lambda: None,
    )


def rollout_setup(sampler: ServerlessSampler) -> RolloutSetup:
    return _rollout_setup(
        Config(), tokenizer=None, sampler=sampler, api_key="unused-test-value",
        router_replay_enabled=True,
    )


def launch_spec(setup: RolloutSetup) -> TITOSidecarLaunchSpec:
    return build_launch_spec(
        setup, TITOSidecarBundle(Path("unused.zip"), "test-digest"),
        TITOSidecarSettings(bundle_root="unused", renderer_name="test"),
        call_classifier="all_policy", metadata={},
    )


class ServerlessR3Test(unittest.IsolatedAsyncioTestCase):
    async def test_sidecar_preserves_parquet_binding_across_snapshot_replacement(self) -> None:
        sampler = ServerlessSampler(sampling_client(
            model="snapshot-a", routing_matrix_format="parquet_v1", r3_store_id="store-a",
        ))
        setup = rollout_setup(sampler)
        first = launch_spec(setup)
        self.assertEqual(first.model, "snapshot-a")
        self.assertEqual(first.routing_matrix_format, "parquet_v1")
        self.assertEqual(first.r3_store_id, "store-a")
        await sampler.replace(sampling_client(
            model="snapshot-b", routing_matrix_format="parquet_v1", r3_store_id="store-b",
        ))
        second = launch_spec(setup)
        self.assertEqual(second.model, "snapshot-b")
        self.assertEqual(first.model, "snapshot-a")
        self.assertEqual(second.routing_matrix_format, "parquet_v1")
        self.assertEqual(second.r3_store_id, "store-b")
        self.assertEqual(first.r3_store_id, "store-a")
        await sampler.aclose()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            launch_spec(setup)

    async def test_sidecar_binding_stays_consistent_while_replacement_drains(self) -> None:
        old = sampling_client(
            model="snapshot-a", routing_matrix_format="base64_inline", r3_store_id="store-a",
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def sample(*args: object, **kwargs: object) -> list[object]:
            started.set()
            await release.wait()
            return []

        old.deployment_sampler.sample_with_prompt_tokens = sample
        sampler = ServerlessSampler(old)
        setup = rollout_setup(sampler)
        sampling = asyncio.create_task(sampler.sample_with_prompt_tokens([1]))
        await started.wait()
        replacing = asyncio.create_task(sampler.replace(sampling_client(
            model="snapshot-b", routing_matrix_format="parquet_v1", r3_store_id="store-b",
        )))
        try:
            await asyncio.sleep(0)
            self.assertFalse(replacing.done())
            self.assertEqual(sampler.model, "snapshot-b")
            self.assertEqual(setup.model, "snapshot-a")
            spec = launch_spec(setup)
            self.assertEqual(spec.model, "snapshot-b")
            self.assertEqual(spec.routing_matrix_format, "parquet_v1")
            self.assertEqual(spec.r3_store_id, "store-b")
        finally:
            release.set()
            await sampling
            await replacing
            await sampler.aclose()

    async def test_sidecar_preserves_store_before_lazy_inference_negotiation(self) -> None:
        client = sampling_client()
        client.deployment_sampler = DeploymentSampler(
            inference_url="https://example.invalid", model="snapshot",
            api_key="unused-test-value", tokenizer=None,
        )
        client.deployment_sampler.negotiate_routing_matrix_format("parquet_v1", "store-a")
        sampler = ServerlessSampler(client)
        spec = launch_spec(rollout_setup(sampler))
        # Negotiation with inference happens inside the sidecar on its first R3 call.
        self.assertEqual(spec.routing_matrix_format, "base64_inline")
        self.assertEqual(spec.r3_store_id, "store-a")
        await sampler.aclose()

    async def test_legacy_sampler_keeps_inline_fallback(self) -> None:
        sampler = ServerlessSampler(sampling_client())
        spec = launch_spec(rollout_setup(sampler))
        self.assertEqual(spec.routing_matrix_format, "base64_inline")
        self.assertIsNone(spec.r3_store_id)
        setup = rollout_setup(sampler)
        setup.sampler = SimpleNamespace()
        self.assertEqual(launch_spec(setup).model, "snapshot")
        await sampler.aclose()


if __name__ == "__main__":
    unittest.main()
