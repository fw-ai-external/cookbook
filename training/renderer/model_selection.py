"""Exact model identifiers for default renderer selection; never path heuristics."""

import json
from pathlib import Path

from training.model_defaults import TOKENIZER_MODELS
from training.renderer._model_info import (
    get_deepseek_info,
    get_gpt_oss_info,
    get_llama_info,
    get_moonshot_info,
    get_nvidia_info,
    get_qwen_info,
)
from training.renderer.thinking_trace import get_thinking_trace_model_capability

_CATALOG_IDS = {alias.casefold(): canonical.casefold() for alias, canonical in TOKENIZER_MODELS.items()}
_DEFAULTS = {
    f"{attributes.organization}/{name}".casefold(): attributes.recommended_renderers[0]
    for registry in (
        get_deepseek_info(),
        get_gpt_oss_info(),
        get_llama_info(),
        get_moonshot_info(),
        get_nvidia_info(),
        get_qwen_info(),
    )
    for name, attributes in registry.items()
}
# Preserve legacy default semantics independently of selectable thinking-history modes.
_LEGACY_FAMILY_DEFAULTS = {
    "deepseek-v4": "deepseek_v4",
    "glm5.1": "glm5",
    "glm5.2": "glm_moe_dsa",
    "glm5.3": "glm53",
    "glm5.3-flash": "glm53_flash",
    "qwen3.5": "qwen3_5",
    "qwen3.6": "qwen3_6",
    "qwen3.8": "qwen3_8",
    "kimi-k2.5": "kimi_k25",
    "kimi-k2.6": "kimi_k25",
    "kimi-k2.7-code": "kimi_k27_code",
    "nemotron3": "nemotron3",
    "nemotron3-ultra": "nemotron3_ultra",
}
# Literal identifiers only. Each group is a reviewed renderer contract, not a
# namespace/prefix rule. Similar-looking forks, paths and unknown variants fail.
_EXACT_GROUPS = {
    "beam": """
        reflection/beam-1.0-501b-a23b reflection/beam-1.0-501b-a23b-nvfp4
        accounts/fireworks/models/beam-1p0-501b-a23b-nvfp4
    """,
    "kimi_k3": "moonshotai/kimi-k3 accounts/fireworks/models/kimi-k3",
    "kimi_k31": "moonshotai/kimi-k3.1-pro accounts/fireworks/models/kimi-k3p1-pro",
    "glm5_release": "zai-org/glm-5 zai-org/glm-5-fp8 accounts/fireworks/models/glm-5",
    "glm53": "zai-org/glm-5.3-bf16",
    "glm53_flash": "zai-org/glm-5.3-flash-bf16",
    "deepseek_v41": """
        deepseek-ai/deepseek-v4.1-flash deepseek-ai/deepseek-v4.1-flash-1017
        accounts/fireworks/models/deepseek-v41-flash accounts/fireworks/models/deepseek-v4p1-flash
    """,
    "deepseek_v4": "deepseek-ai/deepseek-v4-pro-0813",
    "qwen2_5": "qwen/qwen2.5-32b-instruct",
    "qwen3": "qwen/qwen3-0.6b qwen/qwen3-1.7b qwen/qwen3-4b qwen/qwen3-14b",
    "qwen3_instruct": "qwen/qwen3-235b-a22b-instruct-2507-fp8",
    "qwen3_5": "qwen/qwen3.5-0.8b qwen/qwen3.5-122b-a10b qwen/qwen3.5-2b",
    "qwen3_vl_instruct": """
        qwen/qwen3-vl-8b-instruct qwen/qwen3-vl-32b-instruct
        qwen/qwen3-vl-30b-a3b-thinking qwen/qwen3-vl-235b-a22b-thinking
    """,
    "gemma4": """
        google/gemma-4-12b-it google/gemma-4-27b-it google/gemma-4-26b-a4b-it
        google/gemma-4-26b-a4b-it-assistant google/gemma-4-31b-it google/gemma-4-31b-it-assistant
        google/gemma-4-e4b google/gemma-4-e4b-it nvidia/gemma-4-31b-it-nvfp4
    """,
    "minimax_m2": "minimaxai/minimax-m2 minimaxai/minimax-m2.1 minimaxai/minimax-m2.5 minimaxai/minimax-m2.7",
    "minimax_m3": "minimaxai/minimax-m3",
    "nemotron3": """
        nvidia/nvidia-nemotron-3-super-120b nvidia/nvidia-nemotron-h-8b
        nvidia/nvidia-nemotron-3-super-120b-a12b-fp8 nvidia/nvidia-nemotron-3-super-120b-a12b-nvfp4
        nvidia/nvidia-nemotron-nano-9b-v2 nvidia/nvidia-nemotron-nano-12b-v2
        nvidia/nemotron-nano-v2-12b-vl nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-bf16
        nvidia/llama-3.1-nemotron-70b-instruct
    """,
    "nemotron35_vl": """
        nvidia/nvidia-nemotron-3.5-super-vl nvidia/nvidia-nemotron-3.5-super-vl-120b-a12b-bf16
        nvidia/nvidia-nemotron-3.5-super-vl-120b-a12b-nvfp4
        nvidia/nvidia-nemotron-3.5-super-vl-120b-a12b-base-bf16
        accounts/fireworks/models/nemotron-3p5-super-vl
    """,
    "muse_glimmer": "meta-models/muse-glimmer-30b accounts/fireworks/models/muse_glimmer-30b",
    "muse_spark": "meta-models/muse-spark-1.2-816b-a42b-nvfp4-open",
    "glm_moe_dsa": "amd/glm-5.2-mxfp4",
    "mistral": """
        mistralai/devstral-2-123b-instruct-2512 mistralai/devstral-small-2-24b-instruct-2512
        mistralai/devstral-small-2505 mistralai/ministral-3-14b-instruct-2512
        mistralai/ministral-3-3b-instruct-2512 mistralai/ministral-3-8b-instruct-2512
        mistralai/mistral-7b-v0.1 mistralai/mistral-7b-instruct-v0.1 mistralai/mistral-7b-v0.2
        mistralai/mistral-7b-instruct-v0.2 mistralai/mistral-7b-instruct-v0.3
        mistralai/mistral-large-3-675b-instruct-2512 mistralai/mistral-medium-3.5-128b
        mistralai/mistral-nemo-base-2407 mistralai/mistral-nemo-instruct-2407
        mistralai/mistral-small-24b-instruct-2501 mistralai/mistral-small-3.1-24b-instruct-2503
        mistralai/mixtral-8x22b-v0.1 mistralai/mixtral-8x22b-instruct-v0.1
        mistralai/mixtral-8x7b-v0.1 mistralai/mixtral-8x7b-instruct-v0.1
    """,
}
for _renderer, _identifiers in _EXACT_GROUPS.items():
    for _identifier in _identifiers.split():
        _DEFAULTS[_identifier] = _renderer


def resolve_exact_model_renderer(model: str) -> str | None:
    """Resolve a complete registered ID, never a basename or substring."""
    normalized = model.strip().casefold()
    # This is URL syntax normalization, not model-name inference.
    normalized = normalized.removeprefix("https://huggingface.co/")
    canonical = _CATALOG_IDS.get(normalized, normalized)
    capability = get_thinking_trace_model_capability(canonical)
    if capability is not None:
        return _LEGACY_FAMILY_DEFAULTS.get(capability.canonical_family)
    return _DEFAULTS.get(canonical)


def is_exact_mistral_tokenizer_model(model: str) -> bool:
    """Identify only the listed Mistral tokenizer IDs, not directory names."""
    normalized = model.strip().casefold()
    return _CATALOG_IDS.get(normalized, normalized) in _EXACT_GROUPS["mistral"].split()


def local_declared_renderer(model: str) -> str | None:
    """Use only unambiguous artifact declarations; shared types need an override."""
    root = Path(model)
    declarations: dict[str, dict] = {}
    for filename in ("config.json", "tokenizer_config.json"):
        try:
            parsed = json.loads((root / filename).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(parsed, dict):
            declarations[filename] = parsed
    model_type = declarations.get("config.json", {}).get("model_type")
    if model_type is None:
        model_type = declarations.get("tokenizer_config.json", {}).get("kimi_native_profile")
    if not isinstance(model_type, str):
        return None
    return {
        "kimi_k3": "kimi_k3",
        "kimi_k31": "kimi_k31",
        "deepseek_v41": "deepseek_v41",
        "deepseek_v41_text": "deepseek_v41",
        "deepseek_v4.1": "deepseek_v41",
        "deepseek_v4.1_text": "deepseek_v41",
        "minimax_m2": "minimax_m2",
        "minimax_m3_vl": "minimax_m3",
    }.get(model_type)
