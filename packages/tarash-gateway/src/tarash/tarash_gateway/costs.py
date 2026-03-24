"""Cost estimation for AI media generation models."""

from dataclasses import dataclass
from typing import TypeAlias

from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    VideoGenerationConfig,
)

GenerationConfig: TypeAlias = VideoGenerationConfig | ImageGenerationConfig


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost range for a generation request in USD."""

    min_usd: float
    max_usd: float
    model: str
    provider: str

    @property
    def avg_usd(self) -> float:
        return (self.min_usd + self.max_usd) / 2


@dataclass(frozen=True)
class _ModelCost:
    min_usd: float
    max_usd: float


# Provider -> model prefix -> cost per generation
_COST_TABLE: dict[str, dict[str, _ModelCost]] = {
    "fal": {
        # Video models
        "fal-ai/minimax": _ModelCost(0.21, 0.21),
        "fal-ai/kling-video/o1": _ModelCost(0.10, 0.40),
        "fal-ai/kling-video/v2.6": _ModelCost(0.10, 0.40),
        "fal-ai/kling-video/v3": _ModelCost(0.10, 0.40),
        "fal-ai/kling-video/o3": _ModelCost(0.10, 0.40),
        "fal-ai/veo3.1": _ModelCost(0.25, 0.50),
        "fal-ai/veo3": _ModelCost(0.25, 0.50),
        "fal-ai/sora-2": _ModelCost(0.20, 0.60),
        "wan/v2.6/": _ModelCost(0.04, 0.12),
        "fal-ai/wan-25-preview/": _ModelCost(0.04, 0.12),
        "fal-ai/wan/v2.2-14b/animate/": _ModelCost(0.04, 0.08),
        "fal-ai/wan/v2.2-a14b/": _ModelCost(0.04, 0.08),
        "fal-ai/bytedance/seedance": _ModelCost(0.10, 0.30),
        "fal-ai/pixverse/v5.5": _ModelCost(0.10, 0.30),
        "fal-ai/pixverse/v5": _ModelCost(0.10, 0.30),
        "fal-ai/pixverse/lipsync": _ModelCost(0.10, 0.20),
        "fal-ai/bytedance/omnihuman": _ModelCost(0.10, 0.30),
        "fal-ai/sync-lipsync": _ModelCost(0.10, 0.20),
        # Image models
        "fal-ai/flux": _ModelCost(0.01, 0.05),
        "fal-ai/flux-2": _ModelCost(0.03, 0.06),
        "fal-ai/z-image-turbo": _ModelCost(0.005, 0.01),
        "fal-ai/recraft-v3": _ModelCost(0.02, 0.04),
        "fal-ai/recraft": _ModelCost(0.02, 0.04),
        "fal-ai/ideogram": _ModelCost(0.04, 0.08),
        "fal-ai/reve": _ModelCost(0.02, 0.05),
        "xai/grok-imagine-image": _ModelCost(0.03, 0.07),
        "fal-ai/bytedance/seedream/v5/lite": _ModelCost(0.02, 0.04),
        "fal-ai/nano-banana-2": _ModelCost(0.01, 0.03),
    },
    "openai": {
        # Image models
        "gpt-image-1.5": _ModelCost(0.02, 0.19),
        "dall-e-3": _ModelCost(0.04, 0.12),
        "dall-e-2": _ModelCost(0.016, 0.02),
        # Video models
        "sora": _ModelCost(0.20, 0.60),
    },
    "google": {
        # Image models
        "gemini-2.5-flash-image": _ModelCost(0.01, 0.04),
        "gemini-2.5-flash-image-preview": _ModelCost(0.01, 0.04),
        "gemini-3-pro-image-preview": _ModelCost(0.02, 0.06),
        "imagen-3": _ModelCost(0.02, 0.04),
        "imagen-3.0-generate-001": _ModelCost(0.02, 0.04),
        "imagen-3.0-generate-002": _ModelCost(0.02, 0.04),
        "imagen-3.0-fast-generate-001": _ModelCost(0.01, 0.03),
        # Video models
        "veo-3": _ModelCost(0.25, 0.50),
        "veo-2": _ModelCost(0.15, 0.35),
    },
}


def _lookup_cost(provider: str, model: str) -> _ModelCost | None:
    provider_costs = _COST_TABLE.get(provider)
    if provider_costs is None:
        return None

    if model in provider_costs:
        return provider_costs[model]

    # Prefix matching: find the longest prefix that matches
    best_match: str | None = None
    for prefix in provider_costs:
        if model.startswith(prefix):
            if best_match is None or len(prefix) > len(best_match):
                best_match = prefix

    if best_match is not None:
        return provider_costs[best_match]

    return None


def estimate_cost(config: GenerationConfig) -> CostEstimate | None:
    """Estimate the cost range for a generation request.

    Uses a lookup table of known provider/model combinations to return
    a min/max USD cost estimate. Returns None if the provider+model
    combination is not in the cost table.

    Args:
        config: A VideoGenerationConfig or ImageGenerationConfig.

    Returns:
        CostEstimate with min/max USD, or None if unknown model.
    """
    cost = _lookup_cost(config.provider, config.model)
    if cost is None:
        return None

    return CostEstimate(
        min_usd=cost.min_usd,
        max_usd=cost.max_usd,
        model=config.model,
        provider=config.provider,
    )
