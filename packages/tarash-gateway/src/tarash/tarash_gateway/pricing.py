"""Pricing module for generation cost tracking.

Provides a static pricing table and cost resolution logic for all
supported providers and models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tarash.tarash_gateway.models import GenerationCost


@dataclass(frozen=True)
class PricingEntry:
    """A single entry in the pricing table.

    Maps a (provider, model) pair to a per-unit cost.
    """

    usd_per_unit: float
    """Cost in USD per unit of the given unit type."""
    unit: str
    """Unit type (e.g. ``"seconds"``, ``"images"``, ``"characters"``)."""


PRICING_TABLE: dict[tuple[str, str], PricingEntry] = {
    # Fal Video
    ("fal", "fal-ai/minimax"): PricingEntry(usd_per_unit=0.50, unit="videos"),
    ("fal", "fal-ai/pixverse/swap"): PricingEntry(usd_per_unit=0.05, unit="videos"),
    ("fal", "fal-ai/veo3"): PricingEntry(usd_per_unit=0.40, unit="seconds"),
    ("fal", "fal-ai/veo3.1"): PricingEntry(usd_per_unit=0.40, unit="seconds"),
    ("fal", "wan/v2.6"): PricingEntry(usd_per_unit=0.10, unit="seconds"),
    ("fal", "fal-ai/pixverse/lipsync"): PricingEntry(usd_per_unit=0.04, unit="seconds"),
    ("fal", "fal-ai/bytedance/omnihuman"): PricingEntry(
        usd_per_unit=0.14, unit="seconds"
    ),
    ("fal", "fal-ai/sync-lipsync"): PricingEntry(usd_per_unit=0.70, unit="minutes"),
    ("fal", "fal-ai/kling-video/o1"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    ("fal", "fal-ai/kling-video/v2.6"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    ("fal", "fal-ai/kling-video/v3"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    ("fal", "fal-ai/kling-video/o3"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    ("fal", "fal-ai/sora-2"): PricingEntry(
        usd_per_unit=0.00007, unit="compute_seconds"
    ),
    ("fal", "fal-ai/bytedance/seedance"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    ("fal", "fal-ai/pixverse/v5.5"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    ("fal", "fal-ai/pixverse/v5"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    ("fal", "fal-ai/wan-25-preview/v3"): PricingEntry(
        usd_per_unit=0.00007, unit="compute_seconds"
    ),
    ("fal", "fal-ai/wan/v2.2-14b/animate"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    ("fal", "fal-ai/wan/v2.2-a14b"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    # Fal Image
    ("fal", "fal-ai/flux/dev"): PricingEntry(usd_per_unit=0.025, unit="megapixels"),
    ("fal", "fal-ai/flux/schnell"): PricingEntry(usd_per_unit=0.003, unit="megapixels"),
    ("fal", "fal-ai/flux-2"): PricingEntry(usd_per_unit=0.012, unit="megapixels"),
    ("fal", "fal-ai/recraft-v3"): PricingEntry(usd_per_unit=0.04, unit="images"),
    ("fal", "xai/grok-imagine-image"): PricingEntry(usd_per_unit=0.02, unit="images"),
    ("fal", "fal-ai/nano-banana-2"): PricingEntry(usd_per_unit=0.08, unit="images"),
    ("fal", "fal-ai/flux/pro"): PricingEntry(
        usd_per_unit=0.00167, unit="compute_seconds"
    ),
    ("fal", "fal-ai/recraft"): PricingEntry(
        usd_per_unit=0.00007, unit="compute_seconds"
    ),
    ("fal", "fal-ai/ideogram"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    ("fal", "fal-ai/reve"): PricingEntry(usd_per_unit=0.00017, unit="compute_seconds"),
    ("fal", "fal-ai/bytedance/seedream/v5/lite"): PricingEntry(
        usd_per_unit=0.00017, unit="compute_seconds"
    ),
    # OpenAI — gpt-image-1/1.5/mini use token-based cost (OPENAI_IMAGE_TOKEN_RATES).
    # dall-e-3/dall-e-2 are flat per-image pricing. Rates below are the base tier
    # (standard quality, 1024x1024). Actual cost varies by quality and resolution:
    #   dall-e-3: $0.04 (std 1024) / $0.08 (std 1792 or HD 1024) / $0.12 (HD 1792)
    #   dall-e-2: $0.016 (256) / $0.018 (512) / $0.02 (1024)
    ("openai", "dall-e-3"): PricingEntry(usd_per_unit=0.04, unit="images"),
    ("openai", "dall-e-2"): PricingEntry(usd_per_unit=0.02, unit="images"),
    ("openai", "sora"): PricingEntry(usd_per_unit=0.10, unit="seconds"),
    # Google
    ("google", "gemini-2.5-flash-image"): PricingEntry(
        usd_per_unit=0.039, unit="images"
    ),
    ("google", "gemini-2.5-flash-image-preview"): PricingEntry(
        usd_per_unit=0.039, unit="images"
    ),
    ("google", "gemini-3-pro-image-preview"): PricingEntry(
        usd_per_unit=0.134, unit="images"
    ),
    ("google", "imagen-3.0-generate-001"): PricingEntry(
        usd_per_unit=0.04, unit="images"
    ),
    ("google", "imagen-3.0-generate-002"): PricingEntry(
        usd_per_unit=0.04, unit="images"
    ),
    ("google", "imagen-3.0-fast-generate-001"): PricingEntry(
        usd_per_unit=0.02, unit="images"
    ),
    ("google", "veo-3.0-generate-preview"): PricingEntry(
        usd_per_unit=0.40, unit="seconds"
    ),
    # Runway
    ("runway", "gen4.5"): PricingEntry(usd_per_unit=0.12, unit="seconds"),
    ("runway", "gen4_turbo"): PricingEntry(usd_per_unit=0.05, unit="seconds"),
    ("runway", "gen4_aleph"): PricingEntry(usd_per_unit=0.15, unit="seconds"),
    ("runway", "gen3a_turbo"): PricingEntry(usd_per_unit=0.05, unit="seconds"),
    # Stability
    ("stability", "sd3.5-large"): PricingEntry(usd_per_unit=0.065, unit="images"),
    ("stability", "sd3.5-medium"): PricingEntry(usd_per_unit=0.035, unit="images"),
    ("stability", "stable-image-ultra"): PricingEntry(usd_per_unit=0.08, unit="images"),
    # xAI
    ("xai", "grok-imagine-image"): PricingEntry(usd_per_unit=0.02, unit="images"),
    ("xai", "grok-2-image"): PricingEntry(usd_per_unit=0.02, unit="images"),
    ("xai", "grok-imagine-video"): PricingEntry(usd_per_unit=0.05, unit="seconds"),
    # ElevenLabs
    ("elevenlabs", "eleven_multilingual_v2"): PricingEntry(
        usd_per_unit=0.00024, unit="characters"
    ),
    ("elevenlabs", "eleven_turbo_v2"): PricingEntry(
        usd_per_unit=0.00024, unit="characters"
    ),
    # Cartesia
    ("cartesia", "sonic"): PricingEntry(usd_per_unit=0.000011, unit="characters"),
    # Sarvam
    ("sarvam", "bulbul:v2"): PricingEntry(usd_per_unit=0.000018, unit="characters"),
    ("sarvam", "bulbul:v3"): PricingEntry(usd_per_unit=0.000036, unit="characters"),
    ("sarvam", "bulbul:v3-beta"): PricingEntry(
        usd_per_unit=0.000036, unit="characters"
    ),
    # Hume
    ("hume", "octave"): PricingEntry(usd_per_unit=0.00015, unit="characters"),
    ("hume", "octave-v2"): PricingEntry(usd_per_unit=0.0000076, unit="characters"),
}


def lookup_pricing_table(
    provider: str, model: str, quantity: float
) -> GenerationCost | None:
    """Look up a (provider, model) pair in the pricing table and compute cost.

    Args:
        provider: Provider identifier (e.g. ``"fal"``).
        model: Model name (e.g. ``"fal-ai/veo3"``).
        quantity: The quantity to multiply by the per-unit price.

    Returns:
        A ``GenerationCost`` with computed ``amount_usd``, or ``None`` if the
        pair is not found in the table.
    """
    entry = PRICING_TABLE.get((provider, model))
    if entry is None:
        return None
    return GenerationCost(
        amount_usd=entry.usd_per_unit * quantity,
        raw_amount=quantity,
        raw_unit=entry.unit,
    )


def resolve_cost(
    provider: str,
    model: str,
    api_cost: GenerationCost | None,
    quantity: float,
) -> GenerationCost | None:
    """Resolve the cost for a generation request.

    If an API-reported cost is provided, it is returned as-is (even if
    ``amount_usd`` is ``None``). Otherwise, the pricing table is consulted.

    Args:
        provider: Provider identifier.
        model: Model name.
        api_cost: Cost reported by the provider API, or ``None``.
        quantity: Quantity to use for pricing table lookup.

    Returns:
        A ``GenerationCost``, or ``None`` if no cost information is available.
    """
    if api_cost is not None:
        return api_cost
    return lookup_pricing_table(provider, model, quantity)


# ==================== Token-Based Pricing (OpenAI Image) ====================

# Per-token rates (USD per token) for OpenAI image models.
# Source: https://openai.com/api/pricing/
# Each model has separate rates for text and image tokens.
OPENAI_IMAGE_TOKEN_RATES: dict[str, dict[str, float]] = {
    "gpt-image-1": {
        "text_input": 5.00 / 1_000_000,
        "image_input": 10.00 / 1_000_000,
        "cached_text_input": 1.25 / 1_000_000,
        "cached_image_input": 2.50 / 1_000_000,
        "image_output": 40.00 / 1_000_000,
    },
    "gpt-image-1.5": {
        "text_input": 5.00 / 1_000_000,
        "image_input": 8.00 / 1_000_000,
        "cached_text_input": 1.25 / 1_000_000,
        "cached_image_input": 2.00 / 1_000_000,
        "image_output": 32.00 / 1_000_000,
        "text_output": 10.00 / 1_000_000,
    },
    "gpt-image-1-mini": {
        "text_input": 2.00 / 1_000_000,
        "image_input": 2.50 / 1_000_000,
        "cached_text_input": 0.20 / 1_000_000,
        "cached_image_input": 0.25 / 1_000_000,
        "image_output": 8.00 / 1_000_000,
    },
}


def _safe_int(val: Any) -> int:
    """Safely extract an integer from a value, returning 0 for non-numeric types."""
    if isinstance(val, (int, float)):
        return int(val)
    return 0


def compute_openai_image_cost(model: str, usage: Any) -> GenerationCost | None:
    """Compute cost from OpenAI image API usage token breakdown.

    Uses separate per-token rates for text input, image input, cached input,
    and output tokens. Returns ``None`` if the model has no known rates or
    usage data is not available.

    Args:
        model: Model name (e.g., ``"gpt-image-1"``, ``"gpt-image-1.5"``).
        usage: OpenAI usage object with ``input_tokens``, ``output_tokens``,
               ``input_tokens_details``, and ``output_tokens_details``.

    Returns:
        A ``GenerationCost`` with exact token-based cost, or ``None``.
    """
    rates = OPENAI_IMAGE_TOKEN_RATES.get(model)
    if rates is None or usage is None:
        return None

    # Validate that usage has real numeric data (not a MagicMock)
    total_tokens = _safe_int(getattr(usage, "total_tokens", 0))
    if total_tokens == 0:
        return None

    total_cost = 0.0

    # Input token breakdown
    input_details = getattr(usage, "input_tokens_details", None)
    has_input_details = input_details is not None and isinstance(
        getattr(input_details, "text_tokens", None), (int, float)
    )

    if has_input_details:
        text_input = _safe_int(getattr(input_details, "text_tokens", 0))
        image_input = _safe_int(getattr(input_details, "image_tokens", 0))
        cached_tokens = _safe_int(getattr(input_details, "cached_tokens", 0))

        # Cached tokens reduce cost — subtract from uncached counts
        # The API reports cached_tokens as a total; distribute proportionally
        uncached_text = max(0, text_input - cached_tokens)
        cached_text = min(text_input, cached_tokens)
        remaining_cached = max(0, cached_tokens - cached_text)
        uncached_image = max(0, image_input - remaining_cached)
        cached_image = min(image_input, remaining_cached)

        total_cost += uncached_text * rates["text_input"]
        total_cost += cached_text * rates.get("cached_text_input", rates["text_input"])
        total_cost += uncached_image * rates["image_input"]
        total_cost += cached_image * rates.get(
            "cached_image_input", rates["image_input"]
        )
    else:
        # No detailed breakdown — use image_input rate for all input tokens
        input_tokens = _safe_int(getattr(usage, "input_tokens", 0))
        total_cost += input_tokens * rates["image_input"]

    # Output tokens
    output_tokens = _safe_int(getattr(usage, "output_tokens", 0))
    output_details = getattr(usage, "output_tokens_details", None)
    has_output_details = output_details is not None and isinstance(
        getattr(output_details, "image_tokens", None), (int, float)
    )

    if has_output_details:
        image_out = _safe_int(getattr(output_details, "image_tokens", 0))
        text_out = _safe_int(getattr(output_details, "text_tokens", 0))
        total_cost += image_out * rates["image_output"]
        total_cost += text_out * rates.get("text_output", rates["image_output"])
    else:
        total_cost += output_tokens * rates["image_output"]

    return GenerationCost(
        amount_usd=total_cost,
        raw_amount=float(total_tokens),
        raw_unit="tokens",
    )
