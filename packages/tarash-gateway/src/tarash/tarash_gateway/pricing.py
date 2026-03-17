"""Pricing data for generation cost tracking.

Provides a static pricing table for all supported providers and models.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class PricingEntry:
    """A single entry in the pricing table.

    Maps a (provider, model) pair to a per-unit cost.
    """

    usd_per_unit: Decimal
    """Cost in USD per unit of the given unit type."""
    unit: str
    """Unit type (e.g. ``"seconds"``, ``"images"``, ``"characters"``)."""


PRICING_TABLE: dict[tuple[str, str], PricingEntry] = {
    # Fal Video
    ("fal", "fal-ai/minimax"): PricingEntry(
        usd_per_unit=Decimal("0.50"), unit="videos"
    ),
    ("fal", "fal-ai/pixverse/swap"): PricingEntry(
        usd_per_unit=Decimal("0.05"), unit="videos"
    ),
    ("fal", "fal-ai/veo3"): PricingEntry(usd_per_unit=Decimal("0.40"), unit="seconds"),
    ("fal", "fal-ai/veo3.1"): PricingEntry(
        usd_per_unit=Decimal("0.40"), unit="seconds"
    ),
    ("fal", "wan/v2.6"): PricingEntry(usd_per_unit=Decimal("0.10"), unit="seconds"),
    ("fal", "fal-ai/pixverse/lipsync"): PricingEntry(
        usd_per_unit=Decimal("0.04"), unit="seconds"
    ),
    ("fal", "fal-ai/bytedance/omnihuman"): PricingEntry(
        usd_per_unit=Decimal("0.14"), unit="seconds"
    ),
    ("fal", "fal-ai/sync-lipsync"): PricingEntry(
        usd_per_unit=Decimal("0.70"), unit="minutes"
    ),
    ("fal", "fal-ai/kling-video/o1"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/kling-video/v2.6"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/kling-video/v3"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/kling-video/o3"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/sora-2"): PricingEntry(
        usd_per_unit=Decimal("0.00007"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/bytedance/seedance"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/pixverse/v5.5"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/pixverse/v5"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/wan-25-preview/v3"): PricingEntry(
        usd_per_unit=Decimal("0.00007"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/wan/v2.2-14b/animate"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/wan/v2.2-a14b"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    # Fal Image
    ("fal", "fal-ai/flux/dev"): PricingEntry(
        usd_per_unit=Decimal("0.025"), unit="megapixels"
    ),
    ("fal", "fal-ai/flux/schnell"): PricingEntry(
        usd_per_unit=Decimal("0.003"), unit="megapixels"
    ),
    ("fal", "fal-ai/flux-2"): PricingEntry(
        usd_per_unit=Decimal("0.012"), unit="megapixels"
    ),
    ("fal", "fal-ai/recraft-v3"): PricingEntry(
        usd_per_unit=Decimal("0.04"), unit="images"
    ),
    ("fal", "xai/grok-imagine-image"): PricingEntry(
        usd_per_unit=Decimal("0.02"), unit="images"
    ),
    ("fal", "fal-ai/nano-banana-2"): PricingEntry(
        usd_per_unit=Decimal("0.08"), unit="images"
    ),
    ("fal", "fal-ai/flux/pro"): PricingEntry(
        usd_per_unit=Decimal("0.00167"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/recraft"): PricingEntry(
        usd_per_unit=Decimal("0.00007"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/ideogram"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/reve"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    ("fal", "fal-ai/bytedance/seedream/v5/lite"): PricingEntry(
        usd_per_unit=Decimal("0.00017"), unit="compute_seconds"
    ),
    # OpenAI — gpt-image-1/1.5/mini use token-based cost (OPENAI_IMAGE_TOKEN_RATES).
    # dall-e-3/dall-e-2 are flat per-image pricing. Rates below are the base tier
    # (standard quality, 1024x1024). Actual cost varies by quality and resolution:
    #   dall-e-3: $0.04 (std 1024) / $0.08 (std 1792 or HD 1024) / $0.12 (HD 1792)
    #   dall-e-2: $0.016 (256) / $0.018 (512) / $0.02 (1024)
    ("openai", "dall-e-3"): PricingEntry(usd_per_unit=Decimal("0.04"), unit="images"),
    ("openai", "dall-e-2"): PricingEntry(usd_per_unit=Decimal("0.02"), unit="images"),
    ("openai", "sora"): PricingEntry(usd_per_unit=Decimal("0.10"), unit="seconds"),
    # Google
    ("google", "gemini-2.5-flash-image"): PricingEntry(
        usd_per_unit=Decimal("0.039"), unit="images"
    ),
    ("google", "gemini-2.5-flash-image-preview"): PricingEntry(
        usd_per_unit=Decimal("0.039"), unit="images"
    ),
    ("google", "gemini-3-pro-image-preview"): PricingEntry(
        usd_per_unit=Decimal("0.134"), unit="images"
    ),
    ("google", "imagen-3.0-generate-001"): PricingEntry(
        usd_per_unit=Decimal("0.04"), unit="images"
    ),
    ("google", "imagen-3.0-generate-002"): PricingEntry(
        usd_per_unit=Decimal("0.04"), unit="images"
    ),
    ("google", "imagen-3.0-fast-generate-001"): PricingEntry(
        usd_per_unit=Decimal("0.02"), unit="images"
    ),
    ("google", "veo-3.0-generate-preview"): PricingEntry(
        usd_per_unit=Decimal("0.40"), unit="seconds"
    ),
    # Runway
    ("runway", "gen4.5"): PricingEntry(usd_per_unit=Decimal("0.12"), unit="seconds"),
    ("runway", "gen4_turbo"): PricingEntry(
        usd_per_unit=Decimal("0.05"), unit="seconds"
    ),
    ("runway", "gen4_aleph"): PricingEntry(
        usd_per_unit=Decimal("0.15"), unit="seconds"
    ),
    ("runway", "gen3a_turbo"): PricingEntry(
        usd_per_unit=Decimal("0.05"), unit="seconds"
    ),
    # Stability
    ("stability", "sd3.5-large"): PricingEntry(
        usd_per_unit=Decimal("0.065"), unit="images"
    ),
    ("stability", "sd3.5-medium"): PricingEntry(
        usd_per_unit=Decimal("0.035"), unit="images"
    ),
    ("stability", "stable-image-ultra"): PricingEntry(
        usd_per_unit=Decimal("0.08"), unit="images"
    ),
    # xAI
    ("xai", "grok-imagine-image"): PricingEntry(
        usd_per_unit=Decimal("0.02"), unit="images"
    ),
    ("xai", "grok-2-image"): PricingEntry(usd_per_unit=Decimal("0.02"), unit="images"),
    ("xai", "grok-imagine-video"): PricingEntry(
        usd_per_unit=Decimal("0.05"), unit="seconds"
    ),
    # ElevenLabs
    ("elevenlabs", "eleven_multilingual_v2"): PricingEntry(
        usd_per_unit=Decimal("0.00024"), unit="characters"
    ),
    ("elevenlabs", "eleven_turbo_v2"): PricingEntry(
        usd_per_unit=Decimal("0.00024"), unit="characters"
    ),
    # Cartesia
    ("cartesia", "sonic"): PricingEntry(
        usd_per_unit=Decimal("0.000011"), unit="characters"
    ),
    # Sarvam
    ("sarvam", "bulbul:v2"): PricingEntry(
        usd_per_unit=Decimal("0.000018"), unit="characters"
    ),
    ("sarvam", "bulbul:v3"): PricingEntry(
        usd_per_unit=Decimal("0.000036"), unit="characters"
    ),
    ("sarvam", "bulbul:v3-beta"): PricingEntry(
        usd_per_unit=Decimal("0.000036"), unit="characters"
    ),
    # Hume
    ("hume", "octave"): PricingEntry(
        usd_per_unit=Decimal("0.00015"), unit="characters"
    ),
    ("hume", "octave-v2"): PricingEntry(
        usd_per_unit=Decimal("0.0000076"), unit="characters"
    ),
}


# ==================== Token-Based Pricing (OpenAI Image) ====================

# Per-token rates (USD per token) for OpenAI image models.
# Source: https://openai.com/api/pricing/
# Each model has separate rates for text and image tokens.
OPENAI_IMAGE_TOKEN_RATES: dict[str, dict[str, Decimal]] = {
    "gpt-image-1": {
        "text_input": Decimal("5.00") / 1_000_000,
        "image_input": Decimal("10.00") / 1_000_000,
        "cached_text_input": Decimal("1.25") / 1_000_000,
        "cached_image_input": Decimal("2.50") / 1_000_000,
        "image_output": Decimal("40.00") / 1_000_000,
    },
    "gpt-image-1.5": {
        "text_input": Decimal("5.00") / 1_000_000,
        "image_input": Decimal("8.00") / 1_000_000,
        "cached_text_input": Decimal("1.25") / 1_000_000,
        "cached_image_input": Decimal("2.00") / 1_000_000,
        "image_output": Decimal("32.00") / 1_000_000,
        "text_output": Decimal("10.00") / 1_000_000,
    },
    "gpt-image-1-mini": {
        "text_input": Decimal("2.00") / 1_000_000,
        "image_input": Decimal("2.50") / 1_000_000,
        "cached_text_input": Decimal("0.20") / 1_000_000,
        "cached_image_input": Decimal("0.25") / 1_000_000,
        "image_output": Decimal("8.00") / 1_000_000,
    },
}
