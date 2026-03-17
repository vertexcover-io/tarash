"""Pricing module for generation cost tracking.

Provides a static pricing table and cost resolution logic for all
supported providers and models.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    # OpenAI
    ("openai", "gpt-image-1"): PricingEntry(usd_per_unit=0.042, unit="images"),
    ("openai", "gpt-image-1.5"): PricingEntry(usd_per_unit=0.04, unit="images"),
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
    ("sarvam", "bulbul-v2"): PricingEntry(usd_per_unit=0.000018, unit="characters"),
    ("sarvam", "bulbul-v3"): PricingEntry(usd_per_unit=0.000036, unit="characters"),
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
