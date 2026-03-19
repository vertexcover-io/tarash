"""Unit tests for pricing data and GenerationCost class methods.

Tests PricingEntry, PRICING_TABLE (data-only), and GenerationCost factory
methods: from_pricing_table(), from_token_usage(), from_credits().
"""

import dataclasses
from decimal import Decimal
from types import SimpleNamespace

import pytest

from tarash.tarash_gateway.models import GenerationCost
from tarash.tarash_gateway.pricing import (
    OPENAI_IMAGE_TOKEN_RATES,
    PRICING_TABLE,
    PricingEntry,
)


# ==================== PricingEntry ====================


def test_pricing_entry_creation():
    """PricingEntry can be created with Decimal usd_per_unit and unit."""
    entry = PricingEntry(usd_per_unit=Decimal("0.40"), unit="seconds")
    assert entry.usd_per_unit == Decimal("0.40")
    assert entry.unit == "seconds"


def test_pricing_entry_is_frozen():
    """PricingEntry is immutable (frozen dataclass)."""
    entry = PricingEntry(usd_per_unit=Decimal("0.50"), unit="videos")
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.usd_per_unit = Decimal("1.0")  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.unit = "images"  # type: ignore[misc]


def test_pricing_entry_is_dataclass():
    """PricingEntry is a dataclass."""
    assert dataclasses.is_dataclass(PricingEntry)


def test_pricing_entry_fields():
    """PricingEntry has exactly two fields with correct types."""
    fields = {f.name: f.type for f in dataclasses.fields(PricingEntry)}
    assert fields == {
        "usd_per_unit": "Decimal",
        "unit": "str",
    }


# ==================== PRICING_TABLE ====================


def test_pricing_table_is_dict():
    """PRICING_TABLE is a dict keyed by (provider, model) tuples."""
    assert isinstance(PRICING_TABLE, dict)
    for key, value in PRICING_TABLE.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], str)
        assert isinstance(key[1], str)
        assert isinstance(value, PricingEntry)


def test_pricing_table_all_values_are_decimal():
    """All usd_per_unit values in PRICING_TABLE are Decimal instances."""
    for key, entry in PRICING_TABLE.items():
        assert isinstance(entry.usd_per_unit, Decimal), (
            f"Entry {key} has usd_per_unit type {type(entry.usd_per_unit)}, expected Decimal"
        )


def test_pricing_table_spot_check_fal_veo3():
    """Spot-check: Fal veo3 entry."""
    entry = PRICING_TABLE[("fal", "fal-ai/veo3")]
    assert entry.usd_per_unit == Decimal("0.40")
    assert entry.unit == "seconds"


def test_pricing_table_spot_check_fal_minimax():
    """Spot-check: Fal minimax entry."""
    entry = PRICING_TABLE[("fal", "fal-ai/minimax")]
    assert entry.usd_per_unit == Decimal("0.50")
    assert entry.unit == "videos"


def test_pricing_table_spot_check_openai_sora():
    """Spot-check: OpenAI sora entry."""
    entry = PRICING_TABLE[("openai", "sora")]
    assert entry.usd_per_unit == Decimal("0.10")
    assert entry.unit == "seconds"


def test_pricing_table_spot_check_runway_gen4_turbo():
    """Spot-check: Runway gen4_turbo entry."""
    entry = PRICING_TABLE[("runway", "gen4_turbo")]
    assert entry.usd_per_unit == Decimal("0.05")
    assert entry.unit == "seconds"


def test_pricing_table_spot_check_elevenlabs():
    """Spot-check: ElevenLabs eleven_multilingual_v2 entry."""
    entry = PRICING_TABLE[("elevenlabs", "eleven_multilingual_v2")]
    assert entry.usd_per_unit == Decimal("0.00024")
    assert entry.unit == "characters"


def test_pricing_table_spot_check_cartesia():
    """Spot-check: Cartesia sonic entry."""
    entry = PRICING_TABLE[("cartesia", "sonic")]
    assert entry.usd_per_unit == Decimal("0.000011")
    assert entry.unit == "characters"


def test_pricing_table_spot_check_hume_octave():
    """Spot-check: Hume octave entry."""
    entry = PRICING_TABLE[("hume", "octave")]
    assert entry.usd_per_unit == Decimal("0.00015")
    assert entry.unit == "characters"


def test_pricing_table_spot_check_sarvam_bulbul_v2():
    """Spot-check: Sarvam bulbul:v2 entry."""
    entry = PRICING_TABLE[("sarvam", "bulbul:v2")]
    assert entry.usd_per_unit == Decimal("0.000018")
    assert entry.unit == "characters"


def test_pricing_table_spot_check_google_veo():
    """Spot-check: Google veo entry."""
    entry = PRICING_TABLE[("google", "veo-3.0-generate-preview")]
    assert entry.usd_per_unit == Decimal("0.40")
    assert entry.unit == "seconds"


def test_pricing_table_spot_check_stability():
    """Spot-check: Stability sd3.5-large entry."""
    entry = PRICING_TABLE[("stability", "sd3.5-large")]
    assert entry.usd_per_unit == Decimal("0.065")
    assert entry.unit == "images"


def test_pricing_table_spot_check_xai():
    """Spot-check: xAI grok-imagine-image entry."""
    entry = PRICING_TABLE[("xai", "grok-imagine-image")]
    assert entry.usd_per_unit == Decimal("0.02")
    assert entry.unit == "images"


def test_pricing_table_spot_check_fal_kling():
    """Spot-check: Fal kling-video/o1 compute_seconds entry."""
    entry = PRICING_TABLE[("fal", "fal-ai/kling-video/o1")]
    assert entry.usd_per_unit == Decimal("0.00017")
    assert entry.unit == "compute_seconds"


def test_pricing_table_no_replicate_entries():
    """No Replicate entries exist in the pricing table."""
    replicate_entries = [k for k in PRICING_TABLE if k[0] == "replicate"]
    assert replicate_entries == []


def test_pricing_table_has_expected_count():
    """PRICING_TABLE has all entries from the design doc."""
    # Count: 19 fal video + 11 fal image + 3 openai (dall-e-3, dall-e-2, sora)
    # + 7 google + 4 runway + 3 stability + 3 xai + 2 elevenlabs + 1 cartesia
    # + 3 sarvam + 2 hume = 58
    assert len(PRICING_TABLE) == 58


# ==================== OPENAI_IMAGE_TOKEN_RATES ====================


def test_openai_token_rates_all_decimal():
    """All token rate values are Decimal instances."""
    for model, rates in OPENAI_IMAGE_TOKEN_RATES.items():
        for rate_key, rate_val in rates.items():
            assert isinstance(rate_val, Decimal), (
                f"Rate {model}/{rate_key} is {type(rate_val)}, expected Decimal"
            )


def test_openai_token_rates_has_expected_models():
    """OPENAI_IMAGE_TOKEN_RATES has the three expected models."""
    assert set(OPENAI_IMAGE_TOKEN_RATES.keys()) == {
        "gpt-image-1",
        "gpt-image-1.5",
        "gpt-image-1-mini",
    }


# ==================== GenerationCost.from_pricing_table ====================


def test_from_pricing_table_known_entry():
    """from_pricing_table returns GenerationCost for known provider/model."""
    result = GenerationCost.from_pricing_table("fal", "fal-ai/minimax", 1.0)
    assert result is not None
    assert result.amount_usd == Decimal("0.50")
    assert result.raw_amount == 1.0
    assert result.raw_unit == "videos"


def test_from_pricing_table_unknown_entry():
    """from_pricing_table returns None for unknown provider/model."""
    result = GenerationCost.from_pricing_table("unknown", "unknown-model", 1.0)
    assert result is None


def test_from_pricing_table_with_quantity():
    """from_pricing_table multiplies usd_per_unit by quantity."""
    result = GenerationCost.from_pricing_table("runway", "gen4_turbo", 10.0)
    assert result is not None
    assert result.amount_usd == Decimal("0.05") * Decimal("10.0")
    assert result.raw_amount == 10.0
    assert result.raw_unit == "seconds"


def test_from_pricing_table_zero_quantity():
    """from_pricing_table with quantity=0.0 returns amount_usd=0."""
    result = GenerationCost.from_pricing_table("fal", "fal-ai/veo3", 0.0)
    assert result is not None
    assert result.amount_usd == Decimal("0")
    assert result.raw_amount == 0.0
    assert result.raw_unit == "seconds"


def test_from_pricing_table_large_quantity():
    """from_pricing_table with very large quantity computes correctly."""
    result = GenerationCost.from_pricing_table(
        "elevenlabs", "eleven_multilingual_v2", 100000.0
    )
    assert result is not None
    assert result.amount_usd == Decimal("0.00024") * Decimal("100000.0")
    assert result.raw_amount == 100000.0
    assert result.raw_unit == "characters"


def test_from_pricing_table_openai_image_not_in_table():
    """OpenAI gpt-image models are not in the pricing table."""
    result = GenerationCost.from_pricing_table("openai", "gpt-image-1", 1.0)
    assert result is None


# ==================== GenerationCost.from_token_usage ====================


def _make_usage(
    total_tokens=0,
    input_tokens=0,
    output_tokens=0,
    input_details=None,
    output_details=None,
):
    """Helper to build a fake OpenAI usage object."""
    ns = SimpleNamespace(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=input_details,
        output_tokens_details=output_details,
    )
    return ns


def test_from_token_usage_basic():
    """from_token_usage computes cost from simple usage (no details)."""
    usage = _make_usage(total_tokens=1100, input_tokens=100, output_tokens=1000)
    result = GenerationCost.from_token_usage("gpt-image-1", usage)
    assert result is not None
    assert isinstance(result.amount_usd, Decimal)
    # input: 100 * 10/1M = 0.001, output: 1000 * 40/1M = 0.04
    expected = Decimal("100") * (Decimal("10") / 1_000_000) + Decimal("1000") * (
        Decimal("40") / 1_000_000
    )
    assert result.amount_usd == expected
    assert result.raw_unit == "tokens"


def test_from_token_usage_unknown_model():
    """from_token_usage returns None for unknown model."""
    usage = _make_usage(total_tokens=100, input_tokens=50, output_tokens=50)
    result = GenerationCost.from_token_usage("unknown-model", usage)
    assert result is None


def test_from_token_usage_none_usage():
    """from_token_usage returns None when usage is None."""
    result = GenerationCost.from_token_usage("gpt-image-1", None)
    assert result is None


def test_from_token_usage_zero_total_tokens():
    """from_token_usage returns None when total_tokens is 0."""
    usage = _make_usage(total_tokens=0, input_tokens=0, output_tokens=0)
    result = GenerationCost.from_token_usage("gpt-image-1", usage)
    assert result is None


def test_from_token_usage_with_input_details():
    """from_token_usage uses detailed input breakdown when available."""
    input_details = SimpleNamespace(
        text_tokens=50,
        image_tokens=100,
        cached_tokens=0,
    )
    usage = _make_usage(
        total_tokens=1150,
        input_tokens=150,
        output_tokens=1000,
        input_details=input_details,
    )
    result = GenerationCost.from_token_usage("gpt-image-1", usage)
    assert result is not None
    assert isinstance(result.amount_usd, Decimal)
    # text: 50 * 5/1M, image: 100 * 10/1M, output: 1000 * 40/1M
    expected = (
        Decimal("50") * (Decimal("5") / 1_000_000)
        + Decimal("100") * (Decimal("10") / 1_000_000)
        + Decimal("1000") * (Decimal("40") / 1_000_000)
    )
    assert result.amount_usd == expected


def test_from_token_usage_has_token_usage_field():
    """from_token_usage populates the token_usage field."""
    input_details = SimpleNamespace(text_tokens=50, image_tokens=100, cached_tokens=10)
    output_details = SimpleNamespace(image_tokens=800, text_tokens=200)
    usage = _make_usage(
        total_tokens=1160,
        input_tokens=150,
        output_tokens=1000,
        input_details=input_details,
        output_details=output_details,
    )
    result = GenerationCost.from_token_usage("gpt-image-1", usage)
    assert result is not None
    assert result.token_usage is not None
    assert result.token_usage.text_input_tokens == 50
    assert result.token_usage.image_input_tokens == 100
    assert result.token_usage.cached_tokens == 10
    assert result.token_usage.image_output_tokens == 800
    assert result.token_usage.text_output_tokens == 200


# ==================== GenerationCost.from_credits ====================


def test_from_credits_basic():
    """from_credits computes cost from credits-based billing."""
    result = GenerationCost.from_credits(
        quantity=10.0,
        credits_per_unit=5.0,
        credit_to_usd=0.01,
    )
    assert result.amount_usd == Decimal("10.0") * Decimal("5.0") * Decimal("0.01")
    assert result.raw_amount == 10.0
    assert result.raw_unit == "credits"


def test_from_credits_custom_unit():
    """from_credits allows custom raw_unit."""
    result = GenerationCost.from_credits(
        quantity=60.0,
        credits_per_unit=1.0,
        credit_to_usd=0.001,
        raw_unit="seconds",
    )
    assert result.amount_usd == Decimal("60.0") * Decimal("1.0") * Decimal("0.001")
    assert result.raw_unit == "seconds"


def test_from_credits_zero_quantity():
    """from_credits with zero quantity returns zero cost."""
    result = GenerationCost.from_credits(
        quantity=0.0, credits_per_unit=5.0, credit_to_usd=0.01
    )
    assert result.amount_usd == Decimal("0")
    assert result.raw_amount == 0.0
