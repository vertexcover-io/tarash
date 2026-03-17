"""Unit tests for pricing module: GenerationCost, PricingEntry, PRICING_TABLE, resolve_cost, lookup_pricing_table.

Covers: REQ-001, REQ-008, REQ-009, REQ-010, REQ-011, REQ-012, EDGE-002, EDGE-009, EDGE-013.
"""

import dataclasses

import pytest

from tarash.tarash_gateway.models import GenerationCost
from tarash.tarash_gateway.pricing import (
    PRICING_TABLE,
    PricingEntry,
    lookup_pricing_table,
    resolve_cost,
)


# ==================== GenerationCost (REQ-001) ====================


def test_generation_cost_creation():
    """GenerationCost can be created with all fields."""
    cost = GenerationCost(amount_usd=3.20, raw_amount=8.0, raw_unit="seconds")
    assert cost.amount_usd == 3.20
    assert cost.raw_amount == 8.0
    assert cost.raw_unit == "seconds"


def test_generation_cost_amount_usd_none():
    """GenerationCost accepts amount_usd=None."""
    cost = GenerationCost(amount_usd=None, raw_amount=5.0, raw_unit="videos")
    assert cost.amount_usd is None
    assert cost.raw_amount == 5.0
    assert cost.raw_unit == "videos"


def test_generation_cost_is_frozen():
    """GenerationCost is immutable (frozen dataclass)."""
    cost = GenerationCost(amount_usd=1.0, raw_amount=1.0, raw_unit="images")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cost.amount_usd = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cost.raw_amount = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cost.raw_unit = "videos"  # type: ignore[misc]


def test_generation_cost_is_dataclass():
    """GenerationCost is a dataclass."""
    assert dataclasses.is_dataclass(GenerationCost)


def test_generation_cost_fields():
    """GenerationCost has exactly three fields with correct names."""
    field_names = [f.name for f in dataclasses.fields(GenerationCost)]
    assert field_names == ["amount_usd", "raw_amount", "raw_unit"]


# ==================== PricingEntry (REQ-008) ====================


def test_pricing_entry_creation():
    """PricingEntry can be created with usd_per_unit and unit."""
    entry = PricingEntry(usd_per_unit=0.40, unit="seconds")
    assert entry.usd_per_unit == 0.40
    assert entry.unit == "seconds"


def test_pricing_entry_is_frozen():
    """PricingEntry is immutable (frozen dataclass)."""
    entry = PricingEntry(usd_per_unit=0.50, unit="videos")
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.usd_per_unit = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.unit = "images"  # type: ignore[misc]


def test_pricing_entry_is_dataclass():
    """PricingEntry is a dataclass."""
    assert dataclasses.is_dataclass(PricingEntry)


def test_pricing_entry_fields():
    """PricingEntry has exactly two fields with correct types."""
    fields = {f.name: f.type for f in dataclasses.fields(PricingEntry)}
    assert fields == {
        "usd_per_unit": "float",
        "unit": "str",
    }


# ==================== PRICING_TABLE (REQ-009) ====================


def test_pricing_table_is_dict():
    """PRICING_TABLE is a dict keyed by (provider, model) tuples."""
    assert isinstance(PRICING_TABLE, dict)
    for key, value in PRICING_TABLE.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], str)
        assert isinstance(key[1], str)
        assert isinstance(value, PricingEntry)


def test_pricing_table_spot_check_fal_veo3():
    """Spot-check: Fal veo3 entry."""
    entry = PRICING_TABLE[("fal", "fal-ai/veo3")]
    assert entry.usd_per_unit == 0.40
    assert entry.unit == "seconds"


def test_pricing_table_spot_check_fal_minimax():
    """Spot-check: Fal minimax entry."""
    entry = PRICING_TABLE[("fal", "fal-ai/minimax")]
    assert entry.usd_per_unit == 0.50
    assert entry.unit == "videos"


def test_pricing_table_spot_check_openai_gpt_image_1():
    """Spot-check: OpenAI gpt-image-1 entry."""
    entry = PRICING_TABLE[("openai", "gpt-image-1")]
    assert entry.usd_per_unit == 0.042
    assert entry.unit == "images"


def test_pricing_table_spot_check_runway_gen4_turbo():
    """Spot-check: Runway gen4_turbo entry."""
    entry = PRICING_TABLE[("runway", "gen4_turbo")]
    assert entry.usd_per_unit == 0.05
    assert entry.unit == "seconds"


def test_pricing_table_spot_check_elevenlabs():
    """Spot-check: ElevenLabs eleven_multilingual_v2 entry."""
    entry = PRICING_TABLE[("elevenlabs", "eleven_multilingual_v2")]
    assert entry.usd_per_unit == 0.00024
    assert entry.unit == "characters"


def test_pricing_table_spot_check_cartesia():
    """Spot-check: Cartesia sonic entry."""
    entry = PRICING_TABLE[("cartesia", "sonic")]
    assert entry.usd_per_unit == 0.000011
    assert entry.unit == "characters"


def test_pricing_table_spot_check_hume_octave():
    """Spot-check: Hume octave entry."""
    entry = PRICING_TABLE[("hume", "octave")]
    assert entry.usd_per_unit == 0.00015
    assert entry.unit == "characters"


def test_pricing_table_spot_check_sarvam_bulbul_v2():
    """Spot-check: Sarvam bulbul-v2 entry."""
    entry = PRICING_TABLE[("sarvam", "bulbul-v2")]
    assert entry.usd_per_unit == 0.000018
    assert entry.unit == "characters"


def test_pricing_table_spot_check_google_veo():
    """Spot-check: Google veo entry."""
    entry = PRICING_TABLE[("google", "veo-3.0-generate-preview")]
    assert entry.usd_per_unit == 0.40
    assert entry.unit == "seconds"


def test_pricing_table_spot_check_stability():
    """Spot-check: Stability sd3.5-large entry."""
    entry = PRICING_TABLE[("stability", "sd3.5-large")]
    assert entry.usd_per_unit == 0.065
    assert entry.unit == "images"


def test_pricing_table_spot_check_xai():
    """Spot-check: xAI grok-imagine-image entry."""
    entry = PRICING_TABLE[("xai", "grok-imagine-image")]
    assert entry.usd_per_unit == 0.02
    assert entry.unit == "images"


def test_pricing_table_spot_check_fal_kling():
    """Spot-check: Fal kling-video/o1 compute_seconds entry."""
    entry = PRICING_TABLE[("fal", "fal-ai/kling-video/o1")]
    assert entry.usd_per_unit == 0.00017
    assert entry.unit == "compute_seconds"


def test_pricing_table_no_replicate_entries():
    """No Replicate entries exist in the pricing table."""
    replicate_entries = [k for k in PRICING_TABLE if k[0] == "replicate"]
    assert replicate_entries == []


def test_pricing_table_has_expected_count():
    """PRICING_TABLE has all entries from the design doc."""
    # Count from the spec: 19 fal video + 11 fal image + 5 openai + 7 google + 4 runway
    # + 3 stability + 3 xai + 2 elevenlabs + 1 cartesia + 2 sarvam + 2 hume = 59
    assert len(PRICING_TABLE) == 59


# ==================== resolve_cost (REQ-010, REQ-011, REQ-012) ====================


def test_resolve_cost_with_non_none_api_cost_returns_it_unchanged():
    """resolve_cost with non-None api_cost returns it directly (REQ-010)."""
    api_cost = GenerationCost(amount_usd=1.50, raw_amount=3.0, raw_unit="seconds")
    result = resolve_cost("fal", "fal-ai/veo3", api_cost=api_cost, quantity=999.0)
    assert result is api_cost  # Same object, not a copy


def test_resolve_cost_with_api_cost_amount_usd_none_returns_it_unchanged():
    """resolve_cost with api_cost that has amount_usd=None returns it as-is (EDGE-009)."""
    api_cost = GenerationCost(amount_usd=None, raw_amount=5.0, raw_unit="tokens")
    result = resolve_cost("fal", "fal-ai/veo3", api_cost=api_cost, quantity=10.0)
    assert result is api_cost


def test_resolve_cost_pricing_table_fallback():
    """resolve_cost with api_cost=None uses pricing table (REQ-011)."""
    result = resolve_cost("fal", "fal-ai/veo3", api_cost=None, quantity=8.0)
    assert result is not None
    assert result.amount_usd == pytest.approx(0.40 * 8.0)
    assert result.raw_amount == 8.0
    assert result.raw_unit == "seconds"


def test_resolve_cost_unknown_provider_model_returns_none():
    """resolve_cost with unknown provider/model and api_cost=None returns None (REQ-012)."""
    result = resolve_cost("unknown", "unknown-model", api_cost=None, quantity=1.0)
    assert result is None


def test_resolve_cost_zero_quantity():
    """resolve_cost with quantity=0.0 returns GenerationCost with amount_usd=0.0 (EDGE-002)."""
    result = resolve_cost("fal", "fal-ai/veo3", api_cost=None, quantity=0.0)
    assert result is not None
    assert result.amount_usd == 0.0
    assert result.raw_amount == 0.0
    assert result.raw_unit == "seconds"


def test_resolve_cost_large_quantity():
    """resolve_cost with very large quantity computes correctly (EDGE-013)."""
    result = resolve_cost(
        "elevenlabs", "eleven_multilingual_v2", api_cost=None, quantity=100000.0
    )
    assert result is not None
    assert result.amount_usd == pytest.approx(0.00024 * 100000.0)
    assert result.raw_amount == 100000.0
    assert result.raw_unit == "characters"


def test_resolve_cost_openai_per_image():
    """resolve_cost for OpenAI gpt-image-1 with quantity=1."""
    result = resolve_cost("openai", "gpt-image-1", api_cost=None, quantity=1.0)
    assert result is not None
    assert result.amount_usd == pytest.approx(0.042)
    assert result.raw_amount == 1.0
    assert result.raw_unit == "images"


# ==================== lookup_pricing_table ====================


def test_lookup_pricing_table_known_entry():
    """lookup_pricing_table returns GenerationCost for known provider/model."""
    result = lookup_pricing_table("fal", "fal-ai/minimax", 1.0)
    assert result is not None
    assert result.amount_usd == pytest.approx(0.50)
    assert result.raw_amount == 1.0
    assert result.raw_unit == "videos"


def test_lookup_pricing_table_unknown_entry():
    """lookup_pricing_table returns None for unknown provider/model (REQ-012)."""
    result = lookup_pricing_table("unknown", "unknown-model", 1.0)
    assert result is None


def test_lookup_pricing_table_with_quantity():
    """lookup_pricing_table multiplies usd_per_unit by quantity."""
    result = lookup_pricing_table("runway", "gen4_turbo", 10.0)
    assert result is not None
    assert result.amount_usd == pytest.approx(0.05 * 10.0)
    assert result.raw_amount == 10.0
    assert result.raw_unit == "seconds"
