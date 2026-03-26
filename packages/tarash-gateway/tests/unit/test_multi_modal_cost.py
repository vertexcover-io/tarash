"""Tests for CostComponent and GenerationCost.breakdown field."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tarash.tarash_gateway.models import CostComponent, GenerationCost
from tarash.tarash_gateway.pricing import OPENAI_MULTI_MODAL_TOKEN_RATES


def test_cost_component_creation():
    """CostComponent can be created with all fields."""
    component = CostComponent(
        amount_usd=Decimal("0.05"),
        raw_amount=1000.0,
        raw_unit="tokens",
    )
    assert component.amount_usd == Decimal("0.05")
    assert component.raw_amount == 1000.0
    assert component.raw_unit == "tokens"


def test_cost_component_frozen():
    """CostComponent is immutable."""
    component = CostComponent(
        amount_usd=Decimal("0.05"), raw_amount=1.0, raw_unit="tokens"
    )
    with pytest.raises(AttributeError):
        component.amount_usd = Decimal("0.10")


def test_cost_component_none_usd():
    """CostComponent allows None for amount_usd."""
    component = CostComponent(amount_usd=None, raw_amount=5.0, raw_unit="images")
    assert component.amount_usd is None


def test_generation_cost_breakdown_default_empty():
    """GenerationCost.breakdown defaults to empty tuple."""
    cost = GenerationCost(
        amount_usd=Decimal("0.10"), raw_amount=5.0, raw_unit="seconds"
    )
    assert cost.breakdown == ()


def test_generation_cost_with_breakdown():
    """GenerationCost can include per-component breakdown."""
    token_component = CostComponent(
        amount_usd=Decimal("0.03"), raw_amount=500.0, raw_unit="tokens"
    )
    image_component = CostComponent(
        amount_usd=Decimal("0.04"), raw_amount=1.0, raw_unit="images"
    )
    cost = GenerationCost(
        amount_usd=Decimal("0.07"),
        raw_amount=501.0,
        raw_unit="mixed",
        breakdown=(token_component, image_component),
    )
    assert len(cost.breakdown) == 2
    assert cost.breakdown[0].raw_unit == "tokens"
    assert cost.breakdown[1].raw_unit == "images"


def test_existing_cost_backwards_compatible():
    """Existing code creating GenerationCost without breakdown still works."""
    cost = GenerationCost(
        amount_usd=Decimal("0.40"), raw_amount=8.0, raw_unit="seconds"
    )
    assert cost.breakdown == ()
    assert cost.amount_usd == Decimal("0.40")


def test_openai_multi_modal_token_rates_exist():
    """OPENAI_MULTI_MODAL_TOKEN_RATES has entries for known models."""
    assert "gpt-4o" in OPENAI_MULTI_MODAL_TOKEN_RATES
    rates = OPENAI_MULTI_MODAL_TOKEN_RATES["gpt-4o"]
    assert "text_input" in rates
    assert "text_output" in rates


def test_from_token_usage_with_custom_rates_table():
    """from_token_usage accepts a custom rates_table parameter."""
    custom_rates = {
        "test-model": {
            "text_input": Decimal("5.00") / 1_000_000,
            "image_input": Decimal("10.00") / 1_000_000,
            "image_output": Decimal("40.00") / 1_000_000,
        }
    }
    usage = MagicMock()
    usage.total_tokens = 100
    usage.input_tokens = 50
    usage.output_tokens = 50
    usage.input_tokens_details = None
    usage.output_tokens_details = None

    cost = GenerationCost.from_token_usage(
        "test-model", usage, rates_table=custom_rates
    )
    assert cost is not None
    assert cost.amount_usd > Decimal("0")
    assert cost.raw_unit == "tokens"


def test_from_token_usage_default_rates_unchanged():
    """from_token_usage without rates_table still uses OPENAI_IMAGE_TOKEN_RATES."""
    usage = MagicMock()
    usage.total_tokens = 100
    usage.input_tokens = 50
    usage.output_tokens = 50
    usage.input_tokens_details = None
    usage.output_tokens_details = None

    # gpt-image-1 is in OPENAI_IMAGE_TOKEN_RATES (default)
    cost = GenerationCost.from_token_usage("gpt-image-1", usage)
    assert cost is not None

    # gpt-4o is NOT in OPENAI_IMAGE_TOKEN_RATES
    cost_none = GenerationCost.from_token_usage("gpt-4o", usage)
    assert cost_none is None


def test_from_token_usage_with_multi_modal_rates():
    """from_token_usage with OPENAI_MULTI_MODAL_TOKEN_RATES works for gpt-4o."""
    usage = MagicMock()
    usage.total_tokens = 100
    usage.input_tokens = 50
    usage.output_tokens = 50
    usage.input_tokens_details = None
    usage.output_tokens_details = None

    cost = GenerationCost.from_token_usage(
        "gpt-4o", usage, rates_table=OPENAI_MULTI_MODAL_TOKEN_RATES
    )
    assert cost is not None
    assert cost.amount_usd > Decimal("0")
