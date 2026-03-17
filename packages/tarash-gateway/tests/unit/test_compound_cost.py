"""Tests for CostComponent and GenerationCost.breakdown field."""

from decimal import Decimal

import pytest

from tarash.tarash_gateway.models import CostComponent, GenerationCost


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
