"""Tests for multi-modal generation models."""

from decimal import Decimal

import pytest

from tarash.tarash_gateway.models import (
    MultiModalGenerationConfig,
    MultiModalGenerationRequest,
    MultiModalGenerationResponse,
    CostComponent,
    GenerationCost,
    ImageOutputItem,
    TextOutputItem,
    ReasoningOutputItem,
    CodeOutputItem,
    UnknownOutputItem,
)


# ---- MultiModalGenerationConfig ----


def test_config_defaults():
    """Config has sensible defaults."""
    config = MultiModalGenerationConfig(
        provider="openai", model="gpt-4o", api_key="test-key"
    )
    assert config.timeout == 300
    assert config.allowed_tools == ["image_generation"]
    assert config.store is True
    assert config.instructions is None
    assert config.api_version is None
    assert config.fallback_configs is None


def test_config_frozen():
    """Config is immutable."""
    config = MultiModalGenerationConfig(
        provider="openai", model="gpt-4o", api_key="test-key"
    )
    with pytest.raises(Exception):
        config.model = "gpt-5"


def test_config_with_fallback():
    """Config supports fallback chain."""
    fallback = MultiModalGenerationConfig(
        provider="openai", model="gpt-4o-mini", api_key="test-key"
    )
    config = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        fallback_configs=[fallback],
    )
    assert len(config.fallback_configs) == 1
    assert config.fallback_configs[0].model == "gpt-4o-mini"


# ---- MultiModalGenerationRequest ----


def test_request_basic():
    """Request with just a prompt."""
    request = MultiModalGenerationRequest(prompt="Generate an image of a cat")
    assert request.prompt == "Generate an image of a cat"
    assert request.input is None
    assert request.previous_response_id is None
    assert request.extra_params == {}


def test_request_captures_extra_fields():
    """Unknown fields captured into extra_params."""
    request = MultiModalGenerationRequest(prompt="test", custom_field="custom_value")
    assert request.extra_params == {"custom_field": "custom_value"}


def test_request_with_input_messages():
    """Request with multi-turn input."""
    messages = [{"role": "user", "content": "Hello"}]
    request = MultiModalGenerationRequest(prompt="test", input=messages)
    assert request.input == messages


# ---- Output Items ----


def test_text_output_item():
    """TextOutputItem holds text content."""
    item = TextOutputItem(content="Hello world")
    assert item.type == "text"
    assert item.content == "Hello world"


def test_image_output_item():
    """ImageOutputItem holds image data."""
    item = ImageOutputItem(url="https://example.com/image.png")
    assert item.type == "image"
    assert item.url == "https://example.com/image.png"
    assert item.base64 is None
    assert item.revised_prompt is None


def test_reasoning_output_item():
    """ReasoningOutputItem holds reasoning summaries."""
    item = ReasoningOutputItem(summary=["Step 1", "Step 2"])
    assert item.type == "reasoning"
    assert len(item.summary) == 2


def test_code_output_item():
    """CodeOutputItem holds code and execution output."""
    item = CodeOutputItem(code="print('hello')", output="hello")
    assert item.type == "code_output"


def test_unknown_output_item():
    """UnknownOutputItem passes through raw data."""
    item = UnknownOutputItem(raw={"some_key": "some_value"})
    assert item.type == "unknown"
    assert item.raw["some_key"] == "some_value"


# ---- MultiModalGenerationResponse ----


def test_response_text_property():
    """Response.text concatenates all TextOutputItems."""
    response = MultiModalGenerationResponse(
        request_id="test-123",
        items=[
            TextOutputItem(content="Hello"),
            ImageOutputItem(url="https://example.com/img.png"),
            TextOutputItem(content="World"),
        ],
        status="completed",
        raw_response={},
    )
    assert response.text == "Hello\nWorld"


def test_response_images_property():
    """Response.images returns only ImageOutputItems."""
    img = ImageOutputItem(url="https://example.com/img.png")
    response = MultiModalGenerationResponse(
        request_id="test-123",
        items=[
            TextOutputItem(content="Hello"),
            img,
        ],
        status="completed",
        raw_response={},
    )
    assert response.images == [img]


def test_response_frozen():
    """Response is immutable."""
    response = MultiModalGenerationResponse(
        request_id="test-123",
        items=[],
        status="completed",
        raw_response={},
    )
    with pytest.raises(Exception):
        response.status = "failed"


def test_response_with_cost_breakdown():
    """Response can carry cost with breakdown."""
    cost = GenerationCost(
        amount_usd=Decimal("0.07"),
        raw_amount=501.0,
        raw_unit="mixed",
        breakdown=(
            CostComponent(
                amount_usd=Decimal("0.03"), raw_amount=500.0, raw_unit="tokens"
            ),
            CostComponent(
                amount_usd=Decimal("0.04"), raw_amount=1.0, raw_unit="images"
            ),
        ),
    )
    response = MultiModalGenerationResponse(
        request_id="test-123",
        items=[TextOutputItem(content="Hello")],
        status="completed",
        cost=cost,
        raw_response={},
    )
    assert response.cost.amount_usd == Decimal("0.07")
    assert len(response.cost.breakdown) == 2
