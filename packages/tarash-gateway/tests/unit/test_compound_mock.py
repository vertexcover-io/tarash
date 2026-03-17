"""Tests for mock provider compound generation."""

import pytest

from tarash.tarash_gateway.mock import MockConfig, MockProviderHandler
from tarash.tarash_gateway.models import (
    CompoundGenerationConfig,
    CompoundGenerationRequest,
    CompoundGenerationResponse,
    TextOutputItem,
    ImageOutputItem,
)


@pytest.fixture
def mock_config():
    return CompoundGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=[],
        mock=MockConfig(enabled=True),
    )


@pytest.fixture
def compound_request():
    return CompoundGenerationRequest(prompt="Test prompt")


def test_mock_compound_sync(mock_config, compound_request):
    """Mock handler returns a compound response synchronously."""
    handler = MockProviderHandler()
    response = handler.generate_compound(mock_config, compound_request)

    assert isinstance(response, CompoundGenerationResponse)
    assert response.status == "completed"
    assert response.is_mock is True
    assert len(response.items) > 0
    assert any(isinstance(item, TextOutputItem) for item in response.items)


async def test_mock_compound_async(mock_config, compound_request):
    """Mock handler returns a compound response asynchronously."""
    handler = MockProviderHandler()
    response = await handler.generate_compound_async(mock_config, compound_request)

    assert isinstance(response, CompoundGenerationResponse)
    assert response.status == "completed"
    assert response.is_mock is True


def test_mock_compound_includes_image_when_tool_allowed(compound_request):
    """Mock includes ImageOutputItem when image_generation is in allowed_tools."""
    config = CompoundGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=["image_generation"],
        mock=MockConfig(enabled=True),
    )
    handler = MockProviderHandler()
    response = handler.generate_compound(config, compound_request)

    assert any(isinstance(item, ImageOutputItem) for item in response.items)


def test_mock_compound_no_image_when_tool_not_allowed(compound_request):
    """Mock omits ImageOutputItem when image_generation is not in allowed_tools."""
    config = CompoundGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=[],
        mock=MockConfig(enabled=True),
    )
    handler = MockProviderHandler()
    response = handler.generate_compound(config, compound_request)

    assert not any(isinstance(item, ImageOutputItem) for item in response.items)


def test_mock_compound_requires_mock_enabled(compound_request):
    """Mock handler raises ValueError if mock not enabled."""
    config = CompoundGenerationConfig(
        provider="openai", model="gpt-4o", api_key="test-key"
    )
    handler = MockProviderHandler()
    with pytest.raises(ValueError, match="mock config"):
        handler.generate_compound(config, compound_request)


def test_mock_compound_text_content_includes_prompt(compound_request):
    """Mock text item includes the original prompt."""
    config = CompoundGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=[],
        mock=MockConfig(enabled=True),
    )
    handler = MockProviderHandler()
    response = handler.generate_compound(config, compound_request)

    text_items = [i for i in response.items if isinstance(i, TextOutputItem)]
    assert len(text_items) == 1
    assert compound_request.prompt in text_items[0].content


def test_mock_compound_request_id_has_mock_prefix(mock_config, compound_request):
    """Mock response request_id starts with 'mock_'."""
    handler = MockProviderHandler()
    response = handler.generate_compound(mock_config, compound_request)

    assert response.request_id.startswith("mock_")
