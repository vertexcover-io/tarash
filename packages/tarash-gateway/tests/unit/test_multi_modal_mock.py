"""Tests for mock provider multi-modal generation."""

import pytest

from tarash.tarash_gateway.mock import MockConfig, MockProviderHandler
from tarash.tarash_gateway.models import (
    MultiModalGenerationConfig,
    MultiModalGenerationRequest,
    MultiModalGenerationResponse,
    TextOutputItem,
    ImageOutputItem,
)


@pytest.fixture
def mock_config():
    return MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=[],
        mock=MockConfig(enabled=True),
    )


@pytest.fixture
def multi_modal_request():
    return MultiModalGenerationRequest(prompt="Test prompt")


def test_mock_multi_modal_sync(mock_config, multi_modal_request):
    """Mock handler returns a multi-modal response synchronously."""
    handler = MockProviderHandler()
    response = handler.generate_multi_modal(mock_config, multi_modal_request)

    assert isinstance(response, MultiModalGenerationResponse)
    assert response.status == "completed"
    assert response.is_mock is True
    assert len(response.items) > 0
    assert any(isinstance(item, TextOutputItem) for item in response.items)


async def test_mock_multi_modal_async(mock_config, multi_modal_request):
    """Mock handler returns a multi-modal response asynchronously."""
    handler = MockProviderHandler()
    response = await handler.generate_multi_modal_async(mock_config, multi_modal_request)

    assert isinstance(response, MultiModalGenerationResponse)
    assert response.status == "completed"
    assert response.is_mock is True


def test_mock_multi_modal_includes_image_when_tool_allowed(multi_modal_request):
    """Mock includes ImageOutputItem when image_generation is in allowed_tools."""
    config = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=["image_generation"],
        mock=MockConfig(enabled=True),
    )
    handler = MockProviderHandler()
    response = handler.generate_multi_modal(config, multi_modal_request)

    assert any(isinstance(item, ImageOutputItem) for item in response.items)


def test_mock_multi_modal_no_image_when_tool_not_allowed(multi_modal_request):
    """Mock omits ImageOutputItem when image_generation is not in allowed_tools."""
    config = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=[],
        mock=MockConfig(enabled=True),
    )
    handler = MockProviderHandler()
    response = handler.generate_multi_modal(config, multi_modal_request)

    assert not any(isinstance(item, ImageOutputItem) for item in response.items)


def test_mock_multi_modal_requires_mock_enabled(multi_modal_request):
    """Mock handler raises ValueError if mock not enabled."""
    config = MultiModalGenerationConfig(
        provider="openai", model="gpt-4o", api_key="test-key"
    )
    handler = MockProviderHandler()
    with pytest.raises(ValueError, match="mock config"):
        handler.generate_multi_modal(config, multi_modal_request)


def test_mock_multi_modal_text_content_includes_prompt(multi_modal_request):
    """Mock text item includes the original prompt."""
    config = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=[],
        mock=MockConfig(enabled=True),
    )
    handler = MockProviderHandler()
    response = handler.generate_multi_modal(config, multi_modal_request)

    text_items = [i for i in response.items if isinstance(i, TextOutputItem)]
    assert len(text_items) == 1
    assert multi_modal_request.prompt in text_items[0].content


def test_mock_multi_modal_request_id_has_mock_prefix(mock_config, multi_modal_request):
    """Mock response request_id starts with 'mock_'."""
    handler = MockProviderHandler()
    response = handler.generate_multi_modal(mock_config, multi_modal_request)

    assert response.request_id.startswith("mock_")
