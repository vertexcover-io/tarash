"""Tests for OpenAI provider compound generation methods."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.models import (
    CompoundGenerationConfig,
    CompoundGenerationRequest,
    CompoundGenerationResponse,
    ImageOutputItem,
    TextOutputItem,
    UnknownOutputItem,
)
from tarash.tarash_gateway.providers.openai import OpenAIProviderHandler


@pytest.fixture
def handler():
    with patch("tarash.tarash_gateway.providers.openai.has_openai", True):
        return OpenAIProviderHandler()


@pytest.fixture
def config():
    return CompoundGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=["image_generation"],
    )


@pytest.fixture
def compound_request():
    return CompoundGenerationRequest(prompt="Write about cats and generate an image")


# ---- Request Conversion ----


def test_convert_compound_request_basic(handler, config, compound_request):
    """Basic prompt is converted to Responses API format."""
    params = handler._convert_compound_request(config, compound_request)

    assert params["model"] == "gpt-4o"
    assert params["input"] == "Write about cats and generate an image"
    assert params["store"] is True
    assert {"type": "image_generation"} in params["tools"]


def test_convert_compound_request_with_input_messages(handler, config):
    """Multi-turn input messages override prompt."""
    messages = [{"role": "user", "content": "Hello"}]
    req = CompoundGenerationRequest(prompt="ignored", input=messages)
    params = handler._convert_compound_request(config, req)
    assert params["input"] == messages


def test_convert_compound_request_with_instructions(handler, compound_request):
    """Instructions are included when set."""
    config = CompoundGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        instructions="You are a helpful assistant",
    )
    params = handler._convert_compound_request(config, compound_request)
    assert params["instructions"] == "You are a helpful assistant"


def test_convert_compound_request_code_interpreter(handler, compound_request):
    """Code interpreter tool is included when allowed."""
    config = CompoundGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=["image_generation", "code_interpreter"],
    )
    params = handler._convert_compound_request(config, compound_request)
    tools = params["tools"]
    assert {"type": "image_generation"} in tools
    assert {"type": "code_interpreter"} in tools


def test_convert_compound_request_extra_params(handler, config):
    """Extra params are merged into request."""
    req = CompoundGenerationRequest(prompt="test", extra_params={"temperature": 0.7})
    params = handler._convert_compound_request(config, req)
    assert params["temperature"] == 0.7


# ---- Response Conversion ----


def test_convert_compound_response_text_output(handler, config, compound_request):
    """Text message output is parsed to TextOutputItem."""
    content_part = MagicMock()
    content_part.type = "output_text"
    content_part.text = "Hello world"

    message_item = MagicMock()
    message_item.type = "message"
    message_item.content = [content_part]

    provider_response = MagicMock()
    provider_response.output = [message_item]
    provider_response.usage = None
    provider_response.model_dump.return_value = {}

    response = handler._convert_compound_response(
        config, compound_request, "test-123", provider_response
    )

    assert len(response.items) == 1
    assert isinstance(response.items[0], TextOutputItem)
    assert response.items[0].content == "Hello world"


def test_convert_compound_response_image_output(handler, config, compound_request):
    """Image generation output is parsed to ImageOutputItem."""
    image_item = MagicMock()
    image_item.type = "image_generation_call"
    image_item.url = "https://example.com/image.png"
    image_item.result = None
    image_item.revised_prompt = "A cute cat"

    provider_response = MagicMock()
    provider_response.output = [image_item]
    provider_response.usage = None
    provider_response.model_dump.return_value = {}

    response = handler._convert_compound_response(
        config, compound_request, "test-123", provider_response
    )

    assert len(response.items) == 1
    assert isinstance(response.items[0], ImageOutputItem)
    assert response.items[0].url == "https://example.com/image.png"
    assert response.items[0].revised_prompt == "A cute cat"


def test_convert_compound_response_mixed_output(handler, config, compound_request):
    """Mixed output items are parsed in order."""
    text_part = MagicMock()
    text_part.type = "output_text"
    text_part.text = "Here is a cat"

    message = MagicMock()
    message.type = "message"
    message.content = [text_part]

    image = MagicMock()
    image.type = "image_generation_call"
    image.url = "https://example.com/cat.png"
    image.result = None
    image.revised_prompt = None

    provider_response = MagicMock()
    provider_response.output = [message, image]
    provider_response.usage = None
    provider_response.model_dump.return_value = {}

    response = handler._convert_compound_response(
        config, compound_request, "test-123", provider_response
    )

    assert len(response.items) == 2
    assert isinstance(response.items[0], TextOutputItem)
    assert isinstance(response.items[1], ImageOutputItem)


def test_convert_compound_response_unknown_type(handler, config, compound_request):
    """Unknown output types are passed through as UnknownOutputItem."""
    unknown = MagicMock()
    unknown.type = "future_type"
    unknown.model_dump.return_value = {"type": "future_type", "data": "value"}

    provider_response = MagicMock()
    provider_response.output = [unknown]
    provider_response.usage = None
    provider_response.model_dump.return_value = {}

    response = handler._convert_compound_response(
        config, compound_request, "test-123", provider_response
    )

    assert len(response.items) == 1
    assert isinstance(response.items[0], UnknownOutputItem)


# ---- Sync/Async Generation ----


@patch("tarash.tarash_gateway.providers.openai.OpenAI")
def test_generate_compound_sync(mock_openai_cls, handler, config, compound_request):
    """Sync compound generation calls client.responses.create."""
    text_part = MagicMock()
    text_part.type = "output_text"
    text_part.text = "Generated text"

    message = MagicMock()
    message.type = "message"
    message.content = [text_part]

    mock_response = MagicMock()
    mock_response.output = [message]
    mock_response.usage = None
    mock_response.model_dump.return_value = {}

    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    result = handler.generate_compound(config, compound_request)

    assert isinstance(result, CompoundGenerationResponse)
    assert result.status == "completed"
    assert result.text == "Generated text"
    mock_client.responses.create.assert_called_once()


@patch("tarash.tarash_gateway.providers.openai.AsyncOpenAI")
async def test_generate_compound_async(
    mock_openai_cls, handler, config, compound_request
):
    """Async compound generation calls client.responses.create."""
    text_part = MagicMock()
    text_part.type = "output_text"
    text_part.text = "Generated text"

    message = MagicMock()
    message.type = "message"
    message.content = [text_part]

    mock_response = MagicMock()
    mock_response.output = [message]
    mock_response.usage = None
    mock_response.model_dump.return_value = {}

    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=mock_response)
    mock_openai_cls.return_value = mock_client

    result = await handler.generate_compound_async(config, compound_request)

    assert isinstance(result, CompoundGenerationResponse)
    assert result.status == "completed"
