"""Tests for multi-modal generation public API."""

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from tarash.tarash_gateway.api import generate_multi_modal, generate_multi_modal_async
from tarash.tarash_gateway.models import (
    MultiModalGenerationConfig,
    MultiModalGenerationRequest,
    MultiModalGenerationResponse,
    GenerationCost,
    TextOutputItem,
)


@pytest.fixture
def config():
    return MultiModalGenerationConfig(
        provider="openai", model="gpt-4o", api_key="test-key"
    )


@pytest.fixture
def multi_modal_request():
    return MultiModalGenerationRequest(prompt="Test prompt")


@pytest.fixture
def mock_response():
    return MultiModalGenerationResponse(
        request_id="test-123",
        items=[TextOutputItem(content="Hello")],
        status="completed",
        cost=GenerationCost(
            amount_usd=Decimal("0.01"), raw_amount=100.0, raw_unit="tokens"
        ),
        raw_response={},
    )


@patch("tarash.tarash_gateway.api._orchestrator")
def test_generate_multi_modal_delegates_to_orchestrator(
    mock_orch, config, multi_modal_request, mock_response
):
    """generate_multi_modal delegates to orchestrator.execute_multi_modal_sync."""
    mock_orch.execute_multi_modal_sync.return_value = mock_response

    result = generate_multi_modal(config, multi_modal_request)

    mock_orch.execute_multi_modal_sync.assert_called_once_with(
        config, multi_modal_request, on_progress=None
    )
    assert result.status == "completed"


@patch("tarash.tarash_gateway.api._orchestrator")
async def test_generate_multi_modal_async_delegates_to_orchestrator(
    mock_orch, config, multi_modal_request, mock_response
):
    """generate_multi_modal_async delegates to orchestrator.execute_multi_modal_async."""
    mock_orch.execute_multi_modal_async = AsyncMock(return_value=mock_response)

    result = await generate_multi_modal_async(config, multi_modal_request)

    mock_orch.execute_multi_modal_async.assert_called_once_with(
        config, multi_modal_request, on_progress=None
    )
    assert result.status == "completed"
