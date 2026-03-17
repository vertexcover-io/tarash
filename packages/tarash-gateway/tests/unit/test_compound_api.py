"""Tests for compound generation public API."""

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from tarash.tarash_gateway.api import generate_compound, generate_compound_async
from tarash.tarash_gateway.models import (
    CompoundGenerationConfig,
    CompoundGenerationRequest,
    CompoundGenerationResponse,
    GenerationCost,
    TextOutputItem,
)


@pytest.fixture
def config():
    return CompoundGenerationConfig(
        provider="openai", model="gpt-4o", api_key="test-key"
    )


@pytest.fixture
def compound_request():
    return CompoundGenerationRequest(prompt="Test prompt")


@pytest.fixture
def mock_response():
    return CompoundGenerationResponse(
        request_id="test-123",
        items=[TextOutputItem(content="Hello")],
        status="completed",
        cost=GenerationCost(
            amount_usd=Decimal("0.01"), raw_amount=100.0, raw_unit="tokens"
        ),
        raw_response={},
    )


@patch("tarash.tarash_gateway.api._ORCHESTRATOR")
def test_generate_compound_delegates_to_orchestrator(
    mock_orch, config, compound_request, mock_response
):
    """generate_compound delegates to orchestrator.execute_compound_sync."""
    mock_orch.execute_compound_sync.return_value = mock_response

    result = generate_compound(config, compound_request)

    mock_orch.execute_compound_sync.assert_called_once_with(
        config, compound_request, on_progress=None
    )
    assert result.status == "completed"


@patch("tarash.tarash_gateway.api._ORCHESTRATOR")
async def test_generate_compound_async_delegates_to_orchestrator(
    mock_orch, config, compound_request, mock_response
):
    """generate_compound_async delegates to orchestrator.execute_compound_async."""
    mock_orch.execute_compound_async = AsyncMock(return_value=mock_response)

    result = await generate_compound_async(config, compound_request)

    mock_orch.execute_compound_async.assert_called_once_with(
        config, compound_request, on_progress=None
    )
    assert result.status == "completed"
