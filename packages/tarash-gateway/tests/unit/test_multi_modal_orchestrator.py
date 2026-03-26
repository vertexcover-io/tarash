"""Tests for multi-modal generation orchestrator methods."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.models import (
    MultiModalGenerationConfig,
    MultiModalGenerationRequest,
    MultiModalGenerationResponse,
    GenerationCost,
    TextOutputItem,
)
from tarash.tarash_gateway.orchestrator import ExecutionOrchestrator
from tarash.tarash_gateway.exceptions import GenerationFailedError, ValidationError


@pytest.fixture
def orchestrator():
    return ExecutionOrchestrator()


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
        raw_response={"output": []},
    )


def test_collect_multi_modal_fallback_chain_single(config):
    """Single config produces chain of length 1."""
    chain = ExecutionOrchestrator.collect_multi_modal_fallback_chain(config)
    assert len(chain) == 1
    assert chain[0] is config


def test_collect_multi_modal_fallback_chain_nested():
    """Nested fallback configs are collected depth-first."""
    fallback2 = MultiModalGenerationConfig(
        provider="openai", model="gpt-4o-mini", api_key="test-key"
    )
    fallback1 = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        fallback_configs=[fallback2],
    )
    primary = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-5",
        api_key="test-key",
        fallback_configs=[fallback1],
    )
    chain = ExecutionOrchestrator.collect_multi_modal_fallback_chain(primary)
    assert len(chain) == 3
    assert chain[0].model == "gpt-5"
    assert chain[1].model == "gpt-4o"
    assert chain[2].model == "gpt-4o-mini"


@patch("tarash.tarash_gateway.orchestrator.get_handler")
def test_execute_multi_modal_sync_success(
    mock_get_handler, orchestrator, config, multi_modal_request, mock_response
):
    """Sync execution returns response with execution_metadata."""
    handler = MagicMock()
    handler.generate_multi_modal.return_value = mock_response
    mock_get_handler.return_value = handler

    result = orchestrator.execute_multi_modal_sync(config, multi_modal_request)

    assert result.status == "completed"
    assert result.execution_metadata is not None
    assert result.execution_metadata.total_attempts == 1
    assert result.execution_metadata.successful_attempt == 1
    assert result.execution_metadata.fallback_triggered is False


@patch("tarash.tarash_gateway.orchestrator.get_handler")
async def test_execute_multi_modal_async_success(
    mock_get_handler, orchestrator, config, multi_modal_request, mock_response
):
    """Async execution returns response with execution_metadata."""
    handler = AsyncMock()
    handler.generate_multi_modal_async = AsyncMock(return_value=mock_response)
    mock_get_handler.return_value = handler

    result = await orchestrator.execute_multi_modal_async(config, multi_modal_request)

    assert result.status == "completed"
    assert result.execution_metadata is not None
    assert result.execution_metadata.total_attempts == 1


@patch("tarash.tarash_gateway.orchestrator.get_handler")
def test_execute_multi_modal_sync_fallback(
    mock_get_handler, orchestrator, multi_modal_request, mock_response
):
    """Sync execution falls back on retryable error."""
    fallback = MultiModalGenerationConfig(
        provider="openai", model="gpt-4o-mini", api_key="test-key"
    )
    config = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        fallback_configs=[fallback],
    )

    handler_fail = MagicMock()
    handler_fail.generate_multi_modal.side_effect = GenerationFailedError(
        "Failed", provider="openai", model="gpt-4o"
    )
    handler_success = MagicMock()
    handler_success.generate_multi_modal.return_value = mock_response

    mock_get_handler.side_effect = [handler_fail, handler_success]

    result = orchestrator.execute_multi_modal_sync(config, multi_modal_request)

    assert result.status == "completed"
    assert result.execution_metadata.fallback_triggered is True
    assert result.execution_metadata.total_attempts == 2


@patch("tarash.tarash_gateway.orchestrator.get_handler")
def test_execute_multi_modal_sync_non_retryable_stops(
    mock_get_handler, orchestrator, config, multi_modal_request
):
    """Non-retryable error stops fallback chain immediately."""
    handler = MagicMock()
    handler.generate_multi_modal.side_effect = ValidationError(
        "Bad input", provider="openai", model="gpt-4o"
    )
    mock_get_handler.return_value = handler

    with pytest.raises(ValidationError, match="Bad input"):
        orchestrator.execute_multi_modal_sync(config, multi_modal_request)
