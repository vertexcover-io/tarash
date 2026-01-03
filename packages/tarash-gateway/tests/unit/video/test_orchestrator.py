"""Tests for FallbackOrchestrator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tarash.tarash_gateway.video.exceptions import (
    HTTPError,
    ValidationError,
)
from tarash.tarash_gateway.video.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.video.orchestrator import FallbackOrchestrator


def test_collect_fallback_chain_no_fallbacks():
    """Test collecting fallback chain with no fallbacks."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="test-key",
    )

    chain = FallbackOrchestrator.collect_fallback_chain(config)

    assert len(chain) == 1
    assert chain[0].model == "fal-ai/veo3.1"


def test_collect_fallback_chain_with_fallbacks():
    """Test collecting fallback chain with multiple fallbacks."""
    fallback1 = VideoGenerationConfig(
        model="replicate/minimax",
        provider="replicate",
        api_key="replicate-key",
    )

    fallback2 = VideoGenerationConfig(
        model="openai/sora-2",
        provider="openai",
        api_key="openai-key",
    )

    config = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="fal-key",
        fallback_configs=[fallback1, fallback2],
    )

    chain = FallbackOrchestrator.collect_fallback_chain(config)

    assert len(chain) == 3
    assert chain[0].model == "fal-ai/veo3.1"
    assert chain[1].model == "replicate/minimax"
    assert chain[2].model == "openai/sora-2"


def test_collect_fallback_chain_depth_first():
    """Test that fallback chain is collected depth-first."""
    fallback2 = VideoGenerationConfig(
        model="openai/sora-2",
        provider="openai",
        api_key="openai-key",
    )

    fallback1 = VideoGenerationConfig(
        model="replicate/minimax",
        provider="replicate",
        api_key="replicate-key",
        fallback_configs=[fallback2],
    )

    config = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="fal-key",
        fallback_configs=[fallback1],
    )

    chain = FallbackOrchestrator.collect_fallback_chain(config)

    # Depth-first: primary -> fallback1 -> fallback2
    assert len(chain) == 3
    assert chain[0].model == "fal-ai/veo3.1"
    assert chain[1].model == "replicate/minimax"
    assert chain[2].model == "openai/sora-2"


@pytest.mark.asyncio
async def test_execute_async_success_first_attempt():
    """Test successful execution on first attempt."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="test-key",
    )

    request = VideoGenerationRequest(prompt="test prompt")

    # Mock handler factory
    async def handler_factory(cfg):
        handler = AsyncMock()
        handler.generate_video_async.return_value = VideoGenerationResponse(
            request_id="req-123",
            video="https://example.com/video.mp4",
            status="completed",
            raw_response={"status": "completed"},
        )
        return handler

    orchestrator = FallbackOrchestrator()
    response = await orchestrator.execute_async(config, request, handler_factory)

    assert response.request_id == "req-123"
    assert response.video == "https://example.com/video.mp4"
    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1
    assert response.execution_metadata.successful_attempt == 1
    assert response.execution_metadata.fallback_triggered is False


@pytest.mark.asyncio
async def test_execute_async_fallback_on_retryable_error():
    """Test fallback triggered on retryable error."""
    fallback_config = VideoGenerationConfig(
        model="replicate/minimax",
        provider="replicate",
        api_key="replicate-key",
    )

    config = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="fal-key",
        fallback_configs=[fallback_config],
    )

    request = VideoGenerationRequest(prompt="test prompt")

    # Mock handler factory - first fails, second succeeds
    call_count = 0

    async def handler_factory(cfg):
        nonlocal call_count
        call_count += 1
        handler = AsyncMock()

        if call_count == 1:
            # First attempt fails with retryable error
            handler.generate_video_async.side_effect = HTTPError(
                "Internal server error",
                provider="fal",
                model="fal-ai/veo3.1",
                status_code=500,
            )
        else:
            # Second attempt succeeds
            handler.generate_video_async.return_value = VideoGenerationResponse(
                request_id="req-456",
                video="https://example.com/video2.mp4",
                status="completed",
                raw_response={"status": "completed"},
            )

        return handler

    orchestrator = FallbackOrchestrator()
    response = await orchestrator.execute_async(config, request, handler_factory)

    assert response.request_id == "req-456"
    assert response.video == "https://example.com/video2.mp4"
    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 2
    assert response.execution_metadata.successful_attempt == 2
    assert response.execution_metadata.fallback_triggered is True


@pytest.mark.asyncio
async def test_execute_async_non_retryable_error_no_fallback():
    """Test that non-retryable errors don't trigger fallback."""
    fallback_config = VideoGenerationConfig(
        model="replicate/minimax",
        provider="replicate",
        api_key="replicate-key",
    )

    config = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="fal-key",
        fallback_configs=[fallback_config],
    )

    request = VideoGenerationRequest(prompt="test prompt")

    # Mock handler factory - fails with non-retryable error
    async def handler_factory(cfg):
        handler = AsyncMock()
        handler.generate_video_async.side_effect = ValidationError(
            "Invalid prompt",
            provider="fal",
            model="fal-ai/veo3.1",
        )
        return handler

    orchestrator = FallbackOrchestrator()

    with pytest.raises(ValidationError, match="Invalid prompt"):
        await orchestrator.execute_async(config, request, handler_factory)


def test_execute_sync_success_first_attempt():
    """Test synchronous execution success on first attempt."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="test-key",
    )

    request = VideoGenerationRequest(prompt="test prompt")

    # Mock handler factory
    def handler_factory(cfg):
        handler = MagicMock()
        handler.generate_video.return_value = VideoGenerationResponse(
            request_id="req-123",
            video="https://example.com/video.mp4",
            status="completed",
            raw_response={"status": "completed"},
        )
        return handler

    orchestrator = FallbackOrchestrator()
    response = orchestrator.execute_sync(config, request, handler_factory)

    assert response.request_id == "req-123"
    assert response.video == "https://example.com/video.mp4"
    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1
    assert response.execution_metadata.successful_attempt == 1
    assert response.execution_metadata.fallback_triggered is False
