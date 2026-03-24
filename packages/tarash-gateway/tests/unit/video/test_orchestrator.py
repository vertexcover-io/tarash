"""Tests for ExecutionOrchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import mock module to trigger VideoGenerationConfig.model_rebuild()
import tarash.tarash_gateway.mock  # noqa: F401
from tarash.tarash_gateway.exceptions import (
    HTTPError,
    ValidationError,
)
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    ImageGenerationConfig,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.orchestrator import ExecutionOrchestrator


def test_collect_fallback_chain_no_fallbacks():
    """Test collecting fallback chain with no fallbacks."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="test-key",
    )

    chain = ExecutionOrchestrator.collect_fallback_chain(config)

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

    chain = ExecutionOrchestrator.collect_fallback_chain(config)

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

    chain = ExecutionOrchestrator.collect_fallback_chain(config)

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

    # Mock handler
    handler = AsyncMock()
    handler.generate_video_async.return_value = VideoGenerationResponse(
        request_id="req-123",
        video="https://example.com/video.mp4",
        status="completed",
        raw_response={"status": "completed"},
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        response = await orchestrator.execute_async(config, request)

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

    # Mock handlers - first fails, second succeeds
    handler1 = AsyncMock()
    handler1.generate_video_async.side_effect = HTTPError(
        "Internal server error",
        provider="fal",
        model="fal-ai/veo3.1",
        status_code=500,
    )

    handler2 = AsyncMock()
    handler2.generate_video_async.return_value = VideoGenerationResponse(
        request_id="req-456",
        video="https://example.com/video2.mp4",
        status="completed",
        raw_response={"status": "completed"},
    )

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        handler = handlers[call_count]
        call_count += 1
        return handler

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler",
        side_effect=get_handler_mock,
    ):
        orchestrator = ExecutionOrchestrator()
        response = await orchestrator.execute_async(config, request)

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

    # Mock handler - fails with non-retryable error
    handler = AsyncMock()
    handler.generate_video_async.side_effect = ValidationError(
        "Invalid prompt",
        provider="fal",
        model="fal-ai/veo3.1",
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(ValidationError, match="Invalid prompt"):
            await orchestrator.execute_async(config, request)


def test_execute_sync_success_first_attempt():
    """Test synchronous execution success on first attempt."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="test-key",
    )

    request = VideoGenerationRequest(prompt="test prompt")

    # Mock handler
    handler = MagicMock()
    handler.generate_video.return_value = VideoGenerationResponse(
        request_id="req-123",
        video="https://example.com/video.mp4",
        status="completed",
        raw_response={"status": "completed"},
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        response = orchestrator.execute_sync(config, request)

    assert response.request_id == "req-123"
    assert response.video == "https://example.com/video.mp4"
    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1
    assert response.execution_metadata.successful_attempt == 1
    assert response.execution_metadata.fallback_triggered is False


# ==================== EDGE-005: NotImplementedError on video methods ====================


@pytest.mark.asyncio
async def test_execute_async_not_implemented_error_reraises_immediately():
    """NotImplementedError from video async handler re-raises without fallback (EDGE-005)."""
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

    handler = AsyncMock()
    handler.generate_video_async.side_effect = NotImplementedError(
        "Video generation not supported"
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(NotImplementedError, match="Video generation not supported"):
            await orchestrator.execute_async(config, request)


def test_execute_sync_not_implemented_error_reraises_immediately():
    """NotImplementedError from video sync handler re-raises without fallback (EDGE-005)."""
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

    handler = MagicMock()
    handler.generate_video.side_effect = NotImplementedError(
        "Video generation not supported"
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(NotImplementedError, match="Video generation not supported"):
            orchestrator.execute_sync(config, request)


# ==================== REQ-001: _collect_fallback_chain with image/audio configs ====================


def test_collect_fallback_chain_with_image_config():
    """_collect_fallback_chain works with ImageGenerationConfig (REQ-001)."""
    fallback = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="fal-key",
    )

    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="openai-key",
        fallback_configs=[fallback],
    )

    chain = ExecutionOrchestrator._collect_fallback_chain(config)

    assert len(chain) == 2
    assert chain[0].model == "dall-e-3"
    assert chain[1].model == "fal-ai/flux/dev"


def test_collect_fallback_chain_with_audio_config():
    """_collect_fallback_chain works with AudioGenerationConfig (REQ-001)."""
    fallback = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="cartesia-key",
    )

    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="elevenlabs-key",
        fallback_configs=[fallback],
    )

    chain = ExecutionOrchestrator._collect_fallback_chain(config)

    assert len(chain) == 2
    assert chain[0].model == "eleven_multilingual_v2"
    assert chain[1].model == "sonic-3"


def test_collect_fallback_chain_image_no_fallbacks():
    """_collect_fallback_chain with image config and no fallbacks (EDGE-001)."""
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="openai-key",
    )

    chain = ExecutionOrchestrator._collect_fallback_chain(config)

    assert len(chain) == 1
    assert chain[0].model == "dall-e-3"


def test_collect_fallback_chain_audio_deeply_nested():
    """_collect_fallback_chain with deeply nested audio fallbacks (EDGE-002)."""
    fallback3 = AudioGenerationConfig(
        model="fal-ai/minimax/speech-2.8-hd",
        provider="fal",
        api_key="fal-key",
    )

    fallback2 = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="cartesia-key",
        fallback_configs=[fallback3],
    )

    fallback1 = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="elevenlabs-key",
        fallback_configs=[fallback2],
    )

    config = AudioGenerationConfig(
        model="tts-1",
        provider="openai",
        api_key="openai-key",
        fallback_configs=[fallback1],
    )

    chain = ExecutionOrchestrator._collect_fallback_chain(config)

    assert len(chain) == 4
    assert chain[0].model == "tts-1"
    assert chain[1].model == "eleven_multilingual_v2"
    assert chain[2].model == "sonic-3"
    assert chain[3].model == "fal-ai/minimax/speech-2.8-hd"
