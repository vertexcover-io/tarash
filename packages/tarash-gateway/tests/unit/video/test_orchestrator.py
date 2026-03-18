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
    ImageGenerationRequest,
    ImageGenerationResponse,
    STSRequest,
    STSResponse,
    TTSRequest,
    TTSResponse,
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


# ==================== Video exhaustion tests ====================


@pytest.mark.asyncio
async def test_execute_async_all_providers_exhausted_raises_last_error():
    """All providers fail with retryable errors — last error is raised."""
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

    handler1 = AsyncMock()
    handler1.generate_video_async.side_effect = HTTPError(
        "Server error 1",
        provider="fal",
        model="fal-ai/veo3.1",
        status_code=500,
    )
    handler2 = AsyncMock()
    handler2.generate_video_async.side_effect = HTTPError(
        "Server error 2",
        provider="replicate",
        model="replicate/minimax",
        status_code=502,
    )

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler",
        side_effect=get_handler_mock,
    ):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(HTTPError, match="Server error 2"):
            await orchestrator.execute_async(config, request)


def test_execute_sync_all_providers_exhausted_raises_last_error():
    """Sync: all providers fail with retryable errors — last error is raised."""
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

    handler1 = MagicMock()
    handler1.generate_video.side_effect = HTTPError(
        "Server error 1",
        provider="fal",
        model="fal-ai/veo3.1",
        status_code=500,
    )
    handler2 = MagicMock()
    handler2.generate_video.side_effect = HTTPError(
        "Server error 2",
        provider="replicate",
        model="replicate/minimax",
        status_code=502,
    )

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler",
        side_effect=get_handler_mock,
    ):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(HTTPError, match="Server error 2"):
            orchestrator.execute_sync(config, request)


def test_execute_sync_non_retryable_error_no_fallback():
    """Sync: non-retryable errors stop the chain immediately."""
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
    handler.generate_video.side_effect = ValidationError(
        "Invalid prompt",
        provider="fal",
        model="fal-ai/veo3.1",
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(ValidationError, match="Invalid prompt"):
            orchestrator.execute_sync(config, request)


def test_execute_sync_fallback_on_retryable_error():
    """Sync: fallback triggered on retryable error, second provider succeeds."""
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

    handler1 = MagicMock()
    handler1.generate_video.side_effect = HTTPError(
        "Internal server error",
        provider="fal",
        model="fal-ai/veo3.1",
        status_code=500,
    )
    handler2 = MagicMock()
    handler2.generate_video.return_value = VideoGenerationResponse(
        request_id="req-fallback",
        video="https://example.com/video2.mp4",
        status="completed",
        raw_response={"status": "completed"},
    )

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler",
        side_effect=get_handler_mock,
    ):
        orchestrator = ExecutionOrchestrator()
        response = orchestrator.execute_sync(config, request)

    assert response.request_id == "req-fallback"
    assert response.execution_metadata.total_attempts == 2
    assert response.execution_metadata.fallback_triggered is True


# ==================== Image fallback chain tests ====================


def test_collect_image_fallback_chain_no_fallbacks():
    """Image: single config yields chain of length 1."""
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="test-key",
    )
    chain = ExecutionOrchestrator.collect_image_fallback_chain(config)
    assert len(chain) == 1
    assert chain[0].model == "dall-e-3"


def test_collect_image_fallback_chain_nested():
    """Image: nested fallbacks are collected depth-first."""
    fb2 = ImageGenerationConfig(
        model="stability/sdxl",
        provider="stability",
        api_key="key-3",
    )
    fb1 = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="key-2",
        fallback_configs=[fb2],
    )
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="key-1",
        fallback_configs=[fb1],
    )
    chain = ExecutionOrchestrator.collect_image_fallback_chain(config)
    assert len(chain) == 3
    assert [c.model for c in chain] == ["dall-e-3", "fal-ai/flux/dev", "stability/sdxl"]


# ==================== Image generation tests ====================


@pytest.fixture
def _image_config():
    return ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="test-key",
    )


@pytest.fixture
def _image_request():
    return ImageGenerationRequest(prompt="a cat")


@pytest.fixture
def _image_response():
    return ImageGenerationResponse(
        request_id="img-123",
        images=["https://example.com/img.png"],
        status="completed",
        raw_response={"ok": True},
    )


@pytest.mark.asyncio
async def test_execute_image_async_success(
    _image_config, _image_request, _image_response
):
    """Image async: success on first attempt attaches metadata."""
    handler = AsyncMock()
    handler.generate_image_async.return_value = _image_response

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        resp = await orchestrator.execute_image_async(_image_config, _image_request)

    assert resp.request_id == "img-123"
    assert resp.execution_metadata is not None
    assert resp.execution_metadata.total_attempts == 1
    assert resp.execution_metadata.fallback_triggered is False


@pytest.mark.asyncio
async def test_execute_image_async_fallback_on_retryable(
    _image_request, _image_response
):
    """Image async: retryable error triggers fallback to next provider."""
    fb = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="key-2",
    )
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = AsyncMock()
    handler1.generate_image_async.side_effect = HTTPError(
        "Server error",
        provider="openai",
        model="dall-e-3",
        status_code=500,
    )
    handler2 = AsyncMock()
    handler2.generate_image_async.return_value = _image_response

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        resp = await orchestrator.execute_image_async(config, _image_request)

    assert resp.request_id == "img-123"
    assert resp.execution_metadata.total_attempts == 2
    assert resp.execution_metadata.fallback_triggered is True


@pytest.mark.asyncio
async def test_execute_image_async_non_retryable_stops(_image_request):
    """Image async: non-retryable error stops chain immediately."""
    fb = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="key-2",
    )
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler = AsyncMock()
    handler.generate_image_async.side_effect = ValidationError(
        "Bad size",
        provider="openai",
        model="dall-e-3",
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(ValidationError, match="Bad size"):
            await orchestrator.execute_image_async(config, _image_request)


@pytest.mark.asyncio
async def test_execute_image_async_all_exhausted(_image_request):
    """Image async: all providers fail with retryable errors — raises last."""
    fb = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="key-2",
    )
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = AsyncMock()
    handler1.generate_image_async.side_effect = HTTPError(
        "Err 1",
        provider="openai",
        model="dall-e-3",
        status_code=500,
    )
    handler2 = AsyncMock()
    handler2.generate_image_async.side_effect = HTTPError(
        "Err 2",
        provider="fal",
        model="fal-ai/flux/dev",
        status_code=502,
    )

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(HTTPError, match="Err 2"):
            await orchestrator.execute_image_async(config, _image_request)


@pytest.mark.asyncio
async def test_execute_image_async_not_implemented_propagates(
    _image_config, _image_request
):
    """Image async: NotImplementedError propagates without fallback."""
    handler = AsyncMock()
    handler.generate_image_async.side_effect = NotImplementedError("not supported")

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(NotImplementedError, match="not supported"):
            await orchestrator.execute_image_async(_image_config, _image_request)


def test_execute_image_sync_success(_image_config, _image_request, _image_response):
    """Image sync: success on first attempt."""
    handler = MagicMock()
    handler.generate_image.return_value = _image_response

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        resp = orchestrator.execute_image_sync(_image_config, _image_request)

    assert resp.request_id == "img-123"
    assert resp.execution_metadata.total_attempts == 1
    assert resp.execution_metadata.fallback_triggered is False


def test_execute_image_sync_fallback_on_retryable(_image_request, _image_response):
    """Image sync: retryable error triggers fallback."""
    fb = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="key-2",
    )
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = MagicMock()
    handler1.generate_image.side_effect = HTTPError(
        "Server error",
        provider="openai",
        model="dall-e-3",
        status_code=500,
    )
    handler2 = MagicMock()
    handler2.generate_image.return_value = _image_response

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        resp = orchestrator.execute_image_sync(config, _image_request)

    assert resp.execution_metadata.total_attempts == 2
    assert resp.execution_metadata.fallback_triggered is True


def test_execute_image_sync_non_retryable_stops(_image_request):
    """Image sync: non-retryable error stops chain."""
    fb = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="key-2",
    )
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler = MagicMock()
    handler.generate_image.side_effect = ValidationError(
        "Bad size",
        provider="openai",
        model="dall-e-3",
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(ValidationError, match="Bad size"):
            orchestrator.execute_image_sync(config, _image_request)


def test_execute_image_sync_all_exhausted(_image_request):
    """Image sync: all providers fail — raises last error."""
    fb = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="key-2",
    )
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = MagicMock()
    handler1.generate_image.side_effect = HTTPError(
        "Err 1",
        provider="openai",
        model="dall-e-3",
        status_code=500,
    )
    handler2 = MagicMock()
    handler2.generate_image.side_effect = HTTPError(
        "Err 2",
        provider="fal",
        model="fal-ai/flux/dev",
        status_code=502,
    )

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(HTTPError, match="Err 2"):
            orchestrator.execute_image_sync(config, _image_request)


def test_execute_image_sync_not_implemented_propagates(_image_config, _image_request):
    """Image sync: NotImplementedError propagates without fallback."""
    handler = MagicMock()
    handler.generate_image.side_effect = NotImplementedError("not supported")

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(NotImplementedError, match="not supported"):
            orchestrator.execute_image_sync(_image_config, _image_request)


# ==================== Audio fallback chain tests ====================


def test_collect_audio_fallback_chain_no_fallbacks():
    """Audio: single config yields chain of length 1."""
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="test-key",
    )
    chain = ExecutionOrchestrator.collect_audio_fallback_chain(config)
    assert len(chain) == 1
    assert chain[0].model == "eleven_multilingual_v2"


def test_collect_audio_fallback_chain_nested():
    """Audio: nested fallbacks are collected depth-first."""
    fb2 = AudioGenerationConfig(
        model="fal-ai/minimax/speech-2.8-hd",
        provider="fal",
        api_key="key-3",
    )
    fb1 = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
        fallback_configs=[fb2],
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb1],
    )
    chain = ExecutionOrchestrator.collect_audio_fallback_chain(config)
    assert len(chain) == 3
    assert [c.model for c in chain] == [
        "eleven_multilingual_v2",
        "sonic-3",
        "fal-ai/minimax/speech-2.8-hd",
    ]


# ==================== TTS generation tests ====================


@pytest.fixture
def _audio_config():
    return AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="test-key",
    )


@pytest.fixture
def _tts_request():
    return TTSRequest(text="Hello world")


@pytest.fixture
def _tts_response():
    return TTSResponse(
        request_id="tts-123",
        audio="base64audiodata",
        status="completed",
        raw_response={"ok": True},
    )


@pytest.fixture
def _sts_request():
    return STSRequest(audio=b"raw-audio-bytes", voice_id="voice-1")


@pytest.fixture
def _sts_response():
    return STSResponse(
        request_id="sts-123",
        audio="base64audiodata",
        status="completed",
        raw_response={"ok": True},
    )


@pytest.mark.asyncio
async def test_execute_tts_async_success(_audio_config, _tts_request, _tts_response):
    """TTS async: success on first attempt attaches metadata."""
    handler = AsyncMock()
    handler.generate_tts_async.return_value = _tts_response

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        resp = await orchestrator.execute_tts_async(_audio_config, _tts_request)

    assert resp.request_id == "tts-123"
    assert resp.execution_metadata is not None
    assert resp.execution_metadata.total_attempts == 1
    assert resp.execution_metadata.fallback_triggered is False


@pytest.mark.asyncio
async def test_execute_tts_async_fallback_on_retryable(_tts_request, _tts_response):
    """TTS async: retryable error triggers fallback."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = AsyncMock()
    handler1.generate_tts_async.side_effect = HTTPError(
        "Server error",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        status_code=500,
    )
    handler2 = AsyncMock()
    handler2.generate_tts_async.return_value = _tts_response

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        resp = await orchestrator.execute_tts_async(config, _tts_request)

    assert resp.execution_metadata.total_attempts == 2
    assert resp.execution_metadata.fallback_triggered is True


@pytest.mark.asyncio
async def test_execute_tts_async_non_retryable_stops(_tts_request):
    """TTS async: non-retryable error stops chain."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler = AsyncMock()
    handler.generate_tts_async.side_effect = ValidationError(
        "Bad voice",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(ValidationError, match="Bad voice"):
            await orchestrator.execute_tts_async(config, _tts_request)


@pytest.mark.asyncio
async def test_execute_tts_async_all_exhausted(_tts_request):
    """TTS async: all providers fail — raises last error."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = AsyncMock()
    handler1.generate_tts_async.side_effect = HTTPError(
        "Err 1",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        status_code=500,
    )
    handler2 = AsyncMock()
    handler2.generate_tts_async.side_effect = HTTPError(
        "Err 2",
        provider="cartesia",
        model="sonic-3",
        status_code=502,
    )

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(HTTPError, match="Err 2"):
            await orchestrator.execute_tts_async(config, _tts_request)


@pytest.mark.asyncio
async def test_execute_tts_async_not_implemented_propagates(
    _audio_config, _tts_request
):
    """TTS async: NotImplementedError propagates without fallback."""
    handler = AsyncMock()
    handler.generate_tts_async.side_effect = NotImplementedError("not supported")

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(NotImplementedError, match="not supported"):
            await orchestrator.execute_tts_async(_audio_config, _tts_request)


def test_execute_tts_sync_success(_audio_config, _tts_request, _tts_response):
    """TTS sync: success on first attempt."""
    handler = MagicMock()
    handler.generate_tts.return_value = _tts_response

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        resp = orchestrator.execute_tts_sync(_audio_config, _tts_request)

    assert resp.request_id == "tts-123"
    assert resp.execution_metadata.total_attempts == 1
    assert resp.execution_metadata.fallback_triggered is False


def test_execute_tts_sync_fallback_on_retryable(_tts_request, _tts_response):
    """TTS sync: retryable error triggers fallback."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = MagicMock()
    handler1.generate_tts.side_effect = HTTPError(
        "Server error",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        status_code=500,
    )
    handler2 = MagicMock()
    handler2.generate_tts.return_value = _tts_response

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        resp = orchestrator.execute_tts_sync(config, _tts_request)

    assert resp.execution_metadata.total_attempts == 2
    assert resp.execution_metadata.fallback_triggered is True


def test_execute_tts_sync_non_retryable_stops(_tts_request):
    """TTS sync: non-retryable error stops chain."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler = MagicMock()
    handler.generate_tts.side_effect = ValidationError(
        "Bad voice",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(ValidationError, match="Bad voice"):
            orchestrator.execute_tts_sync(config, _tts_request)


def test_execute_tts_sync_all_exhausted(_tts_request):
    """TTS sync: all providers fail — raises last error."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = MagicMock()
    handler1.generate_tts.side_effect = HTTPError(
        "Err 1",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        status_code=500,
    )
    handler2 = MagicMock()
    handler2.generate_tts.side_effect = HTTPError(
        "Err 2",
        provider="cartesia",
        model="sonic-3",
        status_code=502,
    )

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(HTTPError, match="Err 2"):
            orchestrator.execute_tts_sync(config, _tts_request)


def test_execute_tts_sync_not_implemented_propagates(_audio_config, _tts_request):
    """TTS sync: NotImplementedError propagates without fallback."""
    handler = MagicMock()
    handler.generate_tts.side_effect = NotImplementedError("not supported")

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(NotImplementedError, match="not supported"):
            orchestrator.execute_tts_sync(_audio_config, _tts_request)


# ==================== STS generation tests ====================


@pytest.mark.asyncio
async def test_execute_sts_async_success(_audio_config, _sts_request, _sts_response):
    """STS async: success on first attempt attaches metadata."""
    handler = AsyncMock()
    handler.generate_sts_async.return_value = _sts_response

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        resp = await orchestrator.execute_sts_async(_audio_config, _sts_request)

    assert resp.request_id == "sts-123"
    assert resp.execution_metadata is not None
    assert resp.execution_metadata.total_attempts == 1
    assert resp.execution_metadata.fallback_triggered is False


@pytest.mark.asyncio
async def test_execute_sts_async_fallback_on_retryable(_sts_request, _sts_response):
    """STS async: retryable error triggers fallback."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = AsyncMock()
    handler1.generate_sts_async.side_effect = HTTPError(
        "Server error",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        status_code=500,
    )
    handler2 = AsyncMock()
    handler2.generate_sts_async.return_value = _sts_response

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        resp = await orchestrator.execute_sts_async(config, _sts_request)

    assert resp.execution_metadata.total_attempts == 2
    assert resp.execution_metadata.fallback_triggered is True


@pytest.mark.asyncio
async def test_execute_sts_async_non_retryable_stops(_sts_request):
    """STS async: non-retryable error stops chain."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler = AsyncMock()
    handler.generate_sts_async.side_effect = ValidationError(
        "Bad voice",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(ValidationError, match="Bad voice"):
            await orchestrator.execute_sts_async(config, _sts_request)


@pytest.mark.asyncio
async def test_execute_sts_async_all_exhausted(_sts_request):
    """STS async: all providers fail — raises last error."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = AsyncMock()
    handler1.generate_sts_async.side_effect = HTTPError(
        "Err 1",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        status_code=500,
    )
    handler2 = AsyncMock()
    handler2.generate_sts_async.side_effect = HTTPError(
        "Err 2",
        provider="cartesia",
        model="sonic-3",
        status_code=502,
    )

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(HTTPError, match="Err 2"):
            await orchestrator.execute_sts_async(config, _sts_request)


@pytest.mark.asyncio
async def test_execute_sts_async_not_implemented_propagates(
    _audio_config, _sts_request
):
    """STS async: NotImplementedError propagates without fallback."""
    handler = AsyncMock()
    handler.generate_sts_async.side_effect = NotImplementedError("not supported")

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(NotImplementedError, match="not supported"):
            await orchestrator.execute_sts_async(_audio_config, _sts_request)


def test_execute_sts_sync_success(_audio_config, _sts_request, _sts_response):
    """STS sync: success on first attempt."""
    handler = MagicMock()
    handler.generate_sts.return_value = _sts_response

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        resp = orchestrator.execute_sts_sync(_audio_config, _sts_request)

    assert resp.request_id == "sts-123"
    assert resp.execution_metadata.total_attempts == 1
    assert resp.execution_metadata.fallback_triggered is False


def test_execute_sts_sync_fallback_on_retryable(_sts_request, _sts_response):
    """STS sync: retryable error triggers fallback."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = MagicMock()
    handler1.generate_sts.side_effect = HTTPError(
        "Server error",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        status_code=500,
    )
    handler2 = MagicMock()
    handler2.generate_sts.return_value = _sts_response

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        resp = orchestrator.execute_sts_sync(config, _sts_request)

    assert resp.execution_metadata.total_attempts == 2
    assert resp.execution_metadata.fallback_triggered is True


def test_execute_sts_sync_non_retryable_stops(_sts_request):
    """STS sync: non-retryable error stops chain."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler = MagicMock()
    handler.generate_sts.side_effect = ValidationError(
        "Bad voice",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
    )

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(ValidationError, match="Bad voice"):
            orchestrator.execute_sts_sync(config, _sts_request)


def test_execute_sts_sync_all_exhausted(_sts_request):
    """STS sync: all providers fail — raises last error."""
    fb = AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="key-2",
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="key-1",
        fallback_configs=[fb],
    )

    handler1 = MagicMock()
    handler1.generate_sts.side_effect = HTTPError(
        "Err 1",
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        status_code=500,
    )
    handler2 = MagicMock()
    handler2.generate_sts.side_effect = HTTPError(
        "Err 2",
        provider="cartesia",
        model="sonic-3",
        status_code=502,
    )

    handlers = [handler1, handler2]
    call_count = 0

    def get_handler_mock(cfg):
        nonlocal call_count
        h = handlers[call_count]
        call_count += 1
        return h

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", side_effect=get_handler_mock
    ):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(HTTPError, match="Err 2"):
            orchestrator.execute_sts_sync(config, _sts_request)


def test_execute_sts_sync_not_implemented_propagates(_audio_config, _sts_request):
    """STS sync: NotImplementedError propagates without fallback."""
    handler = MagicMock()
    handler.generate_sts.side_effect = NotImplementedError("not supported")

    with patch("tarash.tarash_gateway.orchestrator.get_handler", return_value=handler):
        orchestrator = ExecutionOrchestrator()
        with pytest.raises(NotImplementedError, match="not supported"):
            orchestrator.execute_sts_sync(_audio_config, _sts_request)
