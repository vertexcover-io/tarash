"""Tests for ReplicateProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.video.exceptions import (
    GenerationFailedError,
)
from tarash.tarash_gateway.video.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.video.providers.replicate import (
    ReplicateProviderHandler,
    get_replicate_field_mappers,
    parse_replicate_status,
    KLING_V21_FIELD_MAPPERS,
    MINIMAX_FIELD_MAPPERS,
    LUMA_FIELD_MAPPERS,
    WAN_FIELD_MAPPERS,
    GENERIC_REPLICATE_FIELD_MAPPERS,
)


# ==================== Fixtures ====================


@pytest.fixture
def mock_replicate():
    """Patch replicate module and provide mocks."""
    mock_client = MagicMock()
    mock_predictions = MagicMock()
    mock_client.predictions = mock_predictions

    with patch(
        "tarash.tarash_gateway.video.providers.replicate.Client",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.fixture
def handler():
    """Create a ReplicateProviderHandler instance."""
    with patch(
        "tarash.tarash_gateway.video.providers.replicate.replicate", MagicMock()
    ):
        return ReplicateProviderHandler()


@pytest.fixture
def base_config():
    """Create a base VideoGenerationConfig."""
    return VideoGenerationConfig(
        model="kwaivgi/kling-v2.1",
        provider="replicate",
        api_key="test-api-key",
        timeout=600,
        poll_interval=1,
        max_poll_attempts=10,
    )


@pytest.fixture
def base_request():
    """Create a base VideoGenerationRequest."""
    return VideoGenerationRequest(prompt="Test prompt for video generation")


# ==================== Field Mapper Registry Tests ====================


def test_get_replicate_field_mappers_kling_exact():
    """Test exact match for Kling model."""
    mappers = get_replicate_field_mappers("kwaivgi/kling")
    assert mappers is KLING_V21_FIELD_MAPPERS


def test_get_replicate_field_mappers_kling_version():
    """Test prefix match for Kling model with version."""
    mappers = get_replicate_field_mappers("kwaivgi/kling-v2.1:abc123")
    assert mappers is KLING_V21_FIELD_MAPPERS


def test_get_replicate_field_mappers_minimax():
    """Test prefix match for Minimax models."""
    mappers = get_replicate_field_mappers("minimax/video-model")
    assert mappers is MINIMAX_FIELD_MAPPERS


def test_get_replicate_field_mappers_luma():
    """Test prefix match for Luma Dream Machine."""
    mappers = get_replicate_field_mappers("luma/dream-machine")
    assert mappers is LUMA_FIELD_MAPPERS


def test_get_replicate_field_mappers_wan():
    """Test prefix match for Wan (Alibaba) models."""
    mappers = get_replicate_field_mappers("wan-video/some-model")
    assert mappers is WAN_FIELD_MAPPERS


def test_get_replicate_field_mappers_unknown():
    """Test fallback to generic mappers for unknown models."""
    mappers = get_replicate_field_mappers("unknown/model")
    assert mappers is GENERIC_REPLICATE_FIELD_MAPPERS


def test_get_replicate_field_mappers_strips_version():
    """Test that version suffix is stripped before matching."""
    mappers = get_replicate_field_mappers("luma/dream-machine:v1.0.0")
    assert mappers is LUMA_FIELD_MAPPERS


# ==================== Status Parsing Tests ====================


def test_parse_replicate_status_starting():
    """Test parsing 'starting' status."""
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-123"
    mock_prediction.status = "starting"
    mock_prediction.progress = None
    mock_prediction.logs = None
    mock_prediction.error = None

    result = parse_replicate_status(mock_prediction)

    assert result.request_id == "pred-123"
    assert result.status == "queued"
    assert result.update["replicate_status"] == "starting"
    assert result.error is None


def test_parse_replicate_status_processing():
    """Test parsing 'processing' status with logs."""
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-456"
    mock_prediction.status = "processing"
    mock_prediction.progress = None
    mock_prediction.logs = "Step 1/10 complete\nStep 2/10 complete"
    mock_prediction.error = None

    result = parse_replicate_status(mock_prediction)

    assert result.request_id == "pred-456"
    assert result.status == "processing"
    assert "logs" in result.update
    assert result.error is None


def test_parse_replicate_status_succeeded():
    """Test parsing 'succeeded' status."""
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-789"
    mock_prediction.status = "succeeded"
    mock_prediction.progress = None
    mock_prediction.logs = "Complete"
    mock_prediction.error = None

    result = parse_replicate_status(mock_prediction)

    assert result.request_id == "pred-789"
    assert result.status == "completed"


def test_parse_replicate_status_failed():
    """Test parsing 'failed' status with error."""
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-fail"
    mock_prediction.status = "failed"
    mock_prediction.progress = None
    mock_prediction.logs = "Error occurred"
    mock_prediction.error = "Model execution failed"

    result = parse_replicate_status(mock_prediction)

    assert result.request_id == "pred-fail"
    assert result.status == "failed"
    assert result.error == "Model execution failed"


def test_parse_replicate_status_canceled():
    """Test parsing 'canceled' status."""
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-cancel"
    mock_prediction.status = "canceled"
    mock_prediction.progress = None
    mock_prediction.logs = None
    mock_prediction.error = None

    result = parse_replicate_status(mock_prediction)

    assert result.request_id == "pred-cancel"
    assert result.status == "failed"


def test_parse_replicate_status_with_progress():
    """Test parsing status with progress information."""
    mock_progress = MagicMock()
    mock_progress.percentage = 50
    mock_progress.current = 5
    mock_progress.total = 10

    mock_prediction = MagicMock()
    mock_prediction.id = "pred-progress"
    mock_prediction.status = "processing"
    mock_prediction.progress = mock_progress
    mock_prediction.logs = None
    mock_prediction.error = None

    result = parse_replicate_status(mock_prediction)

    assert result.progress_percent == 50
    assert result.update["progress"] == 50
    assert result.update["current"] == 5
    assert result.update["total"] == 10


# ==================== Initialization Tests ====================


def test_init_creates_empty_cache(handler):
    """Test that handler initializes with empty client cache."""
    assert handler._client_cache == {}


def test_init_raises_import_error_without_replicate():
    """Test that ImportError is raised when replicate is not installed."""
    with patch(
        "tarash.tarash_gateway.video.providers.replicate.replicate",
        None,
    ):
        with pytest.raises(ImportError, match="replicate is required"):
            ReplicateProviderHandler()


# ==================== Client Management Tests ====================


def test_get_client_creates_and_caches_client(handler, base_config, mock_replicate):
    """Test client creation and caching."""
    handler._client_cache.clear()

    client1 = handler._get_client(base_config)
    client2 = handler._get_client(base_config)

    assert client1 is client2  # Same instance (cached)


def test_get_client_creates_different_clients_for_different_keys(handler):
    """Test different clients for different API keys."""
    handler._client_cache.clear()

    mock_client1 = MagicMock()
    mock_client2 = MagicMock()

    config1 = VideoGenerationConfig(
        model="kwaivgi/kling-v2.1",
        provider="replicate",
        api_key="key1",
        timeout=600,
    )
    config2 = VideoGenerationConfig(
        model="kwaivgi/kling-v2.1",
        provider="replicate",
        api_key="key2",
        timeout=600,
    )

    with patch(
        "tarash.tarash_gateway.video.providers.replicate.Client",
        side_effect=[mock_client1, mock_client2],
    ):
        client1 = handler._get_client(config1)
        client2 = handler._get_client(config2)

        assert client1 is not client2


# ==================== Request Conversion Tests ====================


def test_convert_request_with_minimal_fields(handler, base_config):
    """Test conversion with only prompt."""
    request = VideoGenerationRequest(prompt="A test video")
    result = handler._convert_request(base_config, request)

    assert result == {"prompt": "A test video"}


def test_convert_request_with_kling_fields(handler):
    """Test conversion with Kling-specific fields."""
    config = VideoGenerationConfig(
        model="kwaivgi/kling-v2.1",
        provider="replicate",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A beautiful sunset",
        duration_seconds=10,
        aspect_ratio="16:9",
        negative_prompt="blurry, distorted",
        seed=42,
        image_list=[{"image": "https://example.com/image.jpg", "type": "reference"}],
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A beautiful sunset"
    assert result["duration"] == 10
    assert result["aspect_ratio"] == "16:9"
    assert result["negative_prompt"] == "blurry, distorted"
    assert result["seed"] == 42
    assert result["image"] == "https://example.com/image.jpg"


def test_convert_request_with_minimax_duration_validation(handler):
    """Test Minimax duration validation (6s or 10s only)."""
    from tarash.tarash_gateway.video.exceptions import ValidationError

    config = VideoGenerationConfig(
        model="minimax/video-model",
        provider="replicate",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="test",
        duration_seconds=6,
    )

    result = handler._convert_request(config, request)
    assert result["duration"] == "6s"

    # Test invalid duration
    request_invalid = VideoGenerationRequest(
        prompt="test",
        duration_seconds=15,  # Invalid
    )

    with pytest.raises(ValidationError, match="Invalid duration"):
        handler._convert_request(config, request_invalid)


def test_convert_request_with_extra_params(handler, base_config, base_request):
    """Test that extra_params are merged into output."""
    request = VideoGenerationRequest(
        prompt="Test video",
        extra_params={"custom_param": "value", "another_param": 123},
    )

    result = handler._convert_request(base_config, request)

    assert result["prompt"] == "Test video"
    assert result["custom_param"] == "value"
    assert result["another_param"] == 123


# ==================== Response Conversion Tests ====================


def test_convert_response_with_string_output(handler, base_config, base_request):
    """Test conversion when output is a string URL."""
    result = handler._convert_response(
        base_config, base_request, "pred-123", "https://example.com/video.mp4"
    )

    assert result.request_id == "pred-123"
    assert result.video == "https://example.com/video.mp4"
    assert result.status == "completed"


def test_convert_response_with_list_output(handler, base_config, base_request):
    """Test conversion when output is a list of URLs."""
    result = handler._convert_response(
        base_config,
        base_request,
        "pred-456",
        ["https://example.com/video1.mp4", "https://example.com/video2.mp4"],
    )

    assert result.request_id == "pred-456"
    assert result.video == "https://example.com/video1.mp4"


def test_convert_response_with_dict_output(handler, base_config, base_request):
    """Test conversion when output is a dict with 'video' key."""
    result = handler._convert_response(
        base_config,
        base_request,
        "pred-789",
        {"video": "https://example.com/video.mp4", "duration": 5.0},
    )

    assert result.video == "https://example.com/video.mp4"


def test_convert_response_with_file_output_object(handler, base_config, base_request):
    """Test conversion when output has a URL attribute (FileOutput-like)."""
    mock_file_output = MagicMock()
    mock_file_output.url = "https://example.com/video.mp4"

    result = handler._convert_response(
        base_config, base_request, "pred-abc", mock_file_output
    )

    assert result.video == "https://example.com/video.mp4"


def test_convert_response_with_none_output_raises_error(
    handler, base_config, base_request
):
    """Test that None output raises GenerationFailedError."""
    with pytest.raises(GenerationFailedError, match="no output was returned"):
        handler._convert_response(base_config, base_request, "pred-null", None)


def test_convert_response_with_invalid_output_raises_error(
    handler, base_config, base_request
):
    """Test that unrecognized output format raises GenerationFailedError."""
    with pytest.raises(GenerationFailedError, match="Could not extract video URL"):
        handler._convert_response(
            base_config, base_request, "pred-invalid", {"unrelated": "data"}
        )


# ==================== Error Handling Tests ====================


def test_handle_error_with_video_generation_error(handler, base_config, base_request):
    """Test GenerationFailedError is returned as-is."""
    error = GenerationFailedError(
        "Test error", provider="replicate", model="test-model"
    )
    result = handler._handle_error(base_config, base_request, "pred-1", error)

    assert result is error


def test_handle_error_with_unknown_exception(handler, base_config, base_request):
    """Test unknown exception is converted to GenerationFailedError."""
    unknown_error = RuntimeError("Something went wrong")
    result = handler._handle_error(base_config, base_request, "pred-2", unknown_error)

    assert isinstance(result, GenerationFailedError)
    assert "Replicate API error" in result.message
    assert result.provider == "replicate"
    assert result.request_id == "pred-2"
    assert result.raw_response["prediction_id"] == "pred-2"


# ==================== Sync Video Generation Tests ====================


def test_generate_video_success_without_progress(handler, base_config, base_request):
    """Test successful sync generation without progress callback."""
    with patch(
        "tarash.tarash_gateway.video.providers.replicate.replicate"
    ) as mock_replicate_module:
        mock_replicate_module.run.return_value = "https://example.com/video.mp4"

        # Also need to patch Client since it's used in _get_client
        with patch("tarash.tarash_gateway.video.providers.replicate.Client"):
            handler._client_cache.clear()
            result = handler.generate_video(base_config, base_request)

            assert result.video == "https://example.com/video.mp4"
            assert result.status == "completed"


def test_generate_video_success_with_progress_callback(
    handler, base_config, base_request, mock_replicate
):
    """Test successful sync generation with progress callback."""
    # Setup mock prediction
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-progress"
    mock_prediction.status = "starting"
    mock_prediction.progress = None
    mock_prediction.logs = None
    mock_prediction.error = None
    mock_prediction.output = "https://example.com/video.mp4"

    # Simulate status changes
    call_count = [0]

    def reload_side_effect():
        call_count[0] += 1
        if call_count[0] == 1:
            mock_prediction.status = "processing"
        else:
            mock_prediction.status = "succeeded"

    mock_prediction.reload = reload_side_effect
    mock_replicate.predictions.create.return_value = mock_prediction

    handler._client_cache.clear()

    progress_calls = []

    def progress_callback(update):
        progress_calls.append(update)

    with patch("time.sleep"):  # Speed up test
        result = handler.generate_video(
            base_config, base_request, on_progress=progress_callback
        )

    assert result.request_id == "pred-progress"
    assert result.video == "https://example.com/video.mp4"
    assert len(progress_calls) >= 1


def test_generate_video_handles_failure(
    handler, base_config, base_request, mock_replicate
):
    """Test handling of failed prediction."""
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-fail"
    mock_prediction.status = "failed"
    mock_prediction.progress = None
    mock_prediction.logs = "Error logs"
    mock_prediction.error = "Model crashed"
    mock_prediction.output = None
    mock_prediction.reload = MagicMock()

    mock_replicate.predictions.create.return_value = mock_prediction

    handler._client_cache.clear()

    with pytest.raises(GenerationFailedError, match="Model crashed"):
        handler.generate_video(base_config, base_request, on_progress=lambda x: None)


def test_generate_video_handles_timeout(handler, base_request, mock_replicate):
    """Test handling of prediction timeout."""
    config = VideoGenerationConfig(
        model="kwaivgi/kling-v2.1",
        provider="replicate",
        api_key="test-key",
        poll_interval=1,
        max_poll_attempts=2,  # Very short for test
    )

    mock_prediction = MagicMock()
    mock_prediction.id = "pred-timeout"
    mock_prediction.status = "processing"
    mock_prediction.progress = None
    mock_prediction.logs = None
    mock_prediction.error = None
    mock_prediction.reload = MagicMock()

    mock_replicate.predictions.create.return_value = mock_prediction

    handler._client_cache.clear()

    with patch("time.sleep"):
        with pytest.raises(GenerationFailedError, match="timed out"):
            handler.generate_video(config, base_request, on_progress=lambda x: None)


# ==================== Async Video Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_video_async_success_without_progress(
    handler, base_config, base_request
):
    """Test successful async generation without progress callback."""
    with patch(
        "tarash.tarash_gateway.video.providers.replicate.replicate"
    ) as mock_replicate_module:
        mock_replicate_module.async_run = AsyncMock(
            return_value="https://example.com/video.mp4"
        )

        result = await handler.generate_video_async(base_config, base_request)

        assert result.video == "https://example.com/video.mp4"
        assert result.status == "completed"


@pytest.mark.asyncio
async def test_generate_video_async_success_with_progress_callback(
    handler, base_config, base_request, mock_replicate
):
    """Test successful async generation with progress callback."""
    # Setup mock prediction
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-async-progress"
    mock_prediction.status = "starting"
    mock_prediction.progress = None
    mock_prediction.logs = None
    mock_prediction.error = None
    mock_prediction.output = "https://example.com/video.mp4"

    # Simulate status changes
    call_count = [0]

    def reload_side_effect():
        call_count[0] += 1
        if call_count[0] == 1:
            mock_prediction.status = "processing"
        else:
            mock_prediction.status = "succeeded"

    mock_prediction.reload = reload_side_effect
    mock_replicate.predictions.create.return_value = mock_prediction

    handler._client_cache.clear()

    progress_calls = []

    async def async_progress_callback(update):
        progress_calls.append(update)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await handler.generate_video_async(
            base_config, base_request, on_progress=async_progress_callback
        )

    assert result.request_id == "pred-async-progress"
    assert result.video == "https://example.com/video.mp4"
    assert len(progress_calls) >= 1


@pytest.mark.asyncio
async def test_generate_video_async_handles_failure(
    handler, base_config, base_request, mock_replicate
):
    """Test handling of failed prediction in async mode."""
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-async-fail"
    mock_prediction.status = "failed"
    mock_prediction.progress = None
    mock_prediction.logs = "Error logs"
    mock_prediction.error = "Async model crashed"
    mock_prediction.output = None
    mock_prediction.reload = MagicMock()

    mock_replicate.predictions.create.return_value = mock_prediction

    handler._client_cache.clear()

    with pytest.raises(GenerationFailedError, match="Async model crashed"):
        await handler.generate_video_async(
            base_config, base_request, on_progress=lambda x: None
        )


@pytest.mark.asyncio
async def test_generate_video_async_wraps_unknown_exceptions(
    handler, base_config, base_request
):
    """Test unknown exceptions are wrapped by decorator in async mode."""
    with patch(
        "tarash.tarash_gateway.video.providers.replicate.replicate"
    ) as mock_replicate_module:
        mock_replicate_module.async_run = AsyncMock(
            side_effect=RuntimeError("Unexpected async error")
        )

        # Error is wrapped by _handle_error which creates "Replicate API error" message
        with pytest.raises(GenerationFailedError, match="Replicate API error"):
            await handler.generate_video_async(base_config, base_request)


# ==================== Integration-style Tests ====================


def test_full_kling_request_conversion(handler):
    """Test complete Kling model request conversion with all fields."""
    config = VideoGenerationConfig(
        model="kwaivgi/kling-v2.1",
        provider="replicate",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A cinematic shot of a dragon flying over mountains",
        duration_seconds=10,
        aspect_ratio="16:9",
        negative_prompt="low quality, blurry",
        seed=12345,
        image_list=[{"image": "https://example.com/dragon.jpg", "type": "reference"}],
        extra_params={"cfg_scale": 0.8},
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A cinematic shot of a dragon flying over mountains"
    assert result["duration"] == 10
    assert result["aspect_ratio"] == "16:9"
    assert result["negative_prompt"] == "low quality, blurry"
    assert result["seed"] == 12345
    assert result["image"] == "https://example.com/dragon.jpg"
    assert result["cfg_scale"] == 0.8


def test_full_luma_request_conversion(handler):
    """Test complete Luma Dream Machine request conversion."""
    config = VideoGenerationConfig(
        model="luma/dream-machine",
        provider="replicate",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A serene ocean sunset",
        aspect_ratio="16:9",
        image_list=[
            {"image": "https://example.com/start.jpg", "type": "first_frame"},
            {"image": "https://example.com/end.jpg", "type": "last_frame"},
        ],
        extra_params={"loop": True},
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A serene ocean sunset"
    assert result["aspect_ratio"] == "16:9"
    assert result["start_image_url"] == "https://example.com/start.jpg"
    assert result["end_image_url"] == "https://example.com/end.jpg"
    assert result["loop"] is True
