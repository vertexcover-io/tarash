"""Tests for ReplicateProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.video.exceptions import (
    GenerationFailedError,
    TimeoutError,
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
    VEO31_FIELD_MAPPERS,
    GENERIC_REPLICATE_FIELD_MAPPERS,
)


# ==================== Fixtures ====================


@pytest.fixture
def mock_replicate():
    """Patch replicate module and provide mocks (v2.0.0 API)."""
    mock_client = MagicMock()
    mock_predictions = MagicMock()
    mock_client.predictions = mock_predictions

    with patch(
        "tarash.tarash_gateway.video.providers.replicate.Replicate",
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


def test_get_replicate_field_mappers_veo3():
    """Test prefix match for Google Veo 3 and 3.1 models (same API)."""
    # Veo 3.1 (prefix match from google/veo-3)
    mappers_v31 = get_replicate_field_mappers("google/veo-3.1")
    assert mappers_v31 is VEO31_FIELD_MAPPERS

    # Veo 3 (exact match)
    mappers_v3 = get_replicate_field_mappers("google/veo-3")
    assert mappers_v3 is VEO31_FIELD_MAPPERS

    # With version tags
    mappers_v31_versioned = get_replicate_field_mappers("google/veo-3.1:abc123")
    assert mappers_v31_versioned is VEO31_FIELD_MAPPERS


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
        "tarash.tarash_gateway.video.providers.replicate.Replicate",
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


def test_convert_request_with_veo31_duration_validation(handler):
    """Test Veo 3.1 duration validation (4, 6, or 8 seconds only)."""
    from tarash.tarash_gateway.video.exceptions import ValidationError

    config = VideoGenerationConfig(
        model="google/veo-3.1",
        provider="replicate",
        api_key="test-key",
    )

    # Test valid durations
    for valid_duration in [4, 6, 8]:
        request_valid = VideoGenerationRequest(
            prompt="test",
            duration_seconds=valid_duration,
        )
        result = handler._convert_request(config, request_valid)
        assert result["duration"] == valid_duration

    # Test invalid duration (5 seconds is not allowed)
    request_invalid = VideoGenerationRequest(
        prompt="test",
        duration_seconds=5,
    )

    with pytest.raises(ValidationError, match="Invalid duration"):
        handler._convert_request(config, request_invalid)


def test_convert_request_with_veo31_first_last_frame(handler):
    """Test Veo 3.1 with first and last frame."""
    config = VideoGenerationConfig(
        model="google/veo-3.1",
        provider="replicate",
        api_key="test-key",
    )

    request = VideoGenerationRequest(
        prompt="A smooth transition",
        duration_seconds=6,
        aspect_ratio="16:9",
        resolution="1080p",
        image_list=[
            {
                "image": "https://example.com/first.jpg",
                "type": "first_frame",
            },
            {
                "image": "https://example.com/last.jpg",
                "type": "last_frame",
            },
        ],
        generate_audio=True,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A smooth transition"
    assert result["duration"] == 6
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "1080p"
    assert result["image"] == "https://example.com/first.jpg"
    assert result["last_frame"] == "https://example.com/last.jpg"
    assert result["generate_audio"] is True


def test_convert_request_with_veo31_reference_images(handler):
    """Test Veo 3.1 with reference images (R2V)."""
    config = VideoGenerationConfig(
        model="google/veo-3.1",
        provider="replicate",
        api_key="test-key",
    )

    request = VideoGenerationRequest(
        prompt="A woman giving a podcast interview",
        duration_seconds=8,
        aspect_ratio="16:9",
        resolution="1080p",
        image_list=[
            {
                "image": "https://example.com/ref1.jpg",
                "type": "reference",
            },
            {
                "image": "https://example.com/ref2.jpg",
                "type": "reference",
            },
            {
                "image": "https://example.com/ref3.jpg",
                "type": "reference",
            },
        ],
        generate_audio=True,
        seed=42,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A woman giving a podcast interview"
    assert result["duration"] == 8
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "1080p"
    assert result["reference_images"] == [
        "https://example.com/ref1.jpg",
        "https://example.com/ref2.jpg",
        "https://example.com/ref3.jpg",
    ]
    assert result["generate_audio"] is True
    assert result["seed"] == 42


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
    """Test successful sync generation without progress callback (v2.0.0 API)."""
    # Mock client with run method
    mock_client = MagicMock()
    mock_client.run.return_value = "https://example.com/video.mp4"

    with patch(
        "tarash.tarash_gateway.video.providers.replicate.Replicate",
        return_value=mock_client,
    ):
        handler._client_cache.clear()
        result = handler.generate_video(base_config, base_request)

        assert result.video == "https://example.com/video.mp4"
        assert result.status == "completed"
        mock_client.run.assert_called_once()


def test_generate_video_success_with_progress_callback(
    handler, base_config, base_request, mock_replicate
):
    """Test successful sync generation with progress callback (v2.0.0 API)."""
    # Setup mock predictions with different states
    mock_prediction_1 = MagicMock()
    mock_prediction_1.id = "pred-progress"
    mock_prediction_1.status = "starting"
    mock_prediction_1.progress = None
    mock_prediction_1.logs = None
    mock_prediction_1.error = None

    mock_prediction_2 = MagicMock()
    mock_prediction_2.id = "pred-progress"
    mock_prediction_2.status = "processing"
    mock_prediction_2.progress = None
    mock_prediction_2.logs = None
    mock_prediction_2.error = None

    mock_prediction_3 = MagicMock()
    mock_prediction_3.id = "pred-progress"
    mock_prediction_3.status = "succeeded"
    mock_prediction_3.progress = None
    mock_prediction_3.logs = None
    mock_prediction_3.error = None
    mock_prediction_3.output = "https://example.com/video.mp4"

    # v2.0.0 API: predictions.create returns initial prediction
    mock_replicate.predictions.create.return_value = mock_prediction_1

    # v2.0.0 API: predictions.get returns updated states
    mock_replicate.predictions.get.side_effect = [
        mock_prediction_2,  # First poll: processing
        mock_prediction_3,  # Second poll: succeeded
    ]

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
    """Test handling of failed prediction (v2.0.0 API)."""
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-fail"
    mock_prediction.status = "failed"
    mock_prediction.progress = None
    mock_prediction.logs = "Error logs"
    mock_prediction.error = "Model crashed"
    mock_prediction.output = None

    mock_replicate.predictions.create.return_value = mock_prediction
    # v2.0.0 API: predictions.get returns the failed prediction
    mock_replicate.predictions.get.return_value = mock_prediction

    handler._client_cache.clear()

    with pytest.raises(GenerationFailedError, match="Model crashed"):
        handler.generate_video(base_config, base_request, on_progress=lambda x: None)


def test_generate_video_handles_timeout(handler, base_request, mock_replicate):
    """Test handling of prediction timeout (v2.0.0 API)."""
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

    mock_replicate.predictions.create.return_value = mock_prediction
    # v2.0.0 API: predictions.get keeps returning processing status
    mock_replicate.predictions.get.return_value = mock_prediction

    handler._client_cache.clear()

    with patch("time.sleep"):
        with pytest.raises(TimeoutError, match="timed out"):
            handler.generate_video(config, base_request, on_progress=lambda x: None)


# ==================== Async Video Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_video_async_success_without_progress(
    handler, base_config, base_request
):
    """Test successful async generation without progress callback (v2.0.0 API)."""
    # Mock AsyncReplicate client
    mock_async_client = AsyncMock()
    mock_async_client.run = AsyncMock(return_value="https://example.com/video.mp4")

    with patch(
        "tarash.tarash_gateway.video.providers.replicate.AsyncReplicate",
        return_value=mock_async_client,
    ):
        result = await handler.generate_video_async(base_config, base_request)

        assert result.video == "https://example.com/video.mp4"
        assert result.status == "completed"
        mock_async_client.run.assert_called_once()


@pytest.mark.asyncio
async def test_generate_video_async_success_with_progress_callback(
    handler, base_config, base_request
):
    """Test successful async generation with progress callback (v2.0.0 API)."""
    # Setup mock predictions with different states
    mock_prediction_1 = MagicMock()
    mock_prediction_1.id = "pred-async-progress"
    mock_prediction_1.status = "starting"
    mock_prediction_1.progress = None
    mock_prediction_1.logs = None
    mock_prediction_1.error = None

    mock_prediction_2 = MagicMock()
    mock_prediction_2.id = "pred-async-progress"
    mock_prediction_2.status = "processing"
    mock_prediction_2.progress = None
    mock_prediction_2.logs = None
    mock_prediction_2.error = None

    mock_prediction_3 = MagicMock()
    mock_prediction_3.id = "pred-async-progress"
    mock_prediction_3.status = "succeeded"
    mock_prediction_3.progress = None
    mock_prediction_3.logs = None
    mock_prediction_3.error = None
    mock_prediction_3.output = "https://example.com/video.mp4"

    # Mock AsyncReplicate client
    mock_async_client = AsyncMock()
    mock_async_client.predictions = AsyncMock()
    mock_async_client.predictions.create = AsyncMock(return_value=mock_prediction_1)
    # v2.0.0 API: predictions.get returns updated states
    mock_async_client.predictions.get = AsyncMock(
        side_effect=[
            mock_prediction_2,  # First poll: processing
            mock_prediction_3,  # Second poll: succeeded
        ]
    )

    progress_calls = []

    async def async_progress_callback(update):
        progress_calls.append(update)

    with patch(
        "tarash.tarash_gateway.video.providers.replicate.AsyncReplicate",
        return_value=mock_async_client,
    ):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await handler.generate_video_async(
                base_config, base_request, on_progress=async_progress_callback
            )

    assert result.request_id == "pred-async-progress"
    assert result.video == "https://example.com/video.mp4"
    assert len(progress_calls) >= 1


@pytest.mark.asyncio
async def test_generate_video_async_handles_failure(handler, base_config, base_request):
    """Test handling of failed prediction in async mode (v2.0.0 API)."""
    mock_prediction = MagicMock()
    mock_prediction.id = "pred-async-fail"
    mock_prediction.status = "failed"
    mock_prediction.progress = None
    mock_prediction.logs = "Error logs"
    mock_prediction.error = "Async model crashed"
    mock_prediction.output = None

    # Mock AsyncReplicate client
    mock_async_client = AsyncMock()
    mock_async_client.predictions = AsyncMock()
    mock_async_client.predictions.create = AsyncMock(return_value=mock_prediction)
    # v2.0.0 API: predictions.get returns the failed prediction
    mock_async_client.predictions.get = AsyncMock(return_value=mock_prediction)

    with patch(
        "tarash.tarash_gateway.video.providers.replicate.AsyncReplicate",
        return_value=mock_async_client,
    ):
        with pytest.raises(GenerationFailedError, match="Async model crashed"):
            await handler.generate_video_async(
                base_config, base_request, on_progress=lambda x: None
            )


@pytest.mark.asyncio
async def test_generate_video_async_wraps_unknown_exceptions(
    handler, base_config, base_request
):
    """Test unknown exceptions are wrapped by decorator in async mode (v2.0.0 API)."""
    # Mock AsyncReplicate client that raises exception
    mock_async_client = AsyncMock()
    mock_async_client.run = AsyncMock(
        side_effect=RuntimeError("Unexpected async error")
    )

    with patch(
        "tarash.tarash_gateway.video.providers.replicate.AsyncReplicate",
        return_value=mock_async_client,
    ):
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
