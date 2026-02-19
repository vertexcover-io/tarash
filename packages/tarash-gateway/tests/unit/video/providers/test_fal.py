"""Tests for FalProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fal_client.client import FalClientHTTPError
# from pydantic import ValidationError as Pydantic ValidationError  # Not used with FieldMapper approach

from tarash.tarash_gateway.exceptions import (
    GenerationFailedError,
    HTTPError,
    TarashException,
    ValidationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.models import (
    ImageGenerationRequest,
    VideoGenerationConfig,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.providers.fal import (
    FalProviderHandler,
    get_field_mappers,
    get_image_field_mappers,
    WAN_VIDEO_GENERATION_MAPPERS,
    WAN_ANIMATE_MAPPERS,
    BYTEDANCE_SEEDANCE_FIELD_MAPPERS,
    PIXVERSE_FIELD_MAPPERS,
    parse_fal_status,
    parse_fal_image_status,
)


# ==================== Fixtures ====================


@pytest.fixture
def mock_sync_client():
    """Patch fal_client.SyncClient and provide mock."""
    mock = MagicMock()
    with patch(
        "tarash.tarash_gateway.providers.fal.fal_client.SyncClient",
        return_value=mock,
    ):
        yield mock


@pytest.fixture
def mock_async_client():
    """Patch fal_client.AsyncClient to return a configurable mock.

    Note: In production, AsyncClient is not cached and creates new instances.
    For testing, we use a single mock that can be configured per test.
    """
    mock = AsyncMock()
    with patch(
        "tarash.tarash_gateway.providers.fal.fal_client.AsyncClient",
        return_value=mock,
    ):
        yield mock


@pytest.fixture
def handler():
    """Create a FalProviderHandler instance."""
    return FalProviderHandler()


@pytest.fixture
def base_config():
    """Create a base VideoGenerationConfig."""
    return VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="test-api-key",
        timeout=600,
    )


@pytest.fixture
def base_request():
    """Create a base VideoGenerationRequest."""
    return VideoGenerationRequest(prompt="Test prompt")


# ==================== Initialization Tests ====================


def test_init_creates_empty_caches(handler):
    """Test that handler initializes with empty client caches.

    Note: AsyncClient is not cached, only SyncClient is cached.
    """
    assert handler._sync_client_cache == {}


# ==================== Client Management Tests ====================


def test_get_client_creates_and_caches_sync_client(
    handler, base_config, mock_sync_client
):
    """Test sync client creation and caching."""
    # Clear cache first
    handler._sync_client_cache.clear()

    client1 = handler._get_client(base_config, "sync")
    client2 = handler._get_client(base_config, "sync")

    assert client1 is client2  # Same instance (cached)
    assert client1 is mock_sync_client


def test_get_client_creates_new_async_client_each_time(handler, base_config):
    """Test async client creates new instance each time (not cached).

    AsyncClient is not cached to avoid "Event Loop closed" errors.
    Each async request gets a fresh client instance.
    """
    with patch(
        "tarash.tarash_gateway.providers.fal.fal_client.AsyncClient"
    ) as mock_constructor:
        # Configure mock to return new instances
        mock_constructor.side_effect = [AsyncMock(), AsyncMock()]

        client1 = handler._get_client(base_config, "async")
        client2 = handler._get_client(base_config, "async")

        # Each call creates a new instance (not cached)
        assert client1 is not client2

        # Verify AsyncClient was called twice
        assert mock_constructor.call_count == 2


@pytest.mark.parametrize(
    "api_key,base_url",
    [
        ("key1", None),
        ("key2", None),
        ("key1", "https://api1.example.com"),
        ("key1", "https://api2.example.com"),
    ],
)
def test_get_client_creates_different_clients_for_different_configs(
    handler, api_key, base_url
):
    """Test different clients for different API keys and base_urls."""
    # Clear cache first
    handler._sync_client_cache.clear()

    mock_client1 = MagicMock()
    mock_client2 = MagicMock()

    config1 = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key=api_key,
        base_url=base_url,
        timeout=600,
    )
    config2 = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="different-key",
        base_url="https://different.example.com",
        timeout=600,
    )

    with patch(
        "tarash.tarash_gateway.providers.fal.fal_client.SyncClient",
        side_effect=[mock_client1, mock_client2],
    ):
        client1 = handler._get_client(config1, "sync")
        client2 = handler._get_client(config2, "sync")

        assert client1 is not client2  # Different instances


# ==================== Request Conversion Tests ====================


def test_convert_request_with_minimal_fields(handler, base_config):
    """Test conversion with only prompt."""
    request = VideoGenerationRequest(prompt="A test video")
    result = handler._convert_request(base_config, request)

    assert result == {"prompt": "A test video"}


def test_convert_request_with_all_optional_fields(handler, base_config):
    """Test conversion with all optional fields and validated model_params."""
    request = VideoGenerationRequest(
        prompt="A test video",
        duration_seconds=6,  # Valid for veo3.1: 4, 6, or 8 seconds
        resolution="1080p",
        aspect_ratio="16:9",
        image_list=[{"image": "https://example.com/image.jpg", "type": "reference"}],
        extra_params={"seed": 42, "generate_audio": True, "auto_fix": True},
    )

    result = handler._convert_request(base_config, request)

    assert result["prompt"] == "A test video"
    assert result["duration"] == "6s"  # Veo3.1 uses string format like "6s"
    assert result["resolution"] == "1080p"
    assert result["aspect_ratio"] == "16:9"
    assert (
        result["image_url"] == "https://example.com/image.jpg"
    )  # Veo3.1 uses image_url, not image_urls
    assert result["seed"] == 42
    assert result["generate_audio"] is True
    assert result["auto_fix"] is True
    # None values should not be included
    assert "negative_prompt" not in result


def test_convert_request_propagates_validation_errors(handler):
    """Test that validation errors propagate from Pydantic models."""
    # Use a model that has specific validation (Minimax with duration limit)
    config = VideoGenerationConfig(
        model="fal-ai/minimax-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="test",
        duration_seconds=15,  # Invalid for Minimax (only supports 6 or 10 seconds)
    )

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(config, request)

    assert "Invalid duration" in str(exc_info.value)
    assert "15 seconds" in str(exc_info.value)
    assert "6, 10" in str(exc_info.value)


def test_convert_request_veo31_duration_validation(handler):
    """Test that veo3.1 validates duration correctly."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast",
        provider="fal",
        api_key="test-key",
    )

    # Test invalid duration (5 seconds is not allowed)
    request_invalid = VideoGenerationRequest(
        prompt="test",
        duration_seconds=5,  # Invalid for veo3.1 (only supports 4, 6, or 8 seconds)
    )

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(config, request_invalid)

    assert "Invalid duration" in str(exc_info.value)
    assert "5 seconds" in str(exc_info.value)
    assert "4, 6, 8" in str(exc_info.value)

    # Test valid durations (4, 6, 8 seconds)
    for valid_duration in [4, 6, 8]:
        request_valid = VideoGenerationRequest(
            prompt="test",
            duration_seconds=valid_duration,
        )
        result = handler._convert_request(config, request_valid)
        assert result["duration"] == f"{valid_duration}s"


def test_convert_request_veo31_first_last_frame(handler):
    """Test that veo3.1 supports first-last-frame-to-video."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/first-last-frame-to-video",
        provider="fal",
        api_key="test-key",
    )

    request = VideoGenerationRequest(
        prompt="A woman looks into the camera",
        duration_seconds=6,
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
        aspect_ratio="16:9",
        resolution="720p",
        generate_audio=True,
        auto_fix=True,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A woman looks into the camera"
    assert result["duration"] == "6s"
    assert result["first_frame_url"] == "https://example.com/first.jpg"
    assert result["last_frame_url"] == "https://example.com/last.jpg"
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "720p"
    assert result["generate_audio"] is True
    assert result["auto_fix"] is True


def test_convert_request_veo31_video_to_video(handler):
    """Test that veo3.1 supports video-to-video (extend-video)."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/extend-video",
        provider="fal",
        api_key="test-key",
    )

    request = VideoGenerationRequest(
        prompt="Continue the scene naturally, maintaining the same style and motion",
        video="https://example.com/input-video.mp4",
        aspect_ratio="16:9",
        resolution="720p",
        generate_audio=True,
        auto_fix=False,
    )

    result = handler._convert_request(config, request)

    assert (
        result["prompt"]
        == "Continue the scene naturally, maintaining the same style and motion"
    )
    assert result["video_url"] == "https://example.com/input-video.mp4"
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "720p"
    assert result["generate_audio"] is True
    assert result["auto_fix"] is False


def test_convert_request_veo31_fast_extend_video(handler):
    """Test fal-ai/veo3.1/fast/extend-video with specific API requirements."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/extend-video",
        provider="fal",
        api_key="test-key",
    )

    request = VideoGenerationRequest(
        prompt="Continue the scene naturally, maintaining the same style and motion",
        video="https://example.com/input-video.mp4",
        aspect_ratio="16:9",
        duration_seconds=7,
        resolution="720p",
        generate_audio=True,
        auto_fix=False,
    )

    result = handler._convert_request(config, request)

    assert (
        result["prompt"]
        == "Continue the scene naturally, maintaining the same style and motion"
    )
    assert result["video_url"] == "https://example.com/input-video.mp4"
    assert result["aspect_ratio"] == "16:9"
    assert result["duration"] == "7s"
    assert result["resolution"] == "720p"
    assert result["generate_audio"] is True
    assert result["auto_fix"] is False


def test_convert_request_veo31_fast_extend_video_auto_aspect_ratio(handler):
    """Test extend-video with auto aspect ratio via extra_params."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/extend-video",
        provider="fal",
        api_key="test-key",
    )

    # Use extra_params to pass "auto" aspect_ratio since it's not in the standard enum
    request = VideoGenerationRequest(
        prompt="Continue video",
        video="https://example.com/input.mp4",
        duration_seconds=7,
        extra_params={"aspect_ratio": "auto"},
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Continue video"
    assert result["video_url"] == "https://example.com/input.mp4"
    assert result["aspect_ratio"] == "auto"  # Should come from extra_params
    assert result["duration"] == "7s"


def test_convert_request_veo31_fast_extend_video_9_16(handler):
    """Test extend-video with 9:16 aspect ratio."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/extend-video",
        provider="fal",
        api_key="test-key",
    )

    request = VideoGenerationRequest(
        prompt="Continue vertical video",
        video="https://example.com/vertical.mp4",
        aspect_ratio="9:16",
        duration_seconds=7,
    )

    result = handler._convert_request(config, request)

    assert result["aspect_ratio"] == "9:16"
    assert result["video_url"] == "https://example.com/vertical.mp4"


def test_convert_request_veo31_fast_extend_video_without_video(handler):
    """Test that extend-video works without video_url (validation happens at API level).

    Note: The video_url field is required by the Fal API, but we don't enforce this
    at the field mapper level since VEO3_FIELD_MAPPERS is shared across all variants.
    The API will return an error if video_url is missing.
    """
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/extend-video",
        provider="fal",
        api_key="test-key",
    )

    # Request without video - should map successfully but API will reject it
    request_no_video = VideoGenerationRequest(
        prompt="Continue video",
        duration_seconds=7,
    )

    result = handler._convert_request(config, request_no_video)

    # Should successfully map the fields we have
    assert result["prompt"] == "Continue video"
    assert result["duration"] == "7s"
    # video_url should not be in the result (will cause API error)
    assert "video_url" not in result


def test_convert_request_sora2_remix(handler):
    """Test Sora 2 video-to-video/remix endpoint."""
    config = VideoGenerationConfig(
        model="fal-ai/sora-2/video-to-video/remix",
        provider="fal",
        api_key="test-key",
    )

    request = VideoGenerationRequest(
        prompt="Change the cat's fur color to purple",
        extra_params={"video_id": "video_123"},
        delete_video=True,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Change the cat's fur color to purple"
    assert result["video_id"] == "video_123"
    assert result["delete_video"] is True
    # Should not include text-to-video fields
    assert "aspect_ratio" not in result
    assert "resolution" not in result
    assert "duration" not in result


def test_convert_request_sora2_text_to_video(handler):
    """Test Sora 2 text-to-video with all parameters."""
    config = VideoGenerationConfig(
        model="fal-ai/sora-2/text-to-video",
        provider="fal",
        api_key="test-key",
    )

    request = VideoGenerationRequest(
        prompt="A cat playing with a ball of yarn",
        aspect_ratio="16:9",
        resolution="1080p",
        duration_seconds=8,
        delete_video=False,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A cat playing with a ball of yarn"
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "1080p"
    assert result["duration"] == 8
    assert result["delete_video"] is False


# ==================== Response Conversion Tests ====================


def test_convert_response_with_complete_dict_response(
    handler, base_config, base_request
):
    """Test conversion with complete dict response including all fields."""
    provider_response = {
        "video": {"url": "https://example.com/video.mp4"},
        "audio": {"url": "https://example.com/audio.mp3"},
        "duration": 5.5,
        "resolution": "1080p",
        "aspect_ratio": "16:9",
    }

    result = handler._convert_response(
        base_config, base_request, "req-123", provider_response
    )

    assert result.request_id == "req-123"
    assert result.video == "https://example.com/video.mp4"
    assert result.audio_url == "https://example.com/audio.mp3"
    assert result.duration == 5.5
    assert result.resolution == "1080p"
    assert result.aspect_ratio == "16:9"
    assert result.status == "completed"
    assert result.raw_response == provider_response
    assert result.provider_metadata == {}


def test_convert_response_with_video_url_format(handler, base_config, base_request):
    """Test conversion with video_url format (flat structure)."""
    provider_response = {
        "video_url": "https://example.com/video.mp4",
        "duration": 5.5,
    }

    result = handler._convert_response(
        base_config, base_request, "req-123", provider_response
    )

    assert result.video == "https://example.com/video.mp4"
    assert result.request_id == "req-123"
    assert result.duration == 5.5
    assert result.status == "completed"
    assert result.raw_response == provider_response


def test_convert_response_with_audio_url_format(handler, base_config, base_request):
    """Test conversion with audio_url format (flat structure)."""
    provider_response = {
        "video": {"url": "https://example.com/video.mp4"},
        "audio_url": "https://example.com/audio.mp3",
    }

    result = handler._convert_response(
        base_config, base_request, "req-123", provider_response
    )

    assert result.video == "https://example.com/video.mp4"
    assert result.audio_url == "https://example.com/audio.mp3"
    assert result.status == "completed"


def test_convert_response_with_both_video_and_audio_nested_format(
    handler, base_config, base_request
):
    """Test conversion with both video.url and audio.url (nested structure)."""
    provider_response = {
        "video": {"url": "https://example.com/video.mp4"},
        "audio": {"url": "https://example.com/audio.mp3"},
        "duration": 8.0,
    }

    result = handler._convert_response(
        base_config, base_request, "req-456", provider_response
    )

    assert result.video == "https://example.com/video.mp4"
    assert result.audio_url == "https://example.com/audio.mp3"
    assert result.duration == 8.0


def test_convert_response_with_both_video_and_audio_flat_format(
    handler, base_config, base_request
):
    """Test conversion with both video_url and audio_url (flat structure)."""
    provider_response = {
        "video_url": "https://example.com/video.mp4",
        "audio_url": "https://example.com/audio.mp3",
        "resolution": "1080p",
        "aspect_ratio": "16:9",
    }

    result = handler._convert_response(
        base_config, base_request, "req-789", provider_response
    )

    assert result.video == "https://example.com/video.mp4"
    assert result.audio_url == "https://example.com/audio.mp3"
    assert result.resolution == "1080p"
    assert result.aspect_ratio == "16:9"


def test_convert_response_with_video_as_non_dict(handler, base_config, base_request):
    """Test conversion when 'video' key exists but is not a dict (edge case).

    Should fall through to video_url check.
    """
    provider_response = {
        "video": "some_string",  # Not a dict
        "video_url": "https://example.com/video.mp4",
    }

    result = handler._convert_response(
        base_config, base_request, "req-999", provider_response
    )

    assert result.video == "https://example.com/video.mp4"
    assert result.status == "completed"


def test_convert_response_with_audio_as_non_dict(handler, base_config, base_request):
    """Test conversion when 'audio' key exists but is not a dict (edge case).

    Should fall through to audio_url check.
    """
    provider_response = {
        "video": {"url": "https://example.com/video.mp4"},
        "audio": "some_string",  # Not a dict
        "audio_url": "https://example.com/audio.mp3",
    }

    result = handler._convert_response(
        base_config, base_request, "req-888", provider_response
    )

    assert result.video == "https://example.com/video.mp4"
    assert result.audio_url == "https://example.com/audio.mp3"


def test_convert_response_includes_all_optional_fields(
    handler, base_config, base_request
):
    """Test that all optional fields are extracted correctly."""
    provider_response = {
        "video": {"url": "https://example.com/video.mp4"},
        "audio": {"url": "https://example.com/audio.mp3"},
        "duration": 10.5,
        "resolution": "4k",
        "aspect_ratio": "21:9",
        "extra_field": "ignored",  # Should be preserved in raw_response
    }

    result = handler._convert_response(
        base_config, base_request, "req-complete", provider_response
    )

    assert result.video == "https://example.com/video.mp4"
    assert result.audio_url == "https://example.com/audio.mp3"
    assert result.duration == 10.5
    assert result.resolution == "4k"
    assert result.aspect_ratio == "21:9"
    assert result.request_id == "req-complete"
    assert result.status == "completed"
    assert result.provider_metadata == {}
    assert result.raw_response == provider_response
    assert result.raw_response["extra_field"] == "ignored"


def test_convert_response_with_missing_video_url_raises_error(
    handler, base_config, base_request
):
    """Test missing video URL raises GenerationFailedError with proper error message and raw_response."""
    provider_response = {"some_field": "value"}

    with pytest.raises(GenerationFailedError, match="No video URL found") as exc_info:
        handler._convert_response(
            base_config, base_request, "req-789", provider_response
        )

    assert exc_info.value.provider == "fal"
    assert exc_info.value.raw_response == provider_response

    # Test with non-dict response (empty dict after extraction)
    with pytest.raises(GenerationFailedError, match="No video URL found"):
        handler._convert_response(base_config, base_request, "req-789", {})


def test_convert_response_with_video_dict_but_no_url_key(
    handler, base_config, base_request
):
    """Test when video is a dict but doesn't have 'url' key."""
    provider_response = {
        "video": {"other_key": "value"},  # Dict but no 'url'
        # No video_url either
    }

    with pytest.raises(GenerationFailedError, match="No video URL found") as exc_info:
        handler._convert_response(
            base_config, base_request, "req-no-url", provider_response
        )

    assert exc_info.value.provider == "fal"
    assert exc_info.value.model == base_config.model
    assert exc_info.value.raw_response == provider_response


def test_convert_response_preserves_raw_response(handler, base_config, base_request):
    """Test that raw_response is always preserved in the response."""
    provider_response = {
        "video_url": "https://example.com/video.mp4",
        "internal_id": "fal-12345",
        "processing_time": 45.2,
        "nested": {"data": "value"},
    }

    result = handler._convert_response(
        base_config, base_request, "req-preserve", provider_response
    )

    assert result.raw_response == provider_response
    assert result.raw_response["internal_id"] == "fal-12345"
    assert result.raw_response["processing_time"] == 45.2
    assert result.raw_response["nested"]["data"] == "value"


# ==================== Error Handling Tests ====================


def test_handle_error_with_video_generation_error(handler, base_config, base_request):
    """Test TarashException is returned as-is."""
    error = TarashException("Test error", provider="fal", model="test-model")
    result = handler._handle_error(base_config, base_request, "req-1", error)

    assert result is error


def test_handle_error_with_fal_client_http_error(handler, base_config, base_request):
    """Test FalClientHTTPError is converted to TarashException."""
    mock_response = MagicMock()
    mock_response.content = b"Error response"

    http_error = FalClientHTTPError(
        status_code=500,
        message="Internal Server Error",
        response_headers={"Content-Type": "application/json"},
        response=mock_response,
    )

    result = handler._handle_error(base_config, base_request, "req-2", http_error)

    assert isinstance(result, HTTPError)
    assert "500" in result.message
    assert result.provider == "fal"
    assert result.raw_response["status_code"] == 500
    assert result.raw_response["response_headers"] == {
        "Content-Type": "application/json"
    }


def test_handle_error_with_unknown_exception(handler, base_config, base_request):
    """Test unknown exception is converted to GenerationFailedError."""
    unknown_error = ValueError("Something went wrong")

    result = handler._handle_error(base_config, base_request, "req-3", unknown_error)
    assert isinstance(result, GenerationFailedError)
    assert "Error while generating video" in str(result)


# ==================== Status Parsing Tests ====================


def test_parse_fal_status_completed():
    """Test parsing Completed status with metrics and logs."""

    # Create mock classes that will pass isinstance checks
    class MockCompleted:
        def __init__(self):
            self.metrics = {"duration": 5.0}
            self.logs = ["Log entry 1", "Log entry 2"]

    mock_status = MockCompleted()

    # Patch the Completed import in the fal module
    with patch("tarash.tarash_gateway.providers.fal.Completed", MockCompleted):
        result = parse_fal_status("req-1", mock_status)

    assert result.request_id == "req-1"
    assert result.status == "completed"
    assert result.update["metrics"] == {"duration": 5.0}
    assert result.update["logs"] == ["Log entry 1", "Log entry 2"]


def test_parse_fal_status_queued():
    """Test parsing Queued status with position."""

    class MockQueued:
        def __init__(self):
            self.position = 3

    mock_status = MockQueued()

    with patch("tarash.tarash_gateway.providers.fal.Queued", MockQueued):
        result = parse_fal_status("req-2", mock_status)

    assert result.request_id == "req-2"
    assert result.status == "queued"
    assert result.update["position"] == 3


def test_parse_fal_status_in_progress():
    """Test parsing InProgress status with logs."""

    class MockInProgress:
        def __init__(self):
            self.logs = ["Processing...", "Almost done"]

    mock_status = MockInProgress()

    with patch("tarash.tarash_gateway.providers.fal.InProgress", MockInProgress):
        result = parse_fal_status("req-3", mock_status)

    assert result.request_id == "req-3"
    assert result.status == "processing"
    assert result.update == {"logs": ["Processing...", "Almost done"]}


def test_parse_fal_status_unknown_raises_error():
    """Test parsing unknown status raises ValueError."""
    unknown_status = MagicMock()

    with pytest.raises(ValueError, match="Unknown status"):
        parse_fal_status("req-4", unknown_status)


# ==================== Image Status Parsing Tests ====================


def test_parse_fal_image_status_completed():
    """Test parsing Completed status for image generation."""

    class MockCompleted:
        def __init__(self):
            self.metrics = {"duration": 3.2}
            self.logs = ["Image generated successfully"]

    mock_status = MockCompleted()

    with patch("tarash.tarash_gateway.providers.fal.Completed", MockCompleted):
        result = parse_fal_image_status("req-img-1", mock_status)

    assert result.request_id == "req-img-1"
    assert result.status == "completed"
    assert result.progress_percent == 100
    assert result.update["metrics"] == {"duration": 3.2}
    assert result.update["logs"] == ["Image generated successfully"]


def test_parse_fal_image_status_queued():
    """Test parsing Queued status for image generation."""

    class MockQueued:
        def __init__(self):
            self.position = 5

    mock_status = MockQueued()

    with patch("tarash.tarash_gateway.providers.fal.Queued", MockQueued):
        result = parse_fal_image_status("req-img-2", mock_status)

    assert result.request_id == "req-img-2"
    assert result.status == "queued"
    assert result.progress_percent is None
    assert result.update["position"] == 5


def test_parse_fal_image_status_in_progress():
    """Test parsing InProgress status for image generation."""

    class MockInProgress:
        def __init__(self):
            self.logs = ["Generating image...", "Processing complete"]

    mock_status = MockInProgress()

    with patch("tarash.tarash_gateway.providers.fal.InProgress", MockInProgress):
        result = parse_fal_image_status("req-img-3", mock_status)

    assert result.request_id == "req-img-3"
    assert result.status == "processing"
    assert result.progress_percent is None
    assert result.update["logs"] == ["Generating image...", "Processing complete"]


def test_parse_fal_image_status_unknown_raises_error():
    """Test parsing unknown status raises ValueError for image generation."""
    unknown_status = MagicMock()

    with pytest.raises(ValueError, match="Unknown status"):
        parse_fal_image_status("req-img-4", unknown_status)


# ==================== Async Video Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_video_async_success_with_progress_callbacks(
    handler, base_config, base_request
):
    """Test successful async generation with sync and async progress callbacks."""

    # Create mock status classes
    class MockQueued:
        def __init__(self):
            self.position = 1

    class MockInProgress:
        def __init__(self):
            self.logs = ["Processing"]

    class MockCompleted:
        def __init__(self):
            self.metrics = {}
            self.logs = []

    # Setup mock handler
    mock_handler = AsyncMock()
    mock_handler.request_id = "fal-req-123"

    # Mock iter_events to yield status updates
    async def mock_iter_events(with_logs, interval=None):
        yield MockQueued()
        yield MockInProgress()
        yield MockCompleted()

    mock_handler.iter_events = mock_iter_events
    mock_handler.get = AsyncMock(
        return_value={"video": {"url": "https://example.com/video.mp4"}}
    )

    # Setup mock async client
    mock_async_client = AsyncMock()
    mock_async_client.submit = AsyncMock(return_value=mock_handler)

    # Patch AsyncClient (not cached)
    with (
        patch(
            "tarash.tarash_gateway.providers.fal.fal_client.AsyncClient",
            return_value=mock_async_client,
        ),
        patch("tarash.tarash_gateway.providers.fal.Queued", MockQueued),
        patch("tarash.tarash_gateway.providers.fal.InProgress", MockInProgress),
        patch("tarash.tarash_gateway.providers.fal.Completed", MockCompleted),
    ):
        # Test with sync callback
        progress_calls = []

        def sync_callback(update):
            progress_calls.append(update)

        result = await handler.generate_video_async(
            base_config, base_request, on_progress=sync_callback
        )

        mock_async_client.submit.assert_called_once()
        call_args = mock_async_client.submit.call_args
        assert call_args.args[0] == "fal-ai/veo3.1"
        assert call_args.kwargs["arguments"]["prompt"] == "Test prompt"

        assert result.request_id == "fal-req-123"
        assert result.video == "https://example.com/video.mp4"
        assert len(progress_calls) == 3

        mock_async_client.submit.reset_mock()
        async_progress_calls = []

        async def async_callback(update):
            async_progress_calls.append(update)

        await handler.generate_video_async(
            base_config, base_request, on_progress=async_callback
        )

        assert len(async_progress_calls) == 3


@pytest.mark.asyncio
async def test_generate_video_async_propagates_known_errors(
    handler, base_config, base_request, mock_async_client
):
    """Test that ValidationError and GenerationFailedError propagate without wrapping."""
    # Test Pydantic ValidationError propagation from model validation
    # Use Minimax model with duration that exceeds limits
    minimax_config = VideoGenerationConfig(
        model="fal-ai/minimax-video",
        provider="fal",
        api_key="test-key",
    )
    request_invalid = VideoGenerationRequest(
        prompt="test",
        duration_seconds=15,  # Invalid for Minimax (only supports 6 or 10 seconds)
    )

    with pytest.raises(TarashException) as exc_info:
        await handler.generate_video_async(minimax_config, request_invalid)

    assert "Invalid duration" in str(exc_info.value)
    assert "15 seconds" in str(exc_info.value)

    mock_handler = AsyncMock()
    mock_handler.request_id = "req-1"

    # iter_events needs to be an async generator
    async def empty_iter_events(with_logs, interval=None):
        return
        yield  # Make it a generator

    mock_handler.iter_events = empty_iter_events
    mock_handler.get = AsyncMock(return_value={})  # No video URL
    mock_async_client.submit = AsyncMock(return_value=mock_handler)

    with pytest.raises(GenerationFailedError):
        await handler.generate_video_async(base_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_handles_fal_http_error(
    handler, base_config, base_request, mock_async_client
):
    """Test FalClientHTTPError is handled by _handle_error."""
    mock_response = MagicMock()
    mock_response.content = b"Error"

    http_error = FalClientHTTPError(
        status_code=429,
        message="Rate limited",
        response_headers={},
        response=mock_response,
    )

    mock_async_client.submit = AsyncMock(side_effect=http_error)

    with pytest.raises(TarashException, match="Unknown error"):
        await handler.generate_video_async(base_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_wraps_unknown_exceptions(
    handler, base_config, base_request, mock_async_client
):
    """Test unknown exceptions are wrapped by decorator."""
    mock_async_client.submit = AsyncMock(side_effect=RuntimeError("Unexpected error"))

    with pytest.raises(TarashException, match="Unknown error"):
        await handler.generate_video_async(base_config, base_request)


# ==================== Sync Video Generation Tests ====================


def test_generate_video_success_with_progress_callback(
    handler, base_config, base_request, mock_sync_client
):
    """Test successful sync generation with progress callback."""

    # Create mock status classes
    class MockQueued:
        def __init__(self):
            self.position = 1

    class MockCompleted:
        def __init__(self):
            self.metrics = {}
            self.logs = []

    mock_handler = MagicMock()
    mock_handler.request_id = "fal-req-456"

    def mock_iter_events(with_logs, interval=None):
        yield MockQueued()
        yield MockCompleted()

    mock_handler.iter_events = mock_iter_events
    mock_handler.get.return_value = {"video": {"url": "https://example.com/video.mp4"}}

    progress_calls = []

    def progress_callback(update):
        progress_calls.append(update)

    # Clear cache and patch
    handler._sync_client_cache.clear()
    mock_sync_client.submit.return_value = mock_handler
    with (
        patch("tarash.tarash_gateway.providers.fal.Queued", MockQueued),
        patch("tarash.tarash_gateway.providers.fal.Completed", MockCompleted),
    ):
        result = handler.generate_video(
            base_config, base_request, on_progress=progress_callback
        )

    mock_sync_client.submit.assert_called_once()
    call_args = mock_sync_client.submit.call_args
    assert call_args.args[0] == "fal-ai/veo3.1"
    assert call_args.kwargs["arguments"]["prompt"] == "Test prompt"

    assert result.request_id == "fal-req-456"
    assert result.video == "https://example.com/video.mp4"
    assert len(progress_calls) == 2


def test_generate_video_handles_exceptions(
    handler, base_config, base_request, mock_sync_client
):
    """Test exception handling in sync generation."""
    mock_response = MagicMock()
    mock_response.content = b"Error"

    http_error = FalClientHTTPError(
        status_code=500,
        message="Server error",
        response_headers={},
        response=mock_response,
    )

    mock_sync_client.submit.side_effect = http_error

    handler._sync_client_cache.clear()
    with pytest.raises(TarashException, match="Unknown error"):
        handler.generate_video(base_config, base_request)


# ==================== Wan Field Mapper Selection Tests ====================


def test_get_field_mappers_wan_v26_all_endpoints():
    """Test unified mapper for all Wan v2.6 endpoints."""
    # Text-to-video
    assert get_field_mappers("wan/v2.6/text-to-video") is WAN_VIDEO_GENERATION_MAPPERS
    # Image-to-video
    assert get_field_mappers("wan/v2.6/image-to-video") is WAN_VIDEO_GENERATION_MAPPERS
    # Reference-to-video
    assert (
        get_field_mappers("wan/v2.6/reference-to-video") is WAN_VIDEO_GENERATION_MAPPERS
    )


def test_get_field_mappers_wan_v25_all_endpoints():
    """Test unified mapper for all Wan v2.5 endpoints."""
    # Text-to-video
    assert (
        get_field_mappers("fal-ai/wan-25-preview/text-to-video")
        is WAN_VIDEO_GENERATION_MAPPERS
    )
    # Image-to-video
    assert (
        get_field_mappers("fal-ai/wan-25-preview/image-to-video")
        is WAN_VIDEO_GENERATION_MAPPERS
    )


def test_get_field_mappers_wan_v22_animate():
    """Test Wan v2.2-14b animate/move mapper."""
    mappers = get_field_mappers("fal-ai/wan/v2.2-14b/animate/move")
    assert mappers is WAN_ANIMATE_MAPPERS


# ==================== Wan Request Conversion Tests ====================


def test_wan_v26_text_to_video_conversion(handler):
    """Test Wan v2.6 text-to-video request conversion."""
    config = VideoGenerationConfig(
        model="wan/v2.6/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A fox director making a movie",
        duration_seconds=10,
        aspect_ratio="16:9",
        resolution="1080p",
        seed=42,
        enhance_prompt=True,
        extra_params={
            "multi_shots": True,
            "audio_url": "https://example.com/audio.mp3",
        },
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A fox director making a movie"
    assert result["duration"] == "10"  # String format without 's' suffix
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "1080p"
    assert result["seed"] == 42
    assert result["enable_prompt_expansion"] is True  # Mapped from enhance_prompt
    assert result["multi_shots"] is True
    assert result["audio_url"] == "https://example.com/audio.mp3"


def test_wan_v26_image_to_video_conversion(handler):
    """Test Wan v2.6 image-to-video request conversion."""
    config = VideoGenerationConfig(
        model="wan/v2.6/image-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Dragon warrior walking",
        image_list=[{"image": "https://example.com/dragon.jpg", "type": "reference"}],
        duration_seconds=15,
        negative_prompt="low quality, blurry",
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Dragon warrior walking"
    assert result["image_url"] == "https://example.com/dragon.jpg"
    assert result["duration"] == "15"
    assert result["negative_prompt"] == "low quality, blurry"


def test_wan_v26_reference_to_video_conversion(handler):
    """Test Wan v2.6 reference-to-video request conversion with video_urls."""
    config = VideoGenerationConfig(
        model="wan/v2.6/reference-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Dance battle between @Video1 and @Video2",
        duration_seconds=5,
        extra_params={
            "video_urls": [
                "https://example.com/video1.mp4",
                "https://example.com/video2.mp4",
            ]
        },
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Dance battle between @Video1 and @Video2"
    assert result["duration"] == "5"
    assert result["video_urls"] == [
        "https://example.com/video1.mp4",
        "https://example.com/video2.mp4",
    ]


def test_wan_v25_text_to_video_conversion(handler):
    """Test Wan v2.5 text-to-video request conversion (uses same mapper as v2.6)."""
    config = VideoGenerationConfig(
        model="fal-ai/wan-25-preview/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A white dragon warrior",
        duration_seconds=10,
        aspect_ratio="16:9",
        resolution="720p",
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A white dragon warrior"
    assert result["duration"] == "10"
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "720p"


def test_wan_v22_animate_move_conversion(handler):
    """Test Wan v2.2-14b animate/move conversion with video+image inputs."""
    config = VideoGenerationConfig(
        model="fal-ai/wan/v2.2-14b/animate/move",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="",  # Auto-generated by model
        video="https://example.com/input-video.mp4",
        image_list=[{"image": "https://example.com/style.jpg", "type": "reference"}],
        resolution="720p",
        seed=123,
        extra_params={
            "shift": 8,
            "guidance_scale": 1.5,
            "num_inference_steps": 20,
            "use_turbo": True,
            "video_quality": "high",
            "video_write_mode": "balanced",
        },
    )

    result = handler._convert_request(config, request)

    assert result["video_url"] == "https://example.com/input-video.mp4"
    assert result["image_url"] == "https://example.com/style.jpg"
    assert result["resolution"] == "720p"
    assert result["seed"] == 123
    assert result["shift"] == 8
    assert result["guidance_scale"] == 1.5
    assert result["num_inference_steps"] == 20
    assert result["use_turbo"] is True
    assert result["video_quality"] == "high"
    assert result["video_write_mode"] == "balanced"


# ==================== ByteDance Seedance Field Mapper Selection Tests ====================


def test_get_field_mappers_bytedance_seedance():
    """Test ByteDance Seedance v1.5 Pro mapper selection."""
    # Full model path
    assert (
        get_field_mappers("fal-ai/bytedance/seedance/v1.5/pro/text-to-video")
        is BYTEDANCE_SEEDANCE_FIELD_MAPPERS
    )
    # Prefix match
    assert (
        get_field_mappers("fal-ai/bytedance/seedance")
        is BYTEDANCE_SEEDANCE_FIELD_MAPPERS
    )


# ==================== ByteDance Seedance Request Conversion Tests ====================


def test_bytedance_seedance_minimal_conversion(handler):
    """Test ByteDance Seedance with minimal parameters (prompt only)."""
    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1.5/pro/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Defense attorney declaring 'Ladies and gentlemen, reasonable doubt isn't just a phrase'",
    )

    result = handler._convert_request(config, request)

    assert (
        result["prompt"]
        == "Defense attorney declaring 'Ladies and gentlemen, reasonable doubt isn't just a phrase'"
    )
    # No other fields should be present
    assert "duration" not in result
    assert "aspect_ratio" not in result
    assert "resolution" not in result


def test_bytedance_seedance_full_conversion(handler):
    """Test ByteDance Seedance with all parameters."""
    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1.5/pro/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A cinematic courtroom drama scene",
        duration_seconds=10,
        aspect_ratio="16:9",
        resolution="720p",
        seed=42,
        generate_audio=True,
        extra_params={"camera_fixed": True, "enable_safety_checker": False},
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A cinematic courtroom drama scene"
    assert result["duration"] == "10"  # String format, no suffix
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "720p"
    assert result["seed"] == 42
    assert result["generate_audio"] is True
    assert result["camera_fixed"] is True
    assert result["enable_safety_checker"] is False


def test_bytedance_seedance_duration_validation(handler):
    """Test ByteDance Seedance duration validation (4-12 seconds)."""
    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1.5/pro/text-to-video",
        provider="fal",
        api_key="test-key",
    )

    # Test all valid durations (2-12 seconds for unified mapper - covers both v1 and v1.5)
    for valid_duration in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        request = VideoGenerationRequest(
            prompt="test",
            duration_seconds=valid_duration,
        )
        result = handler._convert_request(config, request)
        assert result["duration"] == str(valid_duration)

    # Test invalid durations
    for invalid_duration in [1, 13, 15, 20]:
        request_invalid = VideoGenerationRequest(
            prompt="test",
            duration_seconds=invalid_duration,
        )
        with pytest.raises(ValidationError) as exc_info:
            handler._convert_request(config, request_invalid)

        assert "Invalid duration" in str(exc_info.value)
        assert f"{invalid_duration} seconds" in str(exc_info.value)
        assert "2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12" in str(exc_info.value)


def test_bytedance_seedance_aspect_ratio_options(handler):
    """Test ByteDance Seedance supports various aspect ratios."""
    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1.5/pro/text-to-video",
        provider="fal",
        api_key="test-key",
    )

    # Test aspect ratios supported by VideoGenerationRequest
    # Note: ByteDance API supports "3:4" but VideoGenerationRequest doesn't include it
    for aspect_ratio in ["21:9", "16:9", "4:3", "1:1", "9:16"]:
        request = VideoGenerationRequest(
            prompt="test",
            aspect_ratio=aspect_ratio,
        )
        result = handler._convert_request(config, request)
        assert result["aspect_ratio"] == aspect_ratio


def test_bytedance_seedance_resolution_options(handler):
    """Test ByteDance Seedance resolution options."""
    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1.5/pro/text-to-video",
        provider="fal",
        api_key="test-key",
    )

    # Test both resolution options
    for resolution in ["480p", "720p"]:
        request = VideoGenerationRequest(
            prompt="test",
            resolution=resolution,
        )
        result = handler._convert_request(config, request)
        assert result["resolution"] == resolution


# ==================== ByteDance Seedance v1 Field Mapper Selection Tests ====================


def test_get_field_mappers_bytedance_seedance_v1():
    """Test ByteDance Seedance v1 mapper selection for all variants."""
    # Text-to-video
    assert (
        get_field_mappers("fal-ai/bytedance/seedance/v1/pro/fast/text-to-video")
        is BYTEDANCE_SEEDANCE_FIELD_MAPPERS
    )
    # Image-to-video
    assert (
        get_field_mappers("fal-ai/bytedance/seedance/v1/pro/image-to-video")
        is BYTEDANCE_SEEDANCE_FIELD_MAPPERS
    )
    # Reference-to-video
    assert (
        get_field_mappers("fal-ai/bytedance/seedance/v1/lite/reference-to-video")
        is BYTEDANCE_SEEDANCE_FIELD_MAPPERS
    )
    # Prefix match
    assert (
        get_field_mappers("fal-ai/bytedance/seedance/v1")
        is BYTEDANCE_SEEDANCE_FIELD_MAPPERS
    )


# ==================== ByteDance Seedance v1 Request Conversion Tests ====================


def test_bytedance_v1_text_to_video(handler):
    """Test ByteDance v1 pro/fast/text-to-video conversion."""
    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1/pro/fast/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A martial artist moves with precision in a quiet dojo",
        duration_seconds=5,
        aspect_ratio="16:9",
        resolution="1080p",
        seed=42,
        extra_params={"camera_fixed": True, "enable_safety_checker": True},
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A martial artist moves with precision in a quiet dojo"
    assert result["duration"] == "5"
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "1080p"
    assert result["seed"] == 42
    assert result["camera_fixed"] is True
    assert result["enable_safety_checker"] is True


def test_bytedance_v1_image_to_video(handler):
    """Test ByteDance v1 pro/image-to-video conversion."""
    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1/pro/image-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A skier glides over fresh snow, joyously smiling",
        image_list=[{"image": "https://example.com/skier.jpg", "type": "reference"}],
        duration_seconds=6,
        aspect_ratio="16:9",
        resolution="1080p",
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A skier glides over fresh snow, joyously smiling"
    assert result["image_url"] == "https://example.com/skier.jpg"
    assert result["duration"] == "6"
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "1080p"


def test_bytedance_v1_image_to_video_with_end_frame(handler):
    """Test ByteDance v1 image-to-video with first and last frame."""
    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1/pro/image-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Smooth transition from start to end pose",
        image_list=[
            {"image": "https://example.com/start.jpg", "type": "reference"},
            {"image": "https://example.com/end.jpg", "type": "last_frame"},
        ],
        duration_seconds=8,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Smooth transition from start to end pose"
    assert result["image_url"] == "https://example.com/start.jpg"
    assert result["end_image_url"] == "https://example.com/end.jpg"
    assert result["duration"] == "8"


def test_bytedance_v1_reference_to_video(handler):
    """Test ByteDance v1 lite/reference-to-video conversion."""
    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1/lite/reference-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="The girl catches the puppy and hugs it",
        image_list=[
            {"image": "https://example.com/ref1.jpg", "type": "reference"},
            {"image": "https://example.com/ref2.jpg", "type": "reference"},
        ],
        duration_seconds=5,
        resolution="720p",
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "The girl catches the puppy and hugs it"
    assert result["reference_image_urls"] == [
        "https://example.com/ref1.jpg",
        "https://example.com/ref2.jpg",
    ]
    assert result["duration"] == "5"
    assert result["resolution"] == "720p"


def test_bytedance_v1_duration_validation(handler):
    """Test ByteDance v1 duration validation (2-12 seconds)."""
    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1/pro/fast/text-to-video",
        provider="fal",
        api_key="test-key",
    )

    # Test all valid durations (2-12 seconds)
    for valid_duration in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        request = VideoGenerationRequest(
            prompt="test",
            duration_seconds=valid_duration,
        )
        result = handler._convert_request(config, request)
        assert result["duration"] == str(valid_duration)

    # Test invalid durations
    for invalid_duration in [1, 13, 15]:
        request_invalid = VideoGenerationRequest(
            prompt="test",
            duration_seconds=invalid_duration,
        )
        with pytest.raises(ValidationError) as exc_info:
            handler._convert_request(config, request_invalid)

        assert "Invalid duration" in str(exc_info.value)
        assert f"{invalid_duration} seconds" in str(exc_info.value)


# ==================== Pixverse Field Mapper Selection Tests ====================


def test_get_field_mappers_pixverse_v55_all_variants():
    """Test Pixverse v5.5 mapper selection for all variants."""
    # Text-to-video
    assert (
        get_field_mappers("fal-ai/pixverse/v5.5/text-to-video")
        is PIXVERSE_FIELD_MAPPERS
    )
    # Image-to-video
    assert (
        get_field_mappers("fal-ai/pixverse/v5.5/image-to-video")
        is PIXVERSE_FIELD_MAPPERS
    )
    # Transition
    assert (
        get_field_mappers("fal-ai/pixverse/v5.5/transition") is PIXVERSE_FIELD_MAPPERS
    )
    # Effects
    assert get_field_mappers("fal-ai/pixverse/v5.5/effects") is PIXVERSE_FIELD_MAPPERS
    # Swap (separate endpoint)
    assert get_field_mappers("fal-ai/pixverse/swap") is PIXVERSE_FIELD_MAPPERS
    # Prefix match
    assert get_field_mappers("fal-ai/pixverse/v5.5") is PIXVERSE_FIELD_MAPPERS


def test_get_field_mappers_pixverse_v5_all_variants():
    """Test Pixverse v5 mapper selection for all variants."""
    # Text-to-video
    assert (
        get_field_mappers("fal-ai/pixverse/v5/text-to-video") is PIXVERSE_FIELD_MAPPERS
    )
    # Image-to-video
    assert (
        get_field_mappers("fal-ai/pixverse/v5/image-to-video") is PIXVERSE_FIELD_MAPPERS
    )
    # Prefix match
    assert get_field_mappers("fal-ai/pixverse/v5") is PIXVERSE_FIELD_MAPPERS


# ==================== Pixverse Request Conversion Tests ====================


def test_pixverse_text_to_video_minimal(handler):
    """Test Pixverse text-to-video with minimal parameters."""
    config = VideoGenerationConfig(
        model="fal-ai/pixverse/v5.5/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A serene lake at sunset with mountains in the background",
    )

    result = handler._convert_request(config, request)

    assert (
        result["prompt"] == "A serene lake at sunset with mountains in the background"
    )
    # No other fields should be present
    assert "duration" not in result
    assert "aspect_ratio" not in result
    assert "resolution" not in result


def test_pixverse_text_to_video_full(handler):
    """Test Pixverse text-to-video with all parameters."""
    config = VideoGenerationConfig(
        model="fal-ai/pixverse/v5.5/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Epic cinematic scene of a warrior",
        duration_seconds=10,
        aspect_ratio="16:9",
        resolution="1080p",
        seed=42,
        negative_prompt="blurry, low quality",
        extra_params={
            "style": "anime",
            "thinking_type": "enabled",
            "generate_audio_switch": True,
            "generate_multi_clip_switch": True,
        },
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Epic cinematic scene of a warrior"
    assert result["duration"] == "10"
    assert result["aspect_ratio"] == "16:9"
    assert result["resolution"] == "1080p"
    assert result["seed"] == 42
    assert result["negative_prompt"] == "blurry, low quality"
    assert result["style"] == "anime"
    assert result["thinking_type"] == "enabled"
    assert result["generate_audio_switch"] is True
    assert result["generate_multi_clip_switch"] is True


def test_pixverse_image_to_video(handler):
    """Test Pixverse image-to-video conversion."""
    config = VideoGenerationConfig(
        model="fal-ai/pixverse/v5.5/image-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A woman warrior walking with her wolf",
        image_list=[{"image": "https://example.com/warrior.jpg", "type": "reference"}],
        duration_seconds=8,
        resolution="720p",
        extra_params={
            "style": "3d_animation",
            "generate_audio_switch": True,
            "generate_multi_clip_switch": False,
        },
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A woman warrior walking with her wolf"
    assert result["image_url"] == "https://example.com/warrior.jpg"
    assert result["duration"] == "8"
    assert result["resolution"] == "720p"
    assert result["style"] == "3d_animation"
    assert result["generate_audio_switch"] is True
    assert result["generate_multi_clip_switch"] is False


def test_pixverse_transition(handler):
    """Test Pixverse transition with first and end frames."""
    config = VideoGenerationConfig(
        model="fal-ai/pixverse/v5.5/transition",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Scene slowly transitions from day to night",
        image_list=[
            {"image": "https://example.com/day.jpg", "type": "first_frame"},
            {"image": "https://example.com/night.jpg", "type": "last_frame"},
        ],
        duration_seconds=5,
        aspect_ratio="16:9",
        extra_params={
            "style": "cyberpunk",
            "generate_audio_switch": False,
        },
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Scene slowly transitions from day to night"
    assert result["first_image_url"] == "https://example.com/day.jpg"
    assert result["end_image_url"] == "https://example.com/night.jpg"
    assert result["duration"] == "5"
    assert result["aspect_ratio"] == "16:9"
    assert result["style"] == "cyberpunk"
    assert result["generate_audio_switch"] is False


def test_pixverse_transition_first_frame_only(handler):
    """Test Pixverse transition with only first frame (end frame optional)."""
    config = VideoGenerationConfig(
        model="fal-ai/pixverse/v5.5/transition",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Animate from this starting frame",
        image_list=[
            {"image": "https://example.com/start.jpg", "type": "first_frame"},
        ],
        duration_seconds=8,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Animate from this starting frame"
    assert result["first_image_url"] == "https://example.com/start.jpg"
    assert "end_image_url" not in result
    assert result["duration"] == "8"


def test_pixverse_effects(handler):
    """Test Pixverse effects variant."""
    config = VideoGenerationConfig(
        model="fal-ai/pixverse/v5.5/effects",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="",  # Prompt not required for effects (empty string is valid)
        image_list=[{"image": "https://example.com/person.jpg", "type": "reference"}],
        duration_seconds=5,
        resolution="720p",
        extra_params={
            "effect": "Zombie Mode",
            "thinking_type": "disabled",
        },
    )

    result = handler._convert_request(config, request)

    assert result["image_url"] == "https://example.com/person.jpg"
    assert result["duration"] == "5"
    assert result["resolution"] == "720p"
    assert result["effect"] == "Zombie Mode"
    assert result["thinking_type"] == "disabled"
    # Empty string prompt is included (valid value)
    assert result["prompt"] == ""


def test_pixverse_swap(handler):
    """Test Pixverse swap variant (person/object/background swap)."""
    config = VideoGenerationConfig(
        model="fal-ai/pixverse/swap",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="",  # Prompt not required for swap (empty string is valid)
        video="https://example.com/original.mp4",
        image_list=[{"image": "https://example.com/target.jpg", "type": "reference"}],
        resolution="720p",
        extra_params={
            "mode": "person",
            "keyframe_id": 1,
            "original_sound_switch": True,
        },
    )

    result = handler._convert_request(config, request)

    assert result["video_url"] == "https://example.com/original.mp4"
    assert result["image_url"] == "https://example.com/target.jpg"
    assert result["resolution"] == "720p"
    assert result["mode"] == "person"
    assert result["keyframe_id"] == 1
    assert result["original_sound_switch"] is True
    # Empty string prompt is included (valid value)
    assert result["prompt"] == ""


def test_pixverse_duration_validation(handler):
    """Test Pixverse duration validation (5, 8, 10 seconds)."""
    config = VideoGenerationConfig(
        model="fal-ai/pixverse/v5.5/text-to-video",
        provider="fal",
        api_key="test-key",
    )

    # Test all valid durations
    for valid_duration in [5, 8, 10]:
        request = VideoGenerationRequest(
            prompt="test",
            duration_seconds=valid_duration,
        )
        result = handler._convert_request(config, request)
        assert result["duration"] == str(valid_duration)

    # Test invalid durations
    for invalid_duration in [3, 4, 6, 7, 9, 12, 15]:
        request_invalid = VideoGenerationRequest(
            prompt="test",
            duration_seconds=invalid_duration,
        )
        with pytest.raises(ValidationError) as exc_info:
            handler._convert_request(config, request_invalid)

        assert "Invalid duration" in str(exc_info.value)
        assert f"{invalid_duration} seconds" in str(exc_info.value)
        assert "5, 8, 10" in str(exc_info.value)


def test_pixverse_v5_same_mapper_as_v55(handler):
    """Test that Pixverse v5 uses same mapper as v5.5."""
    config_v5 = VideoGenerationConfig(
        model="fal-ai/pixverse/v5/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    config_v55 = VideoGenerationConfig(
        model="fal-ai/pixverse/v5.5/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="test",
        duration_seconds=5,
        aspect_ratio="16:9",
    )

    result_v5 = handler._convert_request(config_v5, request)
    result_v55 = handler._convert_request(config_v55, request)

    # Both versions should produce identical results
    assert result_v5 == result_v55
    assert result_v5["prompt"] == "test"
    assert result_v5["duration"] == "5"
    assert result_v5["aspect_ratio"] == "16:9"


# ==================== Error Decorator Tests ====================


@pytest.mark.asyncio
async def test_handle_video_generation_errors_async_propagates_known_errors():
    """Test decorator propagates ValidationError, GenerationFailedError, TarashException."""

    @handle_video_generation_errors
    async def async_func(self, config, request, on_progress=None):
        if request.prompt == "validation":
            raise ValidationError("Invalid", provider="fal")
        elif request.prompt == "provider":
            raise GenerationFailedError("API error", provider="fal")
        elif request.prompt == "video":
            raise TarashException("Gen error", provider="fal")
        else:
            raise RuntimeError("Unknown")

    config = VideoGenerationConfig(model="test-model", provider="fal", api_key="key")

    # Test ValidationError propagates
    with pytest.raises(ValidationError):
        await async_func(None, config, VideoGenerationRequest(prompt="validation"))

    # Test GenerationFailedError propagates
    with pytest.raises(GenerationFailedError):
        await async_func(None, config, VideoGenerationRequest(prompt="provider"))

    # Test TarashException propagates
    with pytest.raises(TarashException):
        await async_func(None, config, VideoGenerationRequest(prompt="video"))

    # Test unknown exception is wrapped
    with pytest.raises(TarashException, match="Unknown error") as exc_info:
        await async_func(None, config, VideoGenerationRequest(prompt="unknown"))

    assert exc_info.value.provider == "fal"
    assert exc_info.value.model == "test-model"
    assert "error" in exc_info.value.raw_response


def test_handle_video_generation_errors_sync_propagates_known_errors():
    """Test decorator propagates known errors for sync functions."""

    @handle_video_generation_errors
    def sync_func(self, config, request, on_progress=None):
        if request.prompt == "validation":
            raise ValidationError("Invalid", provider="fal")
        elif request.prompt == "unknown":
            raise RuntimeError("Unknown")

    config = VideoGenerationConfig(model="test-model", provider="fal", api_key="key")

    # Test ValidationError propagates
    with pytest.raises(ValidationError):
        sync_func(None, config, VideoGenerationRequest(prompt="validation"))

    # Test unknown exception is wrapped
    with pytest.raises(TarashException, match="Unknown error"):
        sync_func(None, config, VideoGenerationRequest(prompt="unknown"))


# ==================== Kling O1 Field Mapper Selection Tests ====================


def test_get_field_mappers_kling_o1_all_variants():
    """Test Kling O1 mapper selection for all three variants."""
    from tarash.tarash_gateway.providers.fal import KLING_O1_FIELD_MAPPERS

    # Image-to-video
    assert (
        get_field_mappers("fal-ai/kling-video/o1/image-to-video")
        is KLING_O1_FIELD_MAPPERS
    )
    # Reference-to-video
    assert (
        get_field_mappers("fal-ai/kling-video/o1/reference-to-video")
        is KLING_O1_FIELD_MAPPERS
    )
    # Video-to-video/edit
    assert (
        get_field_mappers("fal-ai/kling-video/o1/standard/video-to-video/edit")
        is KLING_O1_FIELD_MAPPERS
    )
    # Prefix match
    assert get_field_mappers("fal-ai/kling-video/o1") is KLING_O1_FIELD_MAPPERS


# ==================== Kling O1 Request Conversion Tests ====================


def test_kling_o1_image_to_video_conversion(handler):
    """Test Kling O1 image-to-video request conversion."""
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/image-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Animate this winter scene with @Image1 as start frame",
        image_list=[
            {"image": "https://example.com/winter.jpg", "type": "first_frame"},
        ],
        duration_seconds=5,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Animate this winter scene with @Image1 as start frame"
    assert result["start_image_url"] == "https://example.com/winter.jpg"
    assert "end_image_url" not in result
    assert result["duration"] == "5"  # No "s" suffix
    assert "aspect_ratio" not in result


def test_kling_o1_image_to_video_with_end_frame(handler):
    """Test Kling O1 image-to-video with both start and end frames."""
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/image-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Transition from @Image1 to @Image2",
        image_list=[
            {"image": "https://example.com/start.jpg", "type": "first_frame"},
            {"image": "https://example.com/end.jpg", "type": "last_frame"},
        ],
        duration_seconds=10,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Transition from @Image1 to @Image2"
    assert result["start_image_url"] == "https://example.com/start.jpg"
    assert result["end_image_url"] == "https://example.com/end.jpg"
    assert result["duration"] == "10"


def test_kling_o1_reference_to_video_conversion(handler):
    """Test Kling O1 reference-to-video with elements and reference images."""
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/reference-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Show @Element1 walking through a forest with style of @Image1",
        image_list=[
            # Reference image for style
            {"image": "https://example.com/forest-style.jpg", "type": "reference"},
        ],
        duration_seconds=5,
        aspect_ratio="16:9",
        extra_params={
            "elements": [
                {
                    "frontal_image_url": "https://example.com/char-front.jpg",
                    "reference_image_urls": ["https://example.com/char-side.jpg"],
                }
            ]
        },
    )

    result = handler._convert_request(config, request)

    assert (
        result["prompt"]
        == "Show @Element1 walking through a forest with style of @Image1"
    )
    assert result["duration"] == "5"
    assert result["aspect_ratio"] == "16:9"

    # Check elements structure
    assert "elements" in result
    assert len(result["elements"]) == 1
    assert (
        result["elements"][0]["frontal_image_url"]
        == "https://example.com/char-front.jpg"
    )
    assert result["elements"][0]["reference_image_urls"] == [
        "https://example.com/char-side.jpg"
    ]

    # Check reference images
    assert "image_urls" in result
    assert result["image_urls"] == ["https://example.com/forest-style.jpg"]


def test_kling_o1_reference_to_video_multiple_elements(handler):
    """Test Kling O1 reference-to-video with multiple elements."""
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/reference-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="@Element1 and @Element2 standing together in style of @Image1",
        image_list=[
            # Style reference
            {"image": "https://example.com/style.jpg", "type": "reference"},
        ],
        duration_seconds=10,
        aspect_ratio="1:1",
        extra_params={
            "elements": [
                {"frontal_image_url": "https://example.com/char1.jpg"},
                {
                    "frontal_image_url": "https://example.com/char2-front.jpg",
                    "reference_image_urls": ["https://example.com/char2-back.jpg"],
                },
            ]
        },
    )

    result = handler._convert_request(config, request)

    # Check elements
    assert len(result["elements"]) == 2

    # Element 1 (single image)
    elem1 = next(
        e
        for e in result["elements"]
        if e["frontal_image_url"] == "https://example.com/char1.jpg"
    )
    assert "reference_image_urls" not in elem1

    # Element 2 (multiple images)
    elem2 = next(
        e
        for e in result["elements"]
        if e["frontal_image_url"] == "https://example.com/char2-front.jpg"
    )
    assert elem2["reference_image_urls"] == ["https://example.com/char2-back.jpg"]


def test_kling_o1_video_to_video_edit_conversion(handler):
    """Test Kling O1 video-to-video/edit request conversion."""
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/standard/video-to-video/edit",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Replace character with @Element1 and apply style from @Image1",
        video="https://example.com/input-video.mp4",
        image_list=[
            # Style reference
            {"image": "https://example.com/new-style.jpg", "type": "reference"},
        ],
        extra_params={
            "elements": [{"frontal_image_url": "https://example.com/new-char.jpg"}],
            "keep_audio": True,
        },
    )

    result = handler._convert_request(config, request)

    assert (
        result["prompt"]
        == "Replace character with @Element1 and apply style from @Image1"
    )
    assert result["video_url"] == "https://example.com/input-video.mp4"
    assert result["keep_audio"] is True
    assert "duration" not in result  # video-to-video/edit doesn't use duration

    # Check elements
    assert "elements" in result
    assert len(result["elements"]) == 1
    assert (
        result["elements"][0]["frontal_image_url"] == "https://example.com/new-char.jpg"
    )

    # Check reference images
    assert result["image_urls"] == ["https://example.com/new-style.jpg"]


def test_kling_o1_duration_validation(handler):
    """Test Kling O1 duration validation (only 5 or 10 seconds allowed)."""
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/image-to-video",
        provider="fal",
        api_key="test-key",
    )

    # Valid durations
    for valid_duration in [5, 10]:
        request = VideoGenerationRequest(
            prompt="Test",
            image_list=[
                {"image": "https://example.com/img.jpg", "type": "first_frame"}
            ],
            duration_seconds=valid_duration,
        )
        result = handler._convert_request(config, request)
        assert result["duration"] == str(valid_duration)

    # Invalid duration
    request_invalid = VideoGenerationRequest(
        prompt="Test",
        image_list=[{"image": "https://example.com/img.jpg", "type": "first_frame"}],
        duration_seconds=7,  # Not allowed
    )

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(config, request_invalid)

    assert "Invalid duration" in str(exc_info.value)
    assert "7 seconds" in str(exc_info.value)
    assert "kling-o1" in str(exc_info.value).lower()


def test_kling_o1_no_elements_in_extra_params(handler):
    """Test Kling O1 without elements in extra_params."""
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/reference-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Test with only reference images",
        image_list=[
            {"image": "https://example.com/ref1.jpg", "type": "reference"},
            {"image": "https://example.com/ref2.jpg", "type": "reference"},
        ],
    )

    result = handler._convert_request(config, request)

    # No elements field when not in extra_params
    assert "elements" not in result
    # But reference images should be present
    assert result["image_urls"] == [
        "https://example.com/ref1.jpg",
        "https://example.com/ref2.jpg",
    ]


# ==================== Image Generation Field Mapper Tests ====================


def test_get_image_field_mappers_flux2_pro():
    """Test that FLUX.2 Pro models use correct field mappers."""
    from tarash.tarash_gateway.providers.fal import (
        FLUX2_PRO_IMAGE_FIELD_MAPPERS,
    )

    # Test all three FLUX.2 variants
    assert get_image_field_mappers("fal-ai/flux-2/pro") is FLUX2_PRO_IMAGE_FIELD_MAPPERS
    assert get_image_field_mappers("fal-ai/flux-2/dev") is FLUX2_PRO_IMAGE_FIELD_MAPPERS
    assert (
        get_image_field_mappers("fal-ai/flux-2/flex") is FLUX2_PRO_IMAGE_FIELD_MAPPERS
    )


def test_flux2_pro_field_mappers_basic():
    """Test FLUX.2 Pro field mappers with basic parameters."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-2/pro")

    request = ImageGenerationRequest(
        prompt="A futuristic cityscape at night",
        seed=42,
        n=2,
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A futuristic cityscape at night"
    assert result["seed"] == 42
    assert result["n"] == 2


def test_flux2_pro_multi_reference_images():
    """Test FLUX.2 Pro with multiple reference images."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-2/pro")

    request = ImageGenerationRequest(
        prompt="A character in this style",
        image_list=[
            {"image": "https://example.com/ref1.jpg", "type": "reference"},
            {"image": "https://example.com/ref2.jpg", "type": "reference"},
            {"image": "https://example.com/ref3.jpg", "type": "reference"},
        ],
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A character in this style"
    assert result["reference_images"] == [
        "https://example.com/ref1.jpg",
        "https://example.com/ref2.jpg",
        "https://example.com/ref3.jpg",
    ]


def test_flux2_pro_guidance_scale():
    """Test FLUX.2 Pro with guidance_scale parameter (1.0-20.0)."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-2/pro")

    request = ImageGenerationRequest(
        prompt="Test image",
        extra_params={"guidance_scale": 7.5},
    )

    result = apply_field_mappers(mappers, request)

    assert result["guidance_scale"] == 7.5


def test_flux2_pro_num_inference_steps():
    """Test FLUX.2 Pro with num_inference_steps parameter (1-50)."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-2/pro")

    request = ImageGenerationRequest(
        prompt="Test image",
        extra_params={"num_inference_steps": 30},
    )

    result = apply_field_mappers(mappers, request)

    assert result["num_inference_steps"] == 30


def test_flux2_pro_all_parameters():
    """Test FLUX.2 Pro with all parameters combined."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-2/pro")

    request = ImageGenerationRequest(
        prompt="A stylized portrait",
        seed=12345,
        n=1,
        size="landscape_4_3",
        image_list=[
            {"image": "https://example.com/style1.jpg", "type": "reference"},
            {"image": "https://example.com/style2.jpg", "type": "reference"},
        ],
        extra_params={
            "guidance_scale": 5.0,
            "num_inference_steps": 25,
        },
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A stylized portrait"
    assert result["seed"] == 12345
    assert result["n"] == 1
    assert result["size"] == "landscape_4_3"
    assert result["reference_images"] == [
        "https://example.com/style1.jpg",
        "https://example.com/style2.jpg",
    ]
    assert result["guidance_scale"] == 5.0
    assert result["num_inference_steps"] == 25


def test_get_image_field_mappers_flux_pro_ultra():
    """Test that Flux 1.1 Pro Ultra models use correct field mappers."""
    from tarash.tarash_gateway.providers.fal import (
        FLUX_PRO_ULTRA_FIELD_MAPPERS,
    )

    # Test both Ultra and Raw variants
    assert (
        get_image_field_mappers("fal-ai/flux-pro/v1.1-ultra")
        is FLUX_PRO_ULTRA_FIELD_MAPPERS
    )
    assert (
        get_image_field_mappers("fal-ai/flux-pro/v1.1-raw")
        is FLUX_PRO_ULTRA_FIELD_MAPPERS
    )


def test_flux_pro_ultra_aspect_ratio():
    """Test Flux Pro Ultra with aspect_ratio field."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-pro/v1.1-ultra")

    # Test with standard aspect ratio from ImageGenerationRequest
    request = ImageGenerationRequest(
        prompt="A beautiful landscape",
        aspect_ratio="16:9",
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A beautiful landscape"
    assert result["aspect_ratio"] == "16:9"


def test_flux_pro_ultra_extended_aspect_ratios():
    """Test Flux Pro Ultra supports extended aspect ratios (21:9, 9:21)."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-pro/v1.1-ultra")

    # Test with ultra-wide aspect ratio via extra_params
    request = ImageGenerationRequest(
        prompt="Cinematic wide shot",
        extra_params={"aspect_ratio": "21:9"},
    )

    result = apply_field_mappers(mappers, request)

    assert result["aspect_ratio"] == "21:9"


def test_flux_pro_ultra_raw_mode():
    """Test Flux Pro Ultra with raw mode (boolean for natural aesthetic)."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-pro/v1.1-ultra")

    request = ImageGenerationRequest(
        prompt="Natural portrait",
        extra_params={"raw": True},
    )

    result = apply_field_mappers(mappers, request)

    assert result["raw"] is True


def test_flux_pro_ultra_safety_tolerance():
    """Test Flux Pro Ultra with safety_tolerance field (1-6)."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-pro/v1.1-ultra")

    request = ImageGenerationRequest(
        prompt="Artistic image",
        extra_params={"safety_tolerance": 3},
    )

    result = apply_field_mappers(mappers, request)

    assert result["safety_tolerance"] == 3


def test_flux_pro_ultra_output_format():
    """Test Flux Pro Ultra with output_format field (jpeg, png)."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-pro/v1.1-ultra")

    # Test with PNG format
    request = ImageGenerationRequest(
        prompt="High quality image",
        extra_params={"output_format": "png"},
    )

    result = apply_field_mappers(mappers, request)

    assert result["output_format"] == "png"


def test_flux_pro_ultra_all_parameters():
    """Test Flux Pro Ultra with all parameters combined."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/flux-pro/v1.1-ultra")

    request = ImageGenerationRequest(
        prompt="Ultra high quality cinematic shot",
        seed=99999,
        aspect_ratio="21:9",
        extra_params={
            "raw": False,
            "safety_tolerance": 4,
            "output_format": "jpeg",
        },
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "Ultra high quality cinematic shot"
    assert result["seed"] == 99999
    assert result["aspect_ratio"] == "21:9"
    assert result["raw"] is False
    assert result["safety_tolerance"] == 4
    assert result["output_format"] == "jpeg"


def test_get_image_field_mappers_zimage_turbo():
    """Test that Z-Image-Turbo model uses correct field mappers."""
    from tarash.tarash_gateway.providers.fal import (
        ZIMAGE_TURBO_FIELD_MAPPERS,
    )

    assert get_image_field_mappers("fal-ai/z-image-turbo") is ZIMAGE_TURBO_FIELD_MAPPERS


def test_zimage_turbo_basic_parameters():
    """Test Z-Image-Turbo with basic parameters."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/z-image-turbo")

    request = ImageGenerationRequest(
        prompt="A fast distilled image generation",
        seed=777,
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A fast distilled image generation"
    assert result["seed"] == 777


def test_zimage_turbo_num_inference_steps():
    """Test Z-Image-Turbo with num_inference_steps (default 8 for distilled model)."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/z-image-turbo")

    # Test with custom inference steps
    request = ImageGenerationRequest(
        prompt="Quick generation",
        extra_params={"num_inference_steps": 8},
    )

    result = apply_field_mappers(mappers, request)

    assert result["num_inference_steps"] == 8


def test_zimage_turbo_negative_prompt():
    """Test Z-Image-Turbo with negative_prompt support."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/z-image-turbo")

    request = ImageGenerationRequest(
        prompt="Beautiful landscape",
        negative_prompt="blurry, low quality, distorted",
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "Beautiful landscape"
    assert result["negative_prompt"] == "blurry, low quality, distorted"


def test_zimage_turbo_enable_safety_checker():
    """Test Z-Image-Turbo with enable_safety_checker boolean."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/z-image-turbo")

    # Test with safety checker disabled
    request = ImageGenerationRequest(
        prompt="Artistic content",
        extra_params={"enable_safety_checker": False},
    )

    result = apply_field_mappers(mappers, request)

    assert result["enable_safety_checker"] is False


def test_zimage_turbo_all_parameters():
    """Test Z-Image-Turbo with all parameters combined."""
    from tarash.tarash_gateway.providers.fal import get_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_image_field_mappers("fal-ai/z-image-turbo")

    request = ImageGenerationRequest(
        prompt="Fast distilled generation with high quality",
        seed=12345,
        negative_prompt="blur, artifacts, noise",
        size="landscape_4_3",
        extra_params={
            "num_inference_steps": 8,
            "enable_safety_checker": True,
        },
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "Fast distilled generation with high quality"
    assert result["seed"] == 12345
    assert result["negative_prompt"] == "blur, artifacts, noise"
    assert result["size"] == "landscape_4_3"
    assert result["num_inference_steps"] == 8
    assert result["enable_safety_checker"] is True
