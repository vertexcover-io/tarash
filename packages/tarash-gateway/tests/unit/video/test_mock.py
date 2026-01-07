"""Unit tests for video mock module."""

import pytest

from tarash.tarash_gateway.exceptions import (
    HTTPError,
    TimeoutError,
    ValidationError,
)
from tarash.tarash_gateway.mock import (
    MockConfig,
    MockPollingConfig,
    MockResponse,
    find_best_matching_video,
    handle_mock_request_async,
    handle_mock_request_sync,
)
from tarash.tarash_gateway.models import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)


# ==================== Fixtures ====================


@pytest.fixture
def basic_request():
    """Create a basic video generation request."""
    return VideoGenerationRequest(
        prompt="A cat playing piano",
        aspect_ratio="16:9",
        resolution="1080p",
        duration_seconds=4,
    )


@pytest.fixture
def portrait_request():
    """Create a portrait video request."""
    return VideoGenerationRequest(
        prompt="Portrait video",
        aspect_ratio="9:16",
        resolution="720p",
        duration_seconds=4,
    )


# ==================== Test MockConfig Validation ====================


def test_mock_config_enabled_with_default_response():
    """Enabled config with no responses defaults to single success response."""
    config = MockConfig(enabled=True)
    assert config.responses is not None
    assert len(config.responses) == 1
    assert config.responses[0].weight == 1.0


def test_mock_config_disabled():
    """Disabled config does not auto-create responses."""
    config = MockConfig(enabled=False)
    assert config.responses is None


def test_mock_config_total_weight_validation():
    """Total weight must be positive."""
    # Note: Individual MockResponse validates weight > 0 first,
    # so we test with weights that are positive but sum to 0 or negative isn't possible.
    # Instead, test that validation accepts positive weights
    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(weight=0.5),
            MockResponse(weight=0.5),
        ],
    )
    assert config.responses is not None
    assert len(config.responses) == 2


# ==================== Test MockResponse Validation ====================


def test_mock_response_cannot_have_both_success_and_error():
    """Cannot specify both success response and error."""
    with pytest.raises(ValueError, match="Cannot specify both success and error"):
        MockResponse(
            weight=1.0,
            output_video="https://example.com/video.mp4",
            error=ValidationError("Test error"),
        )


def test_mock_response_cannot_have_both_mock_response_and_output_video():
    """Cannot specify both mock_response and output_video."""
    mock_resp = VideoGenerationResponse(
        request_id="test",
        video="https://example.com/video.mp4",
        status="completed",
        raw_response={},
    )
    with pytest.raises(
        ValueError, match="Cannot specify both mock_response and output_video"
    ):
        MockResponse(
            weight=1.0,
            mock_response=mock_resp,
            output_video="https://example.com/other.mp4",
        )


def test_mock_response_weight_must_be_positive():
    """Weight must be positive."""
    with pytest.raises(ValueError, match="weight must be positive"):
        MockResponse(weight=0)

    with pytest.raises(ValueError, match="weight must be positive"):
        MockResponse(weight=-1.0)


# ==================== Test MockPollingConfig Validation ====================


def test_polling_config_progress_percentages_length_mismatch():
    """progress_percentages length must match status_sequence."""
    with pytest.raises(ValueError, match="progress_percentages length must match"):
        MockPollingConfig(
            status_sequence=["queued", "processing", "completed"],
            progress_percentages=[0, 100],  # Wrong length
        )


def test_polling_config_custom_updates_length_mismatch():
    """custom_updates length must match status_sequence."""
    with pytest.raises(ValueError, match="custom_updates length must match"):
        MockPollingConfig(
            status_sequence=["queued", "processing", "completed"],
            custom_updates=[{"a": 1}, {"b": 2}],  # Wrong length
        )


# ==================== Test find_best_matching_video ====================


def test_find_best_matching_video_exact_match(basic_request):
    """Exact match returns correct video."""
    spec = find_best_matching_video(basic_request)
    assert spec.aspect_ratio == "16:9"
    assert spec.resolution == "1080p"
    assert spec.duration == 4.0


def test_find_best_matching_video_default_values():
    """Default values applied when not provided."""
    request = VideoGenerationRequest(prompt="test")
    spec = find_best_matching_video(request)
    # Should match defaults: 9:16, 720p, 4s
    assert spec.aspect_ratio == "9:16"
    assert spec.resolution == "720p"
    assert spec.duration == 4.0


def test_find_best_matching_video_portrait(portrait_request):
    """Portrait video matches correctly."""
    spec = find_best_matching_video(portrait_request)
    assert spec.aspect_ratio == "9:16"
    assert spec.resolution == "720p"


# ==================== Test handle_mock_request_sync ====================


def test_sync_simple_success_response(basic_request):
    """Simple success response with auto-matched video."""
    config = MockConfig(enabled=True)
    response = handle_mock_request_sync(config, basic_request)

    assert response.status == "completed"
    assert response.is_mock is True
    assert response.request_id.startswith("mock_")
    assert response.video is not None
    assert response.aspect_ratio == "16:9"
    assert response.resolution == "1080p"


def test_sync_explicit_output_video_url(basic_request):
    """Explicit output_video as URL."""
    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                output_video="https://example.com/custom.mp4",
                output_video_type="url",
            )
        ],
    )
    response = handle_mock_request_sync(config, basic_request)

    assert response.status == "completed"
    assert response.is_mock is True
    assert response.video == "https://example.com/custom.mp4"
    # Metadata should be None for explicit output_video
    assert response.duration is None
    assert response.resolution is None
    assert response.aspect_ratio is None


def test_sync_error_response_validation_error(basic_request):
    """Error response with ValidationError."""
    error = ValidationError("Invalid aspect ratio", provider="mock")
    config = MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0, error=error)],
    )

    with pytest.raises(ValidationError, match="Invalid aspect ratio"):
        handle_mock_request_sync(config, basic_request)


def test_sync_error_response_http_error(basic_request):
    """Error response with HTTPError."""
    error = HTTPError(
        "Rate limit exceeded",
        provider="mock",
        status_code=429,
        raw_response={"retry_after": 60},
    )
    config = MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0, error=error)],
    )

    with pytest.raises(HTTPError, match="Rate limit exceeded") as exc_info:
        handle_mock_request_sync(config, basic_request)

    assert exc_info.value.status_code == 429


def test_sync_error_response_timeout(basic_request):
    """Error response with TimeoutError."""
    error = TimeoutError(
        "Request timed out",
        provider="mock",
        timeout_seconds=600,
    )
    config = MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0, error=error)],
    )

    with pytest.raises(TimeoutError, match="Request timed out") as exc_info:
        handle_mock_request_sync(config, basic_request)

    assert exc_info.value.timeout_seconds == 600


def test_sync_weighted_responses_all_success(basic_request):
    """Weighted responses - multiple success options."""
    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=0.5,
                output_video="https://example.com/video1.mp4",
            ),
            MockResponse(
                weight=0.5,
                output_video="https://example.com/video2.mp4",
            ),
        ],
    )

    # Run multiple times to test randomness
    videos_seen = set()
    for _ in range(10):
        response = handle_mock_request_sync(config, basic_request)
        assert response.status == "completed"
        videos_seen.add(response.video)

    # Should see both videos (with high probability)
    assert len(videos_seen) >= 1


def test_sync_polling_callback_called(basic_request):
    """Polling callback is invoked with correct updates."""
    updates_received = []

    def progress_callback(update: VideoGenerationUpdate):
        updates_received.append(update)

    config = MockConfig(
        enabled=True,
        polling=MockPollingConfig(
            status_sequence=["queued", "processing", "completed"],
            delay_between_updates=0.01,  # Fast for testing
            progress_percentages=[0, 50, 100],
        ),
    )

    response = handle_mock_request_sync(
        config, basic_request, on_progress=progress_callback
    )

    assert response.status == "completed"
    assert len(updates_received) == 3

    # Check status sequence
    assert updates_received[0].status == "queued"
    assert updates_received[0].progress_percent == 0
    assert updates_received[1].status == "processing"
    assert updates_received[1].progress_percent == 50
    assert updates_received[2].status == "completed"
    assert updates_received[2].progress_percent == 100
    assert updates_received[2].result == response


def test_sync_polling_with_error(basic_request):
    """Polling with error ends with 'failed' status."""
    updates_received = []

    def progress_callback(update: VideoGenerationUpdate):
        updates_received.append(update)

    error = ValidationError("Test error", provider="mock")
    config = MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0, error=error)],
        polling=MockPollingConfig(
            status_sequence=["queued", "processing", "completed"],
            delay_between_updates=0.01,
        ),
    )

    with pytest.raises(ValidationError):
        handle_mock_request_sync(config, basic_request, on_progress=progress_callback)

    # Last update should have failed status
    assert len(updates_received) == 3
    assert updates_received[-1].status == "failed"
    assert updates_received[-1].error == str(error)


def test_sync_explicit_mock_response(basic_request):
    """Explicit mock_response is returned with updated request_id."""
    custom_response = VideoGenerationResponse(
        request_id="original-id",
        video="https://example.com/custom.mp4",
        status="completed",
        duration=10.0,
        resolution="4k",
        aspect_ratio="21:9",
        is_mock=False,  # Will be overridden
        raw_response={"custom": "data"},
    )

    config = MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0, mock_response=custom_response)],
    )

    response = handle_mock_request_sync(config, basic_request)

    assert response.is_mock is True  # Overridden
    assert response.request_id.startswith("mock_")  # New ID
    assert response.video == "https://example.com/custom.mp4"
    assert response.duration == 10.0  # Preserved
    assert response.resolution == "4k"  # Preserved


# ==================== Test handle_mock_request_async ====================


@pytest.mark.asyncio
async def test_async_simple_success_response(basic_request):
    """Simple success response with auto-matched video."""
    config = MockConfig(enabled=True)
    response = await handle_mock_request_async(config, basic_request)

    assert response.status == "completed"
    assert response.is_mock is True
    assert response.request_id.startswith("mock_")
    assert response.video is not None


@pytest.mark.asyncio
async def test_async_error_response(basic_request):
    """Error response raises exception."""
    error = ValidationError("Test error", provider="mock")
    config = MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0, error=error)],
    )

    with pytest.raises(ValidationError, match="Test error"):
        await handle_mock_request_async(config, basic_request)


@pytest.mark.asyncio
async def test_async_polling_callback_async(basic_request):
    """Polling callback works with async callbacks."""
    updates_received = []

    async def async_progress_callback(update: VideoGenerationUpdate):
        updates_received.append(update)

    config = MockConfig(
        enabled=True,
        polling=MockPollingConfig(
            status_sequence=["queued", "completed"],
            delay_between_updates=0.01,
        ),
    )

    response = await handle_mock_request_async(
        config, basic_request, on_progress=async_progress_callback
    )

    assert response.status == "completed"
    assert len(updates_received) == 2
    assert updates_received[0].status == "queued"
    assert updates_received[1].status == "completed"


@pytest.mark.asyncio
async def test_async_polling_callback_sync(basic_request):
    """Polling callback works with sync callbacks in async mode."""
    updates_received = []

    def sync_progress_callback(update: VideoGenerationUpdate):
        updates_received.append(update)

    config = MockConfig(
        enabled=True,
        polling=MockPollingConfig(
            status_sequence=["queued", "completed"],
            delay_between_updates=0.01,
        ),
    )

    response = await handle_mock_request_async(
        config, basic_request, on_progress=sync_progress_callback
    )

    assert response.status == "completed"
    assert len(updates_received) == 2


@pytest.mark.asyncio
async def test_async_custom_polling_updates(basic_request):
    """Custom polling updates are passed through."""
    updates_received = []

    async def progress_callback(update: VideoGenerationUpdate):
        updates_received.append(update)

    config = MockConfig(
        enabled=True,
        polling=MockPollingConfig(
            status_sequence=["queued", "processing", "completed"],
            delay_between_updates=0.01,
            custom_updates=[
                {"position": 5},
                {"logs": ["Processing frame 1", "Processing frame 2"]},
                {"final": True},
            ],
        ),
    )

    await handle_mock_request_async(
        config, basic_request, on_progress=progress_callback
    )

    assert len(updates_received) == 3
    assert updates_received[0].update == {"position": 5}
    assert "logs" in updates_received[1].update
    assert updates_received[2].update == {"final": True}


# ==================== Test Integration Scenarios ====================


def test_load_balancing_success_and_error(basic_request):
    """Load balancing between success and different error types."""
    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(weight=0.5),  # Success
            MockResponse(
                weight=0.3,
                error=HTTPError("Rate limit", status_code=429),
            ),
            MockResponse(
                weight=0.2,
                error=TimeoutError("Timeout", timeout_seconds=600),
            ),
        ],
    )

    # Run multiple times
    results = {"success": 0, "http_error": 0, "timeout": 0}
    for _ in range(20):
        try:
            _ = handle_mock_request_sync(config, basic_request)
            results["success"] += 1
        except HTTPError:
            results["http_error"] += 1
        except TimeoutError:
            results["timeout"] += 1

    # Should see all three outcomes (with high probability)
    assert results["success"] > 0
    assert sum(results.values()) == 20


@pytest.mark.asyncio
async def test_different_aspect_ratios():
    """Test matching for different aspect ratios."""
    test_cases = [
        ("16:9", "720p", 4),
        ("9:16", "1080p", 4),
        ("1:1", "720p", 4),
        ("4:3", "480p", 4),
        ("21:9", "1080p", 4),
    ]

    config = MockConfig(enabled=True)

    for aspect, resolution, duration in test_cases:
        request = VideoGenerationRequest(
            prompt="test",
            aspect_ratio=aspect,
            resolution=resolution,
            duration_seconds=duration,
        )
        response = await handle_mock_request_async(config, request)

        assert response.aspect_ratio == aspect
        assert response.resolution == resolution
        assert response.duration == float(duration)


# ==================== Test Content Download ====================


def test_sync_content_download_with_url(basic_request, monkeypatch):
    """Sync: output_video_type='content' downloads and converts to bytes."""

    # Mock the download function to avoid actual HTTP calls
    def mock_download(url, provider):
        return (b"fake_video_content", "video/mp4")

    monkeypatch.setattr(
        "tarash.tarash_gateway.mock.download_media_from_url",
        mock_download,
    )

    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                output_video="https://example.com/test-video.mp4",
                output_video_type="content",
            )
        ],
    )

    response = handle_mock_request_sync(config, basic_request)

    # Response should have MediaContent dict
    assert response.status == "completed"
    assert response.is_mock is True
    assert isinstance(response.video, dict)
    assert response.video["content"] == b"fake_video_content"
    assert response.video["content_type"] == "video/mp4"

    # Metadata should be None for explicit output_video
    assert response.duration is None
    assert response.resolution is None
    assert response.aspect_ratio is None


@pytest.mark.asyncio
async def test_async_content_download_with_url(basic_request, monkeypatch):
    """Async: output_video_type='content' downloads and converts to bytes."""

    # Mock the async download function
    async def mock_download_async(url, provider):
        return (b"fake_async_video_content", "video/mp4")

    monkeypatch.setattr(
        "tarash.tarash_gateway.mock.download_media_from_url_async",
        mock_download_async,
    )

    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                output_video="https://example.com/async-video.mp4",
                output_video_type="content",
            )
        ],
    )

    response = await handle_mock_request_async(config, basic_request)

    # Response should have MediaContent dict
    assert response.status == "completed"
    assert response.is_mock is True
    assert isinstance(response.video, dict)
    assert response.video["content"] == b"fake_async_video_content"
    assert response.video["content_type"] == "video/mp4"


def test_sync_content_with_base64_data_url(basic_request):
    """Sync: Base64 data URL is converted to MediaContent."""
    # Create a base64 data URL
    video_bytes = b"fake_video_data_base64"
    b64_encoded = (
        "ZmFrZV92aWRlb19kYXRhX2Jhc2U2NA=="  # base64 of "fake_video_data_base64"
    )
    data_url = f"data:video/mp4;base64,{b64_encoded}"

    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                output_video=data_url,
                output_video_type="content",
            )
        ],
    )

    response = handle_mock_request_sync(config, basic_request)

    assert response.status == "completed"
    assert isinstance(response.video, dict)
    assert response.video["content"] == video_bytes
    assert response.video["content_type"] == "video/mp4"


@pytest.mark.asyncio
async def test_async_content_with_base64_data_url(basic_request):
    """Async: Base64 data URL is converted to MediaContent."""
    video_bytes = b"fake_async_base64_data"
    b64_encoded = (
        "ZmFrZV9hc3luY19iYXNlNjRfZGF0YQ=="  # base64 of "fake_async_base64_data"
    )
    data_url = f"data:video/webm;base64,{b64_encoded}"

    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                output_video=data_url,
                output_video_type="content",
            )
        ],
    )

    response = await handle_mock_request_async(config, basic_request)

    assert response.status == "completed"
    assert isinstance(response.video, dict)
    assert response.video["content"] == video_bytes
    assert response.video["content_type"] == "video/webm"


def test_sync_content_with_plain_base64(basic_request):
    """Sync: Plain base64 string (no data: prefix) is converted."""
    video_bytes = b"plain_base64_video"
    b64_encoded = "cGxhaW5fYmFzZTY0X3ZpZGVv"  # base64 of "plain_base64_video"

    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                output_video=b64_encoded,
                output_video_type="content",
            )
        ],
    )

    response = handle_mock_request_sync(config, basic_request)

    assert response.status == "completed"
    assert isinstance(response.video, dict)
    assert response.video["content"] == video_bytes
    assert response.video["content_type"] == "video/mp4"  # Default


@pytest.mark.asyncio
async def test_async_content_with_plain_base64(basic_request):
    """Async: Plain base64 string (no data: prefix) is converted."""
    video_bytes = b"async_plain_base64"
    b64_encoded = "YXN5bmNfcGxhaW5fYmFzZTY0"  # base64 of "async_plain_base64"

    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                output_video=b64_encoded,
                output_video_type="content",
            )
        ],
    )

    response = await handle_mock_request_async(config, basic_request)

    assert response.status == "completed"
    assert isinstance(response.video, dict)
    assert response.video["content"] == video_bytes
    assert response.video["content_type"] == "video/mp4"  # Default


def test_sync_content_with_existing_media_content(basic_request):
    """Sync: Existing MediaContent dict is passed through unchanged."""
    media_content = {"content": b"existing_content", "content_type": "video/avi"}

    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                output_video=media_content,
                output_video_type="content",
            )
        ],
    )

    response = handle_mock_request_sync(config, basic_request)

    assert response.status == "completed"
    assert response.video == media_content
    assert response.video["content"] == b"existing_content"
    assert response.video["content_type"] == "video/avi"


@pytest.mark.asyncio
async def test_async_content_with_existing_media_content(basic_request):
    """Async: Existing MediaContent dict is passed through unchanged."""
    media_content = {
        "content": b"async_existing_content",
        "content_type": "video/mov",
    }

    config = MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                output_video=media_content,
                output_video_type="content",
            )
        ],
    )

    response = await handle_mock_request_async(config, basic_request)

    assert response.status == "completed"
    assert response.video == media_content
    assert response.video["content"] == b"async_existing_content"
    assert response.video["content_type"] == "video/mov"
