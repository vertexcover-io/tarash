"""Tests for execution metadata models and fallback configuration."""

from datetime import datetime


from tarash.tarash_gateway.models import (
    AttemptMetadata,
    ExecutionMetadata,
    VideoGenerationConfig,
    VideoGenerationResponse,
)


def test_attempt_metadata_elapsed_seconds_computed():
    """Test that elapsed_seconds is computed from timestamps."""
    started_at = datetime(2026, 1, 3, 10, 0, 0)
    ended_at = datetime(2026, 1, 3, 10, 0, 5)  # 5 seconds later

    metadata = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3.1",
        attempt_number=1,
        started_at=started_at,
        ended_at=ended_at,
        status="success",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-123",
    )

    assert metadata.elapsed_seconds == 5.0


def test_attempt_metadata_elapsed_seconds_none_when_not_ended():
    """Test that elapsed_seconds is None when ended_at is None."""
    started_at = datetime(2026, 1, 3, 10, 0, 0)

    metadata = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3.1",
        attempt_number=1,
        started_at=started_at,
        ended_at=None,
        status="processing",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-123",
    )

    assert metadata.elapsed_seconds is None


def test_attempt_metadata_success_status():
    """Test AttemptMetadata for successful attempt."""
    metadata = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3.1",
        attempt_number=1,
        started_at=datetime(2026, 1, 3, 10, 0, 0),
        ended_at=datetime(2026, 1, 3, 10, 0, 5),
        status="success",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-123",
    )

    assert metadata.status == "success"
    assert metadata.error_type is None
    assert metadata.error_message is None
    assert metadata.is_retryable is None


def test_attempt_metadata_failed_status():
    """Test AttemptMetadata for failed attempt."""
    metadata = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3.1",
        attempt_number=1,
        started_at=datetime(2026, 1, 3, 10, 0, 0),
        ended_at=datetime(2026, 1, 3, 10, 0, 5),
        status="failed",
        error_type="HTTPError",
        error_message="500 Internal Server Error",
        is_retryable=True,
        request_id="req-123",
    )

    assert metadata.status == "failed"
    assert metadata.error_type == "HTTPError"
    assert metadata.error_message == "500 Internal Server Error"
    assert metadata.is_retryable is True


def test_execution_metadata_total_elapsed_seconds():
    """Test that total_elapsed_seconds is computed from all attempts."""
    attempt1 = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3.1",
        attempt_number=1,
        started_at=datetime(2026, 1, 3, 10, 0, 0),
        ended_at=datetime(2026, 1, 3, 10, 0, 5),  # 5 seconds
        status="failed",
        error_type="HTTPError",
        error_message="500 Internal Server Error",
        is_retryable=True,
        request_id="req-123",
    )

    attempt2 = AttemptMetadata(
        provider="replicate",
        model="minimax/video-01",
        attempt_number=2,
        started_at=datetime(2026, 1, 3, 10, 0, 6),
        ended_at=datetime(2026, 1, 3, 10, 0, 15),  # 9 seconds
        status="success",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-456",
    )

    metadata = ExecutionMetadata(
        total_attempts=2,
        successful_attempt=2,
        attempts=[attempt1, attempt2],
        fallback_triggered=True,
        configs_in_chain=2,
    )

    # Total: from first start (10:00:00) to last end (10:00:15) = 15 seconds
    assert metadata.total_elapsed_seconds == 15.0


def test_execution_metadata_single_attempt():
    """Test ExecutionMetadata with single successful attempt."""
    attempt = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3.1",
        attempt_number=1,
        started_at=datetime(2026, 1, 3, 10, 0, 0),
        ended_at=datetime(2026, 1, 3, 10, 0, 5),
        status="success",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-123",
    )

    metadata = ExecutionMetadata(
        total_attempts=1,
        successful_attempt=1,
        attempts=[attempt],
        fallback_triggered=False,
        configs_in_chain=1,
    )

    assert metadata.total_attempts == 1
    assert metadata.successful_attempt == 1
    assert metadata.fallback_triggered is False
    assert metadata.configs_in_chain == 1
    assert metadata.total_elapsed_seconds == 5.0


def test_execution_metadata_all_failed():
    """Test ExecutionMetadata when all attempts failed."""
    attempt1 = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3.1",
        attempt_number=1,
        started_at=datetime(2026, 1, 3, 10, 0, 0),
        ended_at=datetime(2026, 1, 3, 10, 0, 5),
        status="failed",
        error_type="HTTPError",
        error_message="500 Internal Server Error",
        is_retryable=True,
        request_id="req-123",
    )

    attempt2 = AttemptMetadata(
        provider="replicate",
        model="minimax/video-01",
        attempt_number=2,
        started_at=datetime(2026, 1, 3, 10, 0, 6),
        ended_at=datetime(2026, 1, 3, 10, 0, 10),
        status="failed",
        error_type="TimeoutError",
        error_message="Request timed out",
        is_retryable=True,
        request_id="req-456",
    )

    metadata = ExecutionMetadata(
        total_attempts=2,
        successful_attempt=None,  # No successful attempt
        attempts=[attempt1, attempt2],
        fallback_triggered=True,
        configs_in_chain=2,
    )

    assert metadata.total_attempts == 2
    assert metadata.successful_attempt is None
    assert metadata.fallback_triggered is True


def test_video_generation_config_with_fallbacks():
    """Test VideoGenerationConfig with fallback configurations."""
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

    assert config.fallback_configs is not None
    assert len(config.fallback_configs) == 2
    assert config.fallback_configs[0].model == "replicate/minimax"
    assert config.fallback_configs[1].model == "openai/sora-2"


def test_video_generation_config_without_fallbacks():
    """Test VideoGenerationConfig without fallback configurations."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="fal-key",
    )

    assert config.fallback_configs is None


def test_video_generation_response_with_execution_metadata():
    """Test VideoGenerationResponse with execution metadata."""
    attempt = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3.1",
        attempt_number=1,
        started_at=datetime(2026, 1, 3, 10, 0, 0),
        ended_at=datetime(2026, 1, 3, 10, 0, 5),
        status="success",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-123",
    )

    exec_metadata = ExecutionMetadata(
        total_attempts=1,
        successful_attempt=1,
        attempts=[attempt],
        fallback_triggered=False,
        configs_in_chain=1,
    )

    response = VideoGenerationResponse(
        request_id="req-123",
        video="https://example.com/video.mp4",
        status="completed",
        raw_response={"status": "completed"},
        execution_metadata=exec_metadata,
    )

    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1
    assert response.execution_metadata.successful_attempt == 1
    assert response.execution_metadata.fallback_triggered is False


def test_video_generation_response_without_execution_metadata():
    """Test VideoGenerationResponse without execution metadata."""
    response = VideoGenerationResponse(
        request_id="req-123",
        video="https://example.com/video.mp4",
        status="completed",
        raw_response={"status": "completed"},
    )

    assert response.execution_metadata is None
