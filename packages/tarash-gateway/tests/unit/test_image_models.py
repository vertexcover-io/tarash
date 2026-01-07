"""Tests for image generation models."""

from datetime import datetime

# Import mock module to trigger model rebuild for forward references
import tarash.tarash_gateway.mock  # noqa: F401

from tarash.tarash_gateway.models import (
    AttemptMetadata,
    ExecutionMetadata,
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageGenerationUpdate,
)


# ==================== ImageGenerationConfig Tests ====================


def test_image_generation_config_basic():
    """Test basic ImageGenerationConfig creation."""
    config = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="test-api-key",
    )

    assert config.model == "fal-ai/flux/dev"
    assert config.provider == "fal"
    assert config.api_key == "test-api-key"
    assert config.timeout == 120  # Default for images
    assert config.max_poll_attempts == 60
    assert config.poll_interval == 2


def test_image_generation_config_with_custom_timeouts():
    """Test ImageGenerationConfig with custom timeout settings."""
    config = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="test-api-key",
        timeout=60,
        max_poll_attempts=30,
        poll_interval=1,
    )

    assert config.timeout == 60
    assert config.max_poll_attempts == 30
    assert config.poll_interval == 1


def test_image_generation_config_with_fallbacks():
    """Test ImageGenerationConfig with fallback configurations."""
    fallback = ImageGenerationConfig(
        model="fal-ai/recraft-v3",
        provider="fal",
        api_key="test-api-key",
    )

    config = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="test-api-key",
        fallback_configs=[fallback],
    )

    assert config.fallback_configs is not None
    assert len(config.fallback_configs) == 1
    assert config.fallback_configs[0].model == "fal-ai/recraft-v3"


def test_image_generation_config_without_fallbacks():
    """Test ImageGenerationConfig without fallback configurations."""
    config = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="test-api-key",
    )

    assert config.fallback_configs is None


# ==================== ImageGenerationRequest Tests ====================


def test_image_generation_request_basic():
    """Test basic ImageGenerationRequest creation."""
    request = ImageGenerationRequest(prompt="A beautiful sunset")

    assert request.prompt == "A beautiful sunset"
    assert request.negative_prompt is None
    assert request.size is None
    assert request.quality is None
    assert request.style is None
    assert request.n is None
    assert request.seed is None
    assert request.image_list == []
    assert request.mask_image is None
    assert request.extra_params == {}


def test_image_generation_request_with_all_params():
    """Test ImageGenerationRequest with all parameters."""
    request = ImageGenerationRequest(
        prompt="A beautiful sunset",
        negative_prompt="blurry, low quality",
        size="1024x1024",
        quality="hd",
        style="vivid",
        n=4,
        seed=12345,
        aspect_ratio="16:9",
    )

    assert request.prompt == "A beautiful sunset"
    assert request.negative_prompt == "blurry, low quality"
    assert request.size == "1024x1024"
    assert request.quality == "hd"
    assert request.style == "vivid"
    assert request.n == 4
    assert request.seed == 12345
    assert request.aspect_ratio == "16:9"


def test_image_generation_request_captures_extra_params():
    """Test that unknown params are captured in extra_params."""
    request = ImageGenerationRequest(
        prompt="A beautiful sunset",
        custom_param="custom_value",
        another_param=123,
    )

    assert request.prompt == "A beautiful sunset"
    assert request.extra_params == {
        "custom_param": "custom_value",
        "another_param": 123,
    }


def test_image_generation_request_merges_extra_params():
    """Test that explicit extra_params are merged with unknown params."""
    request = ImageGenerationRequest(
        prompt="A beautiful sunset",
        extra_params={"existing": "value"},
        new_param="new_value",
    )

    assert request.extra_params == {
        "existing": "value",
        "new_param": "new_value",
    }


# ==================== ImageGenerationResponse Tests ====================


def test_image_generation_response_basic():
    """Test basic ImageGenerationResponse creation."""
    response = ImageGenerationResponse(
        request_id="req-123",
        images=["https://example.com/image1.png"],
        status="completed",
        raw_response={"status": "completed"},
    )

    assert response.request_id == "req-123"
    assert response.images == ["https://example.com/image1.png"]
    assert response.status == "completed"
    assert response.is_mock is False
    assert response.revised_prompt is None
    assert response.content_type == "image/png"


def test_image_generation_response_with_multiple_images():
    """Test ImageGenerationResponse with multiple images."""
    response = ImageGenerationResponse(
        request_id="req-123",
        images=[
            "https://example.com/image1.png",
            "https://example.com/image2.png",
            "https://example.com/image3.png",
        ],
        status="completed",
        raw_response={"status": "completed"},
    )

    assert len(response.images) == 3


def test_image_generation_response_with_revised_prompt():
    """Test ImageGenerationResponse with revised prompt."""
    response = ImageGenerationResponse(
        request_id="req-123",
        images=["https://example.com/image1.png"],
        status="completed",
        raw_response={"status": "completed"},
        revised_prompt="A stunning sunset over the ocean with vibrant colors",
    )

    assert (
        response.revised_prompt
        == "A stunning sunset over the ocean with vibrant colors"
    )


def test_image_generation_response_with_execution_metadata():
    """Test ImageGenerationResponse with execution metadata."""
    attempt = AttemptMetadata(
        provider="fal",
        model="fal-ai/flux/dev",
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

    response = ImageGenerationResponse(
        request_id="req-123",
        images=["https://example.com/image1.png"],
        status="completed",
        raw_response={"status": "completed"},
        execution_metadata=exec_metadata,
    )

    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1
    assert response.execution_metadata.successful_attempt == 1
    assert response.execution_metadata.fallback_triggered is False


def test_image_generation_response_mock():
    """Test ImageGenerationResponse with mock flag."""
    response = ImageGenerationResponse(
        request_id="req-123",
        images=["data:image/png;base64,abc123..."],
        status="completed",
        raw_response={"status": "completed"},
        is_mock=True,
    )

    assert response.is_mock is True


# ==================== ImageGenerationUpdate Tests ====================


def test_image_generation_update_queued():
    """Test ImageGenerationUpdate in queued state."""
    update = ImageGenerationUpdate(
        request_id="req-123",
        status="queued",
        progress_percent=None,
        update={"position": 5},
    )

    assert update.request_id == "req-123"
    assert update.status == "queued"
    assert update.progress_percent is None
    assert update.update == {"position": 5}
    assert update.result is None
    assert update.error is None


def test_image_generation_update_processing():
    """Test ImageGenerationUpdate in processing state."""
    update = ImageGenerationUpdate(
        request_id="req-123",
        status="processing",
        progress_percent=50,
        update={"step": "generating"},
    )

    assert update.status == "processing"
    assert update.progress_percent == 50


def test_image_generation_update_completed():
    """Test ImageGenerationUpdate in completed state with result."""
    result = ImageGenerationResponse(
        request_id="req-123",
        images=["https://example.com/image1.png"],
        status="completed",
        raw_response={"status": "completed"},
    )

    update = ImageGenerationUpdate(
        request_id="req-123",
        status="completed",
        progress_percent=100,
        update={"metrics": {"inference_time": 2.5}},
        result=result,
    )

    assert update.status == "completed"
    assert update.progress_percent == 100
    assert update.result is not None
    assert update.result.request_id == "req-123"


def test_image_generation_update_failed():
    """Test ImageGenerationUpdate in failed state with error."""
    update = ImageGenerationUpdate(
        request_id="req-123",
        status="failed",
        progress_percent=None,
        update={},
        error="Content policy violation",
    )

    assert update.status == "failed"
    assert update.error == "Content policy violation"
    assert update.result is None
