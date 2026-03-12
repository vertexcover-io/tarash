"""Tests for batch generation API.

Each test references the REQ ID or EDGE ID it covers from
dev-docs/plans/batch-generation/SPEC.md.
"""

import pytest

from tarash.tarash_gateway.exceptions import TarashException
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


# ==================== Fixtures ====================


@pytest.fixture
def mock_video_config() -> VideoGenerationConfig:
    """Create a VideoGenerationConfig for testing."""
    return VideoGenerationConfig(
        model="fal-ai/test-model",
        provider="fal",
        api_key="test-key",
    )


@pytest.fixture
def mock_video_request() -> VideoGenerationRequest:
    """Create a VideoGenerationRequest for testing."""
    return VideoGenerationRequest(prompt="A test video prompt")


@pytest.fixture
def mock_video_response() -> VideoGenerationResponse:
    """Create a VideoGenerationResponse for testing."""
    return VideoGenerationResponse(
        request_id="test-req-001",
        video="https://example.com/video.mp4",
        status="completed",
        raw_response={"id": "test-req-001"},
    )


@pytest.fixture
def mock_image_config() -> ImageGenerationConfig:
    """Create an ImageGenerationConfig for testing."""
    return ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="test-key",
    )


@pytest.fixture
def mock_image_request() -> ImageGenerationRequest:
    """Create an ImageGenerationRequest for testing."""
    return ImageGenerationRequest(prompt="A test image prompt")


@pytest.fixture
def mock_image_response() -> ImageGenerationResponse:
    """Create an ImageGenerationResponse for testing."""
    return ImageGenerationResponse(
        request_id="test-req-001",
        images=["https://example.com/image.png"],
        status="completed",
        raw_response={"id": "test-req-001"},
    )


@pytest.fixture
def mock_audio_config() -> AudioGenerationConfig:
    """Create an AudioGenerationConfig for testing."""
    return AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="test-key",
    )


@pytest.fixture
def mock_tts_request() -> TTSRequest:
    """Create a TTSRequest for testing."""
    return TTSRequest(text="Hello, world!")


@pytest.fixture
def mock_tts_response() -> TTSResponse:
    """Create a TTSResponse for testing."""
    return TTSResponse(
        request_id="test-req-001",
        audio="base64audio==",
        status="completed",
        raw_response={"id": "test-req-001"},
    )


@pytest.fixture
def mock_sts_request() -> STSRequest:
    """Create an STSRequest for testing."""
    return STSRequest(audio="base64audio==", voice_id="voice-1")


@pytest.fixture
def mock_sts_response() -> STSResponse:
    """Create an STSResponse for testing."""
    return STSResponse(
        request_id="test-req-001",
        audio="base64audio==",
        status="completed",
        raw_response={"id": "test-req-001"},
    )


# ==================== Phase 1: Batch Model Tests ====================


def test_batch_item_model_fields():
    """REQ-1: BatchItem is a generic Pydantic model with request and optional config."""
    from tarash.tarash_gateway.models import BatchItem

    item = BatchItem(request=VideoGenerationRequest(prompt="test"))
    assert item.request.prompt == "test"
    assert item.config is None

    config = VideoGenerationConfig(model="test", provider="fal", api_key="key")
    item_with_config = BatchItem(
        request=VideoGenerationRequest(prompt="test"), config=config
    )
    assert item_with_config.config is not None
    assert item_with_config.config.model == "test"


def test_batch_item_result_model_fields():
    """REQ-2: BatchItemResult has index, status, response, and error fields."""
    from tarash.tarash_gateway.models import BatchItemResult

    # Completed result
    result = BatchItemResult(
        index=0,
        status="completed",
        response=VideoGenerationResponse(
            request_id="r1",
            video="https://example.com/v.mp4",
            status="completed",
            raw_response={},
        ),
    )
    assert result.index == 0
    assert result.status == "completed"
    assert result.response is not None
    assert result.error is None

    # Failed result
    err = TarashException("fail")
    failed = BatchItemResult(index=1, status="failed", error=err)
    assert failed.status == "failed"
    assert failed.response is None
    assert failed.error is not None


def test_batch_response_model_fields():
    """REQ-3: BatchResponse has results list, total, succeeded, and failed counts."""
    from tarash.tarash_gateway.models import BatchItemResult, BatchResponse

    results = [
        BatchItemResult(index=0, status="completed", response=None),
        BatchItemResult(index=1, status="failed", error=TarashException("x")),
    ]
    br = BatchResponse(results=results, total=2, succeeded=1, failed=1)
    assert len(br.results) == 2
    assert br.total == 2
    assert br.succeeded == 1
    assert br.failed == 1


def test_batch_completion_update_model_fields():
    """REQ-4: BatchCompletionUpdate has index, item_result, completed_count, total_count."""
    from tarash.tarash_gateway.models import BatchCompletionUpdate, BatchItemResult

    item_result = BatchItemResult(index=0, status="completed")
    update = BatchCompletionUpdate(
        index=0,
        item_result=item_result,
        completed_count=1,
        total_count=5,
    )
    assert update.index == 0
    assert update.item_result.status == "completed"
    assert update.completed_count == 1
    assert update.total_count == 5


def test_concrete_type_aliases_exist():
    """REQ-5: Concrete type aliases exist for all four modalities."""
    from tarash.tarash_gateway.models import (
        ImageBatchCompletionUpdate,
        ImageBatchItem,
        ImageBatchItemResult,
        ImageBatchResponse,
        STSBatchCompletionUpdate,
        STSBatchItem,
        STSBatchItemResult,
        STSBatchResponse,
        TTSBatchCompletionUpdate,
        TTSBatchItem,
        TTSBatchItemResult,
        TTSBatchResponse,
        VideoBatchCompletionUpdate,
        VideoBatchItem,
        VideoBatchItemResult,
        VideoBatchResponse,
    )

    # Video aliases
    assert VideoBatchItem is not None
    assert VideoBatchItemResult is not None
    assert VideoBatchResponse is not None
    assert VideoBatchCompletionUpdate is not None

    # Image aliases
    assert ImageBatchItem is not None
    assert ImageBatchItemResult is not None
    assert ImageBatchResponse is not None
    assert ImageBatchCompletionUpdate is not None

    # TTS aliases
    assert TTSBatchItem is not None
    assert TTSBatchItemResult is not None
    assert TTSBatchResponse is not None
    assert TTSBatchCompletionUpdate is not None

    # STS aliases
    assert STSBatchItem is not None
    assert STSBatchItemResult is not None
    assert STSBatchResponse is not None
    assert STSBatchCompletionUpdate is not None
