"""Tests for batch generation API.

Each test references the REQ ID or EDGE ID it covers from
dev-docs/plans/batch-generation/SPEC.md.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.exceptions import TarashException, ValidationError
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


# ==================== Phase 2: Batch Execution Engine Tests ====================


async def test_empty_batch_returns_empty_response(mock_video_config):
    """EDGE-1: Empty batch returns empty BatchResponse immediately."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchResponse

    mock_fn = AsyncMock()
    result = await _execute_batch_async(
        items=[],
        default_config=mock_video_config,
        execute_fn=mock_fn,
    )
    assert isinstance(result, BatchResponse)
    assert result.total == 0
    assert result.succeeded == 0
    assert result.failed == 0
    assert result.results == []
    mock_fn.assert_not_called()


async def test_single_item_batch(
    mock_video_config, mock_video_request, mock_video_response
):
    """EDGE-2: Single-item batch works correctly."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    mock_fn = AsyncMock(return_value=mock_video_response)
    items = [BatchItem(request=mock_video_request)]

    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mock_fn,
    )
    assert result.total == 1
    assert result.succeeded == 1
    assert result.failed == 0
    assert result.results[0].status == "completed"
    assert result.results[0].response == mock_video_response


async def test_default_max_concurrent_is_five(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-10: max_concurrent defaults to 5."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    max_seen_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def tracking_fn(config, request, on_progress=None):
        nonlocal max_seen_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            max_seen_concurrent = max(max_seen_concurrent, current_concurrent)
        await asyncio.sleep(0.01)
        async with lock:
            current_concurrent -= 1
        return mock_video_response

    items = [BatchItem(request=mock_video_request) for _ in range(10)]
    await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=tracking_fn,
    )
    assert max_seen_concurrent <= 5


async def test_max_concurrent_zero_raises(mock_video_config):
    """REQ-11, EDGE-9: max_concurrent=0 raises ValidationError."""
    from tarash.tarash_gateway.batch import _execute_batch_async

    with pytest.raises(ValidationError):
        await _execute_batch_async(
            items=[],
            default_config=mock_video_config,
            execute_fn=AsyncMock(),
            max_concurrent=0,
        )


async def test_max_concurrent_negative_raises(mock_video_config):
    """REQ-11, EDGE-11: max_concurrent=-1 raises ValidationError."""
    from tarash.tarash_gateway.batch import _execute_batch_async

    with pytest.raises(ValidationError):
        await _execute_batch_async(
            items=[],
            default_config=mock_video_config,
            execute_fn=AsyncMock(),
            max_concurrent=-1,
        )


async def test_max_concurrent_51_raises(mock_video_config):
    """REQ-11, EDGE-10: max_concurrent=51 raises ValidationError."""
    from tarash.tarash_gateway.batch import _execute_batch_async

    with pytest.raises(ValidationError):
        await _execute_batch_async(
            items=[],
            default_config=mock_video_config,
            execute_fn=AsyncMock(),
            max_concurrent=51,
        )


async def test_max_concurrent_boundaries_valid(mock_video_config):
    """REQ-11: max_concurrent=1 and max_concurrent=50 are valid."""
    from tarash.tarash_gateway.batch import _execute_batch_async

    # Both should not raise
    await _execute_batch_async(
        items=[],
        default_config=mock_video_config,
        execute_fn=AsyncMock(),
        max_concurrent=1,
    )
    await _execute_batch_async(
        items=[],
        default_config=mock_video_config,
        execute_fn=AsyncMock(),
        max_concurrent=50,
    )


async def test_concurrency_respects_semaphore(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-9, EDGE-4, EDGE-5: Semaphore controls concurrency."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    max_seen = 0
    current = 0
    lock = asyncio.Lock()

    async def tracking_fn(config, request, on_progress=None):
        nonlocal max_seen, current
        async with lock:
            current += 1
            max_seen = max(max_seen, current)
        await asyncio.sleep(0.01)
        async with lock:
            current -= 1
        return mock_video_response

    items = [BatchItem(request=mock_video_request) for _ in range(8)]
    await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=tracking_fn,
        max_concurrent=3,
    )
    assert max_seen <= 3


async def test_item_config_override_used(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-12: Per-item config overrides the default config."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    override_config = VideoGenerationConfig(
        model="override-model", provider="fal", api_key="override-key"
    )
    captured_configs = []

    async def capture_fn(config, request, on_progress=None):
        captured_configs.append(config)
        return mock_video_response

    items = [BatchItem(request=mock_video_request, config=override_config)]
    await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=capture_fn,
    )
    assert captured_configs[0].model == "override-model"


async def test_item_config_none_uses_default(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-12, EDGE-6: Item with config=None uses default config."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    captured_configs = []

    async def capture_fn(config, request, on_progress=None):
        captured_configs.append(config)
        return mock_video_response

    items = [BatchItem(request=mock_video_request)]  # config=None by default
    await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=capture_fn,
    )
    assert captured_configs[0].model == "fal-ai/test-model"


async def test_items_dispatched_concurrently(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-13: Items are dispatched as concurrent tasks via asyncio.gather."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    call_times = []

    async def timing_fn(config, request, on_progress=None):
        call_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.05)
        return mock_video_response

    items = [BatchItem(request=mock_video_request) for _ in range(3)]
    await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=timing_fn,
        max_concurrent=3,
    )
    # All 3 should start nearly simultaneously
    assert len(call_times) == 3
    assert max(call_times) - min(call_times) < 0.03


async def test_calls_single_request_function(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-14: Each item calls the execute_fn with resolved config and request."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    mock_fn = AsyncMock(return_value=mock_video_response)
    items = [BatchItem(request=mock_video_request)]

    await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mock_fn,
    )
    mock_fn.assert_called_once_with(
        mock_video_config, mock_video_request, on_progress=None
    )


async def test_success_produces_completed_result(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-15: Successful call produces completed BatchItemResult."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    mock_fn = AsyncMock(return_value=mock_video_response)
    items = [BatchItem(request=mock_video_request)]

    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mock_fn,
    )
    assert result.results[0].index == 0
    assert result.results[0].status == "completed"
    assert result.results[0].response == mock_video_response
    assert result.results[0].error is None


async def test_tarash_exception_produces_failed_result(
    mock_video_config, mock_video_request
):
    """REQ-16: TarashException produces failed BatchItemResult."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    err = TarashException("provider error", provider="fal")
    mock_fn = AsyncMock(side_effect=err)
    items = [BatchItem(request=mock_video_request)]

    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mock_fn,
    )
    assert result.results[0].status == "failed"
    assert result.results[0].response is None
    assert isinstance(result.results[0].error, TarashException)
    assert result.results[0].error.message == "provider error"


async def test_generic_exception_wrapped_in_tarash_exception(
    mock_video_config, mock_video_request
):
    """REQ-17: Non-TarashException is wrapped in TarashException."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    mock_fn = AsyncMock(side_effect=RuntimeError("unexpected"))
    items = [BatchItem(request=mock_video_request)]

    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mock_fn,
    )
    assert result.results[0].status == "failed"
    assert isinstance(result.results[0].error, TarashException)
    assert "unexpected" in str(result.results[0].error)


async def test_on_item_progress_passed_to_single_request(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-18: on_item_progress is forwarded to execute_fn as on_progress."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    progress_cb = MagicMock()
    mock_fn = AsyncMock(return_value=mock_video_response)
    items = [BatchItem(request=mock_video_request)]

    await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mock_fn,
        on_item_progress=progress_cb,
    )
    mock_fn.assert_called_once_with(
        mock_video_config, mock_video_request, on_progress=progress_cb
    )


async def test_on_batch_progress_called_per_item(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-19: on_batch_progress is called once per completed item."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    updates = []
    mock_fn = AsyncMock(return_value=mock_video_response)
    items = [BatchItem(request=mock_video_request) for _ in range(3)]

    await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mock_fn,
        on_batch_progress=lambda u: updates.append(u),
    )
    assert len(updates) == 3


async def test_on_batch_progress_called_after_result_recorded(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-20: on_batch_progress is called after item result is recorded."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    updates = []
    mock_fn = AsyncMock(return_value=mock_video_response)
    items = [BatchItem(request=mock_video_request)]

    await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mock_fn,
        on_batch_progress=lambda u: updates.append(u),
    )
    # completed_count should reflect the just-completed item
    assert updates[0].completed_count == 1
    assert updates[0].total_count == 1


async def test_on_batch_progress_exception_logged_and_continues(
    mock_video_config, mock_video_request, mock_video_response, caplog
):
    """REQ-21, EDGE-7: on_batch_progress exception is logged, batch continues."""

    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    def bad_callback(update):
        raise RuntimeError("callback exploded")

    mock_fn = AsyncMock(return_value=mock_video_response)
    items = [BatchItem(request=mock_video_request) for _ in range(2)]

    with caplog.at_level(logging.WARNING):
        result = await _execute_batch_async(
            items=items,
            default_config=mock_video_config,
            execute_fn=mock_fn,
            on_batch_progress=bad_callback,
        )

    # Batch should complete successfully despite callback error
    assert result.total == 2
    assert result.succeeded == 2
    assert "callback exploded" in caplog.text


async def test_results_ordered_by_submission_index(
    mock_video_config, mock_video_request
):
    """REQ-22: Results are ordered by original submission index."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    async def variable_delay_fn(config, request, on_progress=None):
        # Items complete in reverse order
        idx = int(request.prompt.split("-")[1])
        await asyncio.sleep((3 - idx) * 0.01)
        return VideoGenerationResponse(
            request_id=f"req-{idx}",
            video=f"https://example.com/{idx}.mp4",
            status="completed",
            raw_response={},
        )

    items = [
        BatchItem(request=VideoGenerationRequest(prompt=f"prompt-{i}"))
        for i in range(3)
    ]
    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=variable_delay_fn,
        max_concurrent=3,
    )
    for i, item_result in enumerate(result.results):
        assert item_result.index == i


async def test_batch_response_total_count(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-23: BatchResponse.total equals len(items)."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    mock_fn = AsyncMock(return_value=mock_video_response)
    items = [BatchItem(request=mock_video_request) for _ in range(5)]

    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mock_fn,
    )
    assert result.total == 5


async def test_batch_response_succeeded_count(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-24: BatchResponse.succeeded counts completed items."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    call_count = 0

    async def mixed_fn(config, request, on_progress=None):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise TarashException("fail")
        return mock_video_response

    items = [BatchItem(request=mock_video_request) for _ in range(3)]
    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mixed_fn,
        max_concurrent=1,  # sequential to control which one fails
    )
    assert result.succeeded == 2


async def test_batch_response_failed_count(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-25: BatchResponse.failed counts failed items."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    call_count = 0

    async def mixed_fn(config, request, on_progress=None):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise TarashException("fail")
        return mock_video_response

    items = [BatchItem(request=mock_video_request) for _ in range(3)]
    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mixed_fn,
        max_concurrent=1,
    )
    assert result.failed == 1


async def test_succeeded_plus_failed_equals_total(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-26: succeeded + failed == total."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    call_count = 0

    async def mixed_fn(config, request, on_progress=None):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:
            raise TarashException("fail")
        return mock_video_response

    items = [BatchItem(request=mock_video_request) for _ in range(4)]
    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mixed_fn,
        max_concurrent=1,
    )
    assert result.succeeded + result.failed == result.total


async def test_all_items_fail(mock_video_config, mock_video_request):
    """EDGE-3: All items fail, batch returns with all failed, no exception raised."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    mock_fn = AsyncMock(side_effect=TarashException("all fail"))
    items = [BatchItem(request=mock_video_request) for _ in range(3)]

    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=mock_fn,
    )
    assert result.total == 3
    assert result.succeeded == 0
    assert result.failed == 3
    for r in result.results:
        assert r.status == "failed"


async def test_sequential_execution_max_concurrent_one(
    mock_video_config, mock_video_request, mock_video_response
):
    """EDGE-4: max_concurrent=1 means items execute one at a time."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    max_seen = 0
    current = 0
    lock = asyncio.Lock()

    async def tracking_fn(config, request, on_progress=None):
        nonlocal max_seen, current
        async with lock:
            current += 1
            max_seen = max(max_seen, current)
        await asyncio.sleep(0.01)
        async with lock:
            current -= 1
        return mock_video_response

    items = [BatchItem(request=mock_video_request) for _ in range(5)]
    await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=tracking_fn,
        max_concurrent=1,
    )
    assert max_seen == 1


async def test_large_batch_with_concurrency_limit(
    mock_video_config, mock_video_request, mock_video_response
):
    """EDGE-12: Large batch (100+ items) with max_concurrent=5."""
    from tarash.tarash_gateway.batch import _execute_batch_async
    from tarash.tarash_gateway.models import BatchItem

    max_seen = 0
    current = 0
    lock = asyncio.Lock()

    async def tracking_fn(config, request, on_progress=None):
        nonlocal max_seen, current
        async with lock:
            current += 1
            max_seen = max(max_seen, current)
        await asyncio.sleep(0.001)
        async with lock:
            current -= 1
        return mock_video_response

    items = [BatchItem(request=mock_video_request) for _ in range(100)]
    result = await _execute_batch_async(
        items=items,
        default_config=mock_video_config,
        execute_fn=tracking_fn,
        max_concurrent=5,
    )
    assert max_seen <= 5
    assert result.total == 100
    assert result.succeeded == 100
    for i, r in enumerate(result.results):
        assert r.index == i


# ==================== Phase 3: API Function Tests ====================


def test_all_eight_batch_functions_importable():
    """REQ-6: All 8 batch functions are importable from api module."""
    from tarash.tarash_gateway.api import (
        generate_image_batch,
        generate_image_batch_async,
        generate_sts_batch,
        generate_sts_batch_async,
        generate_tts_batch,
        generate_tts_batch_async,
        generate_video_batch,
        generate_video_batch_async,
    )

    assert callable(generate_video_batch)
    assert callable(generate_video_batch_async)
    assert callable(generate_image_batch)
    assert callable(generate_image_batch_async)
    assert callable(generate_tts_batch)
    assert callable(generate_tts_batch_async)
    assert callable(generate_sts_batch)
    assert callable(generate_sts_batch_async)


async def test_async_batch_function_signature(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-7: Async batch functions accept config, items, max_concurrent, callbacks."""
    from tarash.tarash_gateway.api import generate_video_batch_async
    from tarash.tarash_gateway.models import VideoBatchItem

    items = [VideoBatchItem(request=mock_video_request)]

    with patch(
        "tarash.tarash_gateway.batch._execute_batch_async",
        new_callable=AsyncMock,
    ) as mock_exec:
        from tarash.tarash_gateway.models import BatchResponse, BatchItemResult

        mock_exec.return_value = BatchResponse(
            results=[
                BatchItemResult(
                    index=0, status="completed", response=mock_video_response
                )
            ],
            total=1,
            succeeded=1,
            failed=0,
        )
        result = await generate_video_batch_async(
            config=mock_video_config,
            items=items,
            max_concurrent=3,
            on_item_progress=None,
            on_batch_progress=None,
        )
        assert result.total == 1


def test_sync_batch_calls_asyncio_run(
    mock_video_config, mock_video_request, mock_video_response
):
    """REQ-8: Sync batch function calls asyncio.run on async counterpart."""
    from tarash.tarash_gateway.api import generate_video_batch
    from tarash.tarash_gateway.models import (
        BatchItemResult,
        BatchResponse,
        VideoBatchItem,
    )

    items = [VideoBatchItem(request=mock_video_request)]

    with patch(
        "tarash.tarash_gateway.batch._execute_batch_async",
        new_callable=AsyncMock,
    ) as mock_exec:
        mock_exec.return_value = BatchResponse(
            results=[
                BatchItemResult(
                    index=0, status="completed", response=mock_video_response
                )
            ],
            total=1,
            succeeded=1,
            failed=0,
        )
        result = generate_video_batch(
            config=mock_video_config,
            items=items,
        )
        assert result.total == 1


# ==================== Phase 4: Export Tests ====================


def test_batch_functions_exported_from_init():
    """REQ-27: All 8 batch API functions are exported from __init__."""
    import tarash.tarash_gateway as tg

    assert hasattr(tg, "generate_video_batch")
    assert hasattr(tg, "generate_video_batch_async")
    assert hasattr(tg, "generate_image_batch")
    assert hasattr(tg, "generate_image_batch_async")
    assert hasattr(tg, "generate_tts_batch")
    assert hasattr(tg, "generate_tts_batch_async")
    assert hasattr(tg, "generate_sts_batch")
    assert hasattr(tg, "generate_sts_batch_async")

    # Check __all__
    for name in [
        "generate_video_batch",
        "generate_video_batch_async",
        "generate_image_batch",
        "generate_image_batch_async",
        "generate_tts_batch",
        "generate_tts_batch_async",
        "generate_sts_batch",
        "generate_sts_batch_async",
    ]:
        assert name in tg.__all__


def test_batch_models_exported_from_init():
    """REQ-28: All batch model types exported from __init__."""
    import tarash.tarash_gateway as tg

    model_names = [
        "BatchItem",
        "BatchItemResult",
        "BatchResponse",
        "BatchCompletionUpdate",
        "VideoBatchItem",
        "VideoBatchItemResult",
        "VideoBatchResponse",
        "VideoBatchCompletionUpdate",
        "ImageBatchItem",
        "ImageBatchItemResult",
        "ImageBatchResponse",
        "ImageBatchCompletionUpdate",
        "TTSBatchItem",
        "TTSBatchItemResult",
        "TTSBatchResponse",
        "TTSBatchCompletionUpdate",
        "STSBatchItem",
        "STSBatchItemResult",
        "STSBatchResponse",
        "STSBatchCompletionUpdate",
    ]
    for name in model_names:
        assert hasattr(tg, name), f"{name} not exported"
        assert name in tg.__all__, f"{name} not in __all__"
