"""Tests for orchestrator cost aggregation (Phase 3).

Covers: REQ-014, REQ-015, REQ-016, REQ-017, REQ-032
Edge cases: EDGE-003, EDGE-004, EDGE-012
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.exceptions import GenerationFailedError
from tarash.tarash_gateway.models import (
    GenerationCost,
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    AudioGenerationConfig,
    STSRequest,
    STSResponse,
    TTSRequest,
    TTSResponse,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.orchestrator import (
    ExecutionOrchestrator,
    _compute_total_cost_usd,
)


# ==================== Fixtures ====================


@pytest.fixture
def orchestrator():
    return ExecutionOrchestrator()


@pytest.fixture
def video_config():
    return VideoGenerationConfig(
        provider="fal",
        model="fal-ai/veo3",
        api_key="test-key",
    )


@pytest.fixture
def video_request():
    return VideoGenerationRequest(prompt="A test video")


@pytest.fixture
def image_config():
    return ImageGenerationConfig(
        provider="openai",
        model="gpt-image-1",
        api_key="test-key",
    )


@pytest.fixture
def image_request():
    return ImageGenerationRequest(prompt="A test image")


@pytest.fixture
def tts_config():
    return AudioGenerationConfig(
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        api_key="test-key",
    )


@pytest.fixture
def tts_request():
    return TTSRequest(text="Hello world", voice_id="test-voice")


@pytest.fixture
def sts_config():
    return AudioGenerationConfig(
        provider="elevenlabs",
        model="eleven_english_sts_v2",
        api_key="test-key",
    )


@pytest.fixture
def sts_request():
    return STSRequest(
        audio="https://example.com/audio.wav",
        voice_id="test-voice",
    )


def _make_video_response(cost=None, request_id="req-123"):
    return VideoGenerationResponse(
        request_id=request_id,
        video="https://example.com/video.mp4",
        status="completed",
        cost=cost,
        raw_response={"test": True},
    )


def _make_image_response(cost=None, request_id="req-123"):
    return ImageGenerationResponse(
        request_id=request_id,
        images=["https://example.com/image.png"],
        status="completed",
        cost=cost,
        raw_response={"test": True},
    )


def _make_tts_response(cost=None, request_id="req-123"):
    return TTSResponse(
        request_id=request_id,
        audio="dGVzdA==",
        status="completed",
        cost=cost,
        raw_response={"test": True},
    )


def _make_sts_response(cost=None, request_id="req-123"):
    return STSResponse(
        request_id=request_id,
        audio="dGVzdA==",
        status="completed",
        cost=cost,
        raw_response={"test": True},
    )


# ==================== _compute_total_cost_usd unit tests ====================


def test_compute_total_cost_usd_single_attempt_with_cost():
    """Single successful attempt with known cost -> total equals that cost (EDGE-004)."""
    from tarash.tarash_gateway.models import AttemptMetadata

    cost = GenerationCost(amount_usd=0.50, raw_amount=10.0, raw_unit="seconds")
    attempt = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3",
        attempt_number=1,
        started_at=datetime.now(),
        ended_at=datetime.now(),
        status="success",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-1",
        cost=cost,
    )
    result = _compute_total_cost_usd([attempt])
    assert result == 0.50


def test_compute_total_cost_usd_multiple_attempts_all_with_costs():
    """Multiple attempts all with costs -> total is sum (REQ-015)."""
    from tarash.tarash_gateway.models import AttemptMetadata

    cost1 = GenerationCost(amount_usd=0.50, raw_amount=10.0, raw_unit="seconds")
    cost2 = GenerationCost(amount_usd=1.00, raw_amount=20.0, raw_unit="seconds")
    attempts = [
        AttemptMetadata(
            provider="fal",
            model="m1",
            attempt_number=1,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="failed",
            error_type="GenerationFailedError",
            error_message="fail",
            is_retryable=True,
            request_id=None,
            cost=cost1,
        ),
        AttemptMetadata(
            provider="fal",
            model="m2",
            attempt_number=2,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="success",
            error_type=None,
            error_message=None,
            is_retryable=None,
            request_id="req-2",
            cost=cost2,
        ),
    ]
    result = _compute_total_cost_usd(attempts)
    assert result == 1.50


def test_compute_total_cost_usd_one_attempt_cost_none():
    """One attempt with cost=None -> total is None (REQ-016)."""
    from tarash.tarash_gateway.models import AttemptMetadata

    cost = GenerationCost(amount_usd=0.50, raw_amount=10.0, raw_unit="seconds")
    attempts = [
        AttemptMetadata(
            provider="fal",
            model="m1",
            attempt_number=1,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="failed",
            error_type="GenerationFailedError",
            error_message="fail",
            is_retryable=True,
            request_id=None,
            cost=None,  # no cost
        ),
        AttemptMetadata(
            provider="fal",
            model="m2",
            attempt_number=2,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="success",
            error_type=None,
            error_message=None,
            is_retryable=None,
            request_id="req-2",
            cost=cost,
        ),
    ]
    result = _compute_total_cost_usd(attempts)
    assert result is None


def test_compute_total_cost_usd_one_attempt_amount_usd_none():
    """One attempt with cost.amount_usd=None -> total is None (REQ-017)."""
    from tarash.tarash_gateway.models import AttemptMetadata

    cost_with = GenerationCost(amount_usd=0.50, raw_amount=10.0, raw_unit="seconds")
    cost_without = GenerationCost(amount_usd=None, raw_amount=10.0, raw_unit="seconds")
    attempts = [
        AttemptMetadata(
            provider="fal",
            model="m1",
            attempt_number=1,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="failed",
            error_type="GenerationFailedError",
            error_message="fail",
            is_retryable=True,
            request_id=None,
            cost=cost_without,
        ),
        AttemptMetadata(
            provider="fal",
            model="m2",
            attempt_number=2,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="success",
            error_type=None,
            error_message=None,
            is_retryable=None,
            request_id="req-2",
            cost=cost_with,
        ),
    ]
    result = _compute_total_cost_usd(attempts)
    assert result is None


def test_compute_total_cost_usd_all_failed_no_costs():
    """All attempts failed, all have cost=None -> total is None (EDGE-012)."""
    from tarash.tarash_gateway.models import AttemptMetadata

    attempts = [
        AttemptMetadata(
            provider="fal",
            model="m1",
            attempt_number=1,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="failed",
            error_type="GenerationFailedError",
            error_message="fail",
            is_retryable=True,
            request_id=None,
            cost=None,
        ),
        AttemptMetadata(
            provider="fal",
            model="m2",
            attempt_number=2,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="failed",
            error_type="GenerationFailedError",
            error_message="fail",
            is_retryable=True,
            request_id=None,
            cost=None,
        ),
    ]
    result = _compute_total_cost_usd(attempts)
    assert result is None


def test_compute_total_cost_usd_fallback_2_fail_1_succeed():
    """Fallback chain: 2 fail + 1 succeed -> total is None because failed have cost=None (EDGE-003)."""
    from tarash.tarash_gateway.models import AttemptMetadata

    cost = GenerationCost(amount_usd=1.00, raw_amount=10.0, raw_unit="seconds")
    attempts = [
        AttemptMetadata(
            provider="fal",
            model="m1",
            attempt_number=1,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="failed",
            error_type="GenerationFailedError",
            error_message="fail",
            is_retryable=True,
            request_id=None,
            cost=None,  # failed, no cost
        ),
        AttemptMetadata(
            provider="fal",
            model="m2",
            attempt_number=2,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="failed",
            error_type="GenerationFailedError",
            error_message="fail",
            is_retryable=True,
            request_id=None,
            cost=None,  # failed, no cost
        ),
        AttemptMetadata(
            provider="fal",
            model="m3",
            attempt_number=3,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            status="success",
            error_type=None,
            error_message=None,
            is_retryable=None,
            request_id="req-3",
            cost=cost,
        ),
    ]
    result = _compute_total_cost_usd(attempts)
    assert result is None


# ==================== Orchestrator integration tests ====================
# Test that the orchestrator populates attempt_metadata.cost and
# execution_metadata.total_cost_usd (REQ-014, REQ-032)


def test_execute_sync_populates_cost(orchestrator, video_config, video_request):
    """Video sync: cost from response is set on attempt and total_cost_usd (REQ-014, REQ-032)."""
    cost = GenerationCost(amount_usd=0.40, raw_amount=8.0, raw_unit="seconds")
    response = _make_video_response(cost=cost)

    mock_handler = MagicMock()
    mock_handler.generate_video.return_value = response

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", return_value=mock_handler
    ):
        result = orchestrator.execute_sync(video_config, video_request)

    assert result.execution_metadata is not None
    assert result.execution_metadata.attempts[0].cost == cost
    assert result.execution_metadata.total_cost_usd == 0.40


@pytest.mark.asyncio
async def test_execute_async_populates_cost(orchestrator, video_config, video_request):
    """Video async: cost from response is set on attempt and total_cost_usd (REQ-014, REQ-032)."""
    cost = GenerationCost(amount_usd=0.40, raw_amount=8.0, raw_unit="seconds")
    response = _make_video_response(cost=cost)

    mock_handler = AsyncMock()
    mock_handler.generate_video_async.return_value = response

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", return_value=mock_handler
    ):
        result = await orchestrator.execute_async(video_config, video_request)

    assert result.execution_metadata is not None
    assert result.execution_metadata.attempts[0].cost == cost
    assert result.execution_metadata.total_cost_usd == 0.40


def test_execute_image_sync_populates_cost(orchestrator, image_config, image_request):
    """Image sync: cost from response is set on attempt and total_cost_usd (REQ-032)."""
    cost = GenerationCost(amount_usd=0.08, raw_amount=1.0, raw_unit="image")
    response = _make_image_response(cost=cost)

    mock_handler = MagicMock()
    mock_handler.generate_image.return_value = response

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", return_value=mock_handler
    ):
        result = orchestrator.execute_image_sync(image_config, image_request)

    assert result.execution_metadata is not None
    assert result.execution_metadata.attempts[0].cost == cost
    assert result.execution_metadata.total_cost_usd == 0.08


@pytest.mark.asyncio
async def test_execute_image_async_populates_cost(
    orchestrator, image_config, image_request
):
    """Image async: cost from response is set on attempt and total_cost_usd (REQ-032)."""
    cost = GenerationCost(amount_usd=0.08, raw_amount=1.0, raw_unit="image")
    response = _make_image_response(cost=cost)

    mock_handler = AsyncMock()
    mock_handler.generate_image_async.return_value = response

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", return_value=mock_handler
    ):
        result = await orchestrator.execute_image_async(image_config, image_request)

    assert result.execution_metadata is not None
    assert result.execution_metadata.attempts[0].cost == cost
    assert result.execution_metadata.total_cost_usd == 0.08


def test_execute_tts_sync_populates_cost(orchestrator, tts_config, tts_request):
    """TTS sync: cost from response is set on attempt and total_cost_usd (REQ-032)."""
    cost = GenerationCost(amount_usd=0.003, raw_amount=11.0, raw_unit="characters")
    response = _make_tts_response(cost=cost)

    mock_handler = MagicMock()
    mock_handler.generate_tts.return_value = response

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", return_value=mock_handler
    ):
        result = orchestrator.execute_tts_sync(tts_config, tts_request)

    assert result.execution_metadata is not None
    assert result.execution_metadata.attempts[0].cost == cost
    assert result.execution_metadata.total_cost_usd == 0.003


@pytest.mark.asyncio
async def test_execute_tts_async_populates_cost(orchestrator, tts_config, tts_request):
    """TTS async: cost from response is set on attempt and total_cost_usd (REQ-032)."""
    cost = GenerationCost(amount_usd=0.003, raw_amount=11.0, raw_unit="characters")
    response = _make_tts_response(cost=cost)

    mock_handler = AsyncMock()
    mock_handler.generate_tts_async.return_value = response

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", return_value=mock_handler
    ):
        result = await orchestrator.execute_tts_async(tts_config, tts_request)

    assert result.execution_metadata is not None
    assert result.execution_metadata.attempts[0].cost == cost
    assert result.execution_metadata.total_cost_usd == 0.003


def test_execute_sts_sync_populates_cost(orchestrator, sts_config, sts_request):
    """STS sync: cost from response is set on attempt and total_cost_usd (REQ-032)."""
    cost = GenerationCost(amount_usd=0.005, raw_amount=100.0, raw_unit="characters")
    response = _make_sts_response(cost=cost)

    mock_handler = MagicMock()
    mock_handler.generate_sts.return_value = response

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", return_value=mock_handler
    ):
        result = orchestrator.execute_sts_sync(sts_config, sts_request)

    assert result.execution_metadata is not None
    assert result.execution_metadata.attempts[0].cost == cost
    assert result.execution_metadata.total_cost_usd == 0.005


@pytest.mark.asyncio
async def test_execute_sts_async_populates_cost(orchestrator, sts_config, sts_request):
    """STS async: cost from response is set on attempt and total_cost_usd (REQ-032)."""
    cost = GenerationCost(amount_usd=0.005, raw_amount=100.0, raw_unit="characters")
    response = _make_sts_response(cost=cost)

    mock_handler = AsyncMock()
    mock_handler.generate_sts_async.return_value = response

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", return_value=mock_handler
    ):
        result = await orchestrator.execute_sts_async(sts_config, sts_request)

    assert result.execution_metadata is not None
    assert result.execution_metadata.attempts[0].cost == cost
    assert result.execution_metadata.total_cost_usd == 0.005


def test_execute_sync_no_cost_on_response(orchestrator, video_config, video_request):
    """Video sync: response with cost=None -> attempt.cost=None, total_cost_usd=None."""
    response = _make_video_response(cost=None)

    mock_handler = MagicMock()
    mock_handler.generate_video.return_value = response

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler", return_value=mock_handler
    ):
        result = orchestrator.execute_sync(video_config, video_request)

    assert result.execution_metadata is not None
    assert result.execution_metadata.attempts[0].cost is None
    assert result.execution_metadata.total_cost_usd is None


def test_execute_sync_fallback_cost_aggregation(orchestrator, video_request):
    """Fallback: first fails (no cost), second succeeds with cost -> total is None (EDGE-003)."""
    fallback_config = VideoGenerationConfig(
        provider="runway",
        model="gen4_turbo",
        api_key="test-key",
    )
    primary_config = VideoGenerationConfig(
        provider="fal",
        model="fal-ai/veo3",
        api_key="test-key",
        fallback_configs=[fallback_config],
    )

    cost = GenerationCost(amount_usd=1.00, raw_amount=10.0, raw_unit="seconds")
    success_response = _make_video_response(cost=cost, request_id="req-fallback")

    mock_handler_fail = MagicMock()
    mock_handler_fail.generate_video.side_effect = GenerationFailedError(
        message="Generation failed", provider="fal", model="fal-ai/veo3"
    )

    mock_handler_success = MagicMock()
    mock_handler_success.generate_video.return_value = success_response

    call_count = 0

    def get_handler_side_effect(cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_handler_fail
        return mock_handler_success

    with patch(
        "tarash.tarash_gateway.orchestrator.get_handler",
        side_effect=get_handler_side_effect,
    ):
        result = orchestrator.execute_sync(primary_config, video_request)

    assert result.execution_metadata is not None
    assert len(result.execution_metadata.attempts) == 2
    # First attempt failed - no cost
    assert result.execution_metadata.attempts[0].cost is None
    assert result.execution_metadata.attempts[0].status == "failed"
    # Second attempt succeeded with cost
    assert result.execution_metadata.attempts[1].cost == cost
    assert result.execution_metadata.attempts[1].status == "success"
    # Total is None because first attempt has cost=None
    assert result.execution_metadata.total_cost_usd is None
    assert result.execution_metadata.fallback_triggered is True
