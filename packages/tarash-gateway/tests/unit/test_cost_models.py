"""Tests for cost fields on response models and metadata dataclasses.

Covers: REQ-002, REQ-003, REQ-004, REQ-005, REQ-006, REQ-007
Edge cases: EDGE-005, EDGE-006, EDGE-007
"""

from datetime import datetime, timezone

import pytest

from tarash.tarash_gateway.models import (
    AttemptMetadata,
    ExecutionMetadata,
    GenerationCost,
    ImageGenerationResponse,
    STSResponse,
    TTSResponse,
    VideoGenerationResponse,
)


# ---- Fixtures ----


@pytest.fixture
def sample_cost():
    """A typical GenerationCost instance."""
    return GenerationCost(amount_usd=0.40, raw_amount=8.0, raw_unit="seconds")


@pytest.fixture
def now():
    return datetime.now(tz=timezone.utc)


# ---- VideoGenerationResponse (REQ-002, EDGE-007) ----


def test_video_response_cost_defaults_to_none():
    """VideoGenerationResponse.cost defaults to None (EDGE-007)."""
    resp = VideoGenerationResponse(
        request_id="vid-1",
        video="https://example.com/video.mp4",
        status="completed",
        raw_response={},
    )
    assert resp.cost is None


def test_video_response_with_explicit_cost(sample_cost):
    """VideoGenerationResponse accepts an explicit GenerationCost (REQ-002)."""
    resp = VideoGenerationResponse(
        request_id="vid-2",
        video="https://example.com/video.mp4",
        status="completed",
        raw_response={},
        cost=sample_cost,
    )
    assert resp.cost is sample_cost
    assert resp.cost.amount_usd == 0.40
    assert resp.cost.raw_amount == 8.0
    assert resp.cost.raw_unit == "seconds"


# ---- ImageGenerationResponse (REQ-003, EDGE-007) ----


def test_image_response_cost_defaults_to_none():
    """ImageGenerationResponse.cost defaults to None (EDGE-007)."""
    resp = ImageGenerationResponse(
        request_id="img-1",
        images=["https://example.com/image.png"],
        status="completed",
        raw_response={},
    )
    assert resp.cost is None


def test_image_response_with_explicit_cost(sample_cost):
    """ImageGenerationResponse accepts an explicit GenerationCost (REQ-003)."""
    resp = ImageGenerationResponse(
        request_id="img-2",
        images=["https://example.com/image.png"],
        status="completed",
        raw_response={},
        cost=sample_cost,
    )
    assert resp.cost is sample_cost


# ---- TTSResponse (REQ-004, EDGE-007) ----


def test_tts_response_cost_defaults_to_none():
    """TTSResponse.cost defaults to None (EDGE-007)."""
    resp = TTSResponse(
        request_id="tts-1",
        audio="base64data",
        status="completed",
        raw_response={},
    )
    assert resp.cost is None


def test_tts_response_with_explicit_cost(sample_cost):
    """TTSResponse accepts an explicit GenerationCost (REQ-004)."""
    resp = TTSResponse(
        request_id="tts-2",
        audio="base64data",
        status="completed",
        raw_response={},
        cost=sample_cost,
    )
    assert resp.cost is sample_cost


# ---- STSResponse (REQ-005, EDGE-007) ----


def test_sts_response_cost_defaults_to_none():
    """STSResponse.cost defaults to None (EDGE-007)."""
    resp = STSResponse(
        request_id="sts-1",
        audio="base64data",
        status="completed",
        raw_response={},
    )
    assert resp.cost is None


def test_sts_response_with_explicit_cost(sample_cost):
    """STSResponse accepts an explicit GenerationCost (REQ-005)."""
    resp = STSResponse(
        request_id="sts-2",
        audio="base64data",
        status="completed",
        raw_response={},
        cost=sample_cost,
    )
    assert resp.cost is sample_cost


# ---- AttemptMetadata (REQ-006, EDGE-005) ----


def test_attempt_metadata_positional_construction_without_cost(now):
    """Existing positional construction still works without cost arg (EDGE-005)."""
    am = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3",
        attempt_number=1,
        started_at=now,
        ended_at=now,
        status="success",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-1",
    )
    assert am.cost is None
    assert am.provider == "fal"
    assert am.request_id == "req-1"


def test_attempt_metadata_with_explicit_cost(now, sample_cost):
    """AttemptMetadata accepts an explicit cost (REQ-006)."""
    am = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3",
        attempt_number=1,
        started_at=now,
        ended_at=now,
        status="success",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-2",
        cost=sample_cost,
    )
    assert am.cost is sample_cost
    assert am.cost.amount_usd == 0.40


# ---- ExecutionMetadata (REQ-007, EDGE-006) ----


def test_execution_metadata_positional_construction_without_total_cost(now):
    """Existing positional construction still works without total_cost_usd (EDGE-006)."""
    am = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3",
        attempt_number=1,
        started_at=now,
        ended_at=now,
        status="success",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-1",
    )
    em = ExecutionMetadata(
        total_attempts=1,
        successful_attempt=1,
        attempts=[am],
        fallback_triggered=False,
        configs_in_chain=1,
    )
    assert em.total_cost_usd is None
    assert em.total_attempts == 1
    assert em.configs_in_chain == 1


def test_execution_metadata_with_explicit_total_cost(now):
    """ExecutionMetadata accepts an explicit total_cost_usd (REQ-007)."""
    am = AttemptMetadata(
        provider="fal",
        model="fal-ai/veo3",
        attempt_number=1,
        started_at=now,
        ended_at=now,
        status="success",
        error_type=None,
        error_message=None,
        is_retryable=None,
        request_id="req-1",
    )
    em = ExecutionMetadata(
        total_attempts=1,
        successful_attempt=1,
        attempts=[am],
        fallback_triggered=False,
        configs_in_chain=1,
        total_cost_usd=1.50,
    )
    assert em.total_cost_usd == 1.50
