"""Tests for Runway provider cost resolution (REQ-024)."""

from unittest.mock import MagicMock

import pytest

from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.pricing import PRICING_TABLE
from tarash.tarash_gateway.providers.runway import RunwayProviderHandler


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    """Create Runway handler instance."""
    return RunwayProviderHandler()


@pytest.fixture
def base_config():
    """Config for gen4_turbo."""
    return VideoGenerationConfig(
        model="gen4_turbo",
        provider="runway",
        api_key="test-key",
    )


@pytest.fixture
def base_request():
    """Create basic video request."""
    return VideoGenerationRequest(prompt="A sunset over mountains")


# ==================== Tests ====================


def test_convert_response_includes_cost_with_duration(
    handler, base_config, base_request
):
    """REQ-024: Runway uses output duration as quantity for cost."""
    task = MagicMock()
    task.status = "SUCCEEDED"
    task.id = "task-123"
    task.output = ["https://example.com/video.mp4"]

    # Runway doesn't have output duration on the task object directly,
    # so we use request.duration_seconds if available, otherwise quantity=1
    response = handler._convert_response(base_config, base_request, "req-123", task)

    assert response.cost is not None
    entry = PRICING_TABLE[("runway", "gen4_turbo")]
    assert response.cost.raw_unit == entry.unit
    # Without duration info on the task, uses quantity=1.0 as fallback
    assert response.cost.amount_usd is not None


def test_convert_response_gen45_model(handler, base_request):
    """REQ-024: gen4.5 model resolves cost correctly."""
    config = VideoGenerationConfig(
        model="gen4.5",
        provider="runway",
        api_key="test-key",
    )
    task = MagicMock()
    task.status = "SUCCEEDED"
    task.id = "task-456"
    task.output = ["https://example.com/video.mp4"]

    response = handler._convert_response(config, base_request, "req-456", task)

    assert response.cost is not None
    assert response.cost.raw_unit == "seconds"


def test_convert_response_unknown_model_no_cost(handler, base_request):
    """EDGE-001: Unknown Runway model returns cost=None."""
    config = VideoGenerationConfig(
        model="gen99_unknown",
        provider="runway",
        api_key="test-key",
    )
    task = MagicMock()
    task.status = "SUCCEEDED"
    task.id = "task-789"
    task.output = ["https://example.com/video.mp4"]

    response = handler._convert_response(config, base_request, "req-789", task)

    assert response.cost is None
