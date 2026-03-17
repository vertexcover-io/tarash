"""Tests for mock cost passthrough (REQ-031, EDGE-010, EDGE-011)."""

import pytest

from tarash.tarash_gateway.mock import (
    MockConfig,
    MockResponse,
    handle_mock_request_async,
    handle_mock_request_sync,
)
from tarash.tarash_gateway.models import (
    GenerationCost,
    VideoGenerationRequest,
)


@pytest.fixture
def basic_request():
    """Create a basic video generation request."""
    return VideoGenerationRequest(
        prompt="A cat playing piano",
        aspect_ratio="16:9",
        resolution="1080p",
        duration_seconds=4,
    )


# REQ-031, EDGE-010: MockResponse with explicit cost passes it through
def test_mock_response_with_explicit_cost_sync(basic_request):
    """MockResponse with explicit cost passes it through to the response."""
    cost = GenerationCost(amount_usd=0.50, raw_amount=4.0, raw_unit="seconds")
    config = MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0, cost=cost)],
    )

    response = handle_mock_request_sync(config, basic_request)

    assert response.cost is not None
    assert response.cost == cost
    assert response.cost.amount_usd == 0.50
    assert response.cost.raw_amount == 4.0
    assert response.cost.raw_unit == "seconds"


# EDGE-011: MockResponse with cost=None (default) → response has cost=None
def test_mock_response_with_default_cost_none_sync(basic_request):
    """MockResponse with default cost=None results in response cost=None."""
    config = MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0)],
    )

    response = handle_mock_request_sync(config, basic_request)

    assert response.cost is None


# REQ-031: Async handler also passes cost through
async def test_mock_response_with_explicit_cost_async(basic_request):
    """Async handler passes explicit cost through to the response."""
    cost = GenerationCost(amount_usd=1.25, raw_amount=10.0, raw_unit="seconds")
    config = MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0, cost=cost)],
    )

    response = await handle_mock_request_async(config, basic_request)

    assert response.cost is not None
    assert response.cost == cost
    assert response.cost.amount_usd == 1.25
    assert response.cost.raw_amount == 10.0
    assert response.cost.raw_unit == "seconds"
