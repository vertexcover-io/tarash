"""End-to-end tests for mock video generation through public API.

Tests the integration between VideoGenerationConfig.mock and the public API (generate_video/generate_video_async).
The unit tests in test_mock.py cover the mock functionality in detail. These E2E tests verify:
1. Mock interception works correctly through the public API
2. Mock disabled behavior falls through to provider selection
"""

import pytest

from tarash.tarash_gateway.video.api import generate_video, generate_video_async
from tarash.tarash_gateway.video.exceptions import ValidationError
from tarash.tarash_gateway.video.mock import MockConfig, MockResponse
from tarash.tarash_gateway.video.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
)

# Rebuild VideoGenerationConfig now that MockConfig is imported
VideoGenerationConfig.model_rebuild()


# ==================== E2E: Mock Interception ====================


def test_e2e_mock_enabled_intercepts_sync():
    """E2E: Mock enabled intercepts sync API call and returns mock response."""
    config = VideoGenerationConfig(
        model="mock-model",
        provider="mock",
        api_key="fake-key",
        mock=MockConfig(enabled=True),
    )

    request = VideoGenerationRequest(
        prompt="Test video",
        aspect_ratio="16:9",
        resolution="1080p",
        duration_seconds=4,
    )

    response = generate_video(config, request)

    # Verify mock interception worked
    assert response.status == "completed"
    assert response.is_mock is True
    assert response.request_id.startswith("mock_")
    assert response.video is not None


@pytest.mark.asyncio
async def test_e2e_mock_enabled_intercepts_async():
    """E2E: Mock enabled intercepts async API call and returns mock response."""
    config = VideoGenerationConfig(
        model="mock-model",
        provider="mock",
        api_key="fake-key",
        mock=MockConfig(enabled=True),
    )

    request = VideoGenerationRequest(
        prompt="Test video",
        aspect_ratio="9:16",
        resolution="720p",
        duration_seconds=4,
    )

    response = await generate_video_async(config, request)

    # Verify mock interception worked
    assert response.status == "completed"
    assert response.is_mock is True
    assert response.request_id.startswith("mock_")
    assert response.video is not None


# ==================== E2E: Mock Disabled ====================


def test_e2e_mock_disabled_uses_provider_sync():
    """E2E: Mock disabled falls through to provider (which fails for 'mock' provider)."""
    config = VideoGenerationConfig(
        model="mock-model",
        provider="mock",  # "mock" is not a real provider
        api_key="fake-key",
        mock=MockConfig(enabled=False),  # Mock disabled
    )

    request = VideoGenerationRequest(prompt="Should fail")

    # When mock is disabled, should try to use "mock" provider which doesn't exist
    with pytest.raises(ValidationError, match="Unsupported provider: mock"):
        generate_video(config, request)


@pytest.mark.asyncio
async def test_e2e_mock_disabled_uses_provider_async():
    """E2E: Mock disabled falls through to provider (async)."""
    config = VideoGenerationConfig(
        model="mock-model",
        provider="mock",  # "mock" is not a real provider
        api_key="fake-key",
        mock=MockConfig(enabled=False),  # Mock disabled
    )

    request = VideoGenerationRequest(prompt="Should fail")

    # When mock is disabled, should try to use "mock" provider which doesn't exist
    with pytest.raises(ValidationError, match="Unsupported provider: mock"):
        await generate_video_async(config, request)


def test_e2e_mock_none_uses_provider():
    """E2E: Mock=None (not configured) uses real provider."""
    config = VideoGenerationConfig(
        model="mock-model",
        provider="mock",  # "mock" is not a real provider
        api_key="fake-key",
        mock=None,  # No mock configuration
    )

    request = VideoGenerationRequest(prompt="Should fail")

    # Should try to use "mock" provider which doesn't exist
    with pytest.raises(ValidationError, match="Unsupported provider: mock"):
        generate_video(config, request)


# ==================== E2E: Mock Error Propagation ====================


def test_e2e_mock_error_propagates_through_api():
    """E2E: Mock errors propagate correctly through public API."""
    error = ValidationError("Test error from mock", provider="mock")
    config = VideoGenerationConfig(
        model="mock-model",
        provider="mock",
        api_key="fake-key",
        mock=MockConfig(
            enabled=True,
            responses=[MockResponse(weight=1.0, error=error)],
        ),
    )

    request = VideoGenerationRequest(prompt="Test")

    # Error should propagate through API
    with pytest.raises(ValidationError, match="Test error from mock"):
        generate_video(config, request)
