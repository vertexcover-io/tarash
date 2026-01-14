"""Test configuration and fixtures."""

from __future__ import annotations

import pytest

from tarash.captions.models import CaptionConfig, CaptionRequest


@pytest.fixture
def caption_request():
    """Create a basic caption request."""
    return CaptionRequest(
        prompt="A video of a cat playing piano",
        language="en",
    )


@pytest.fixture
def caption_config():
    """Create a basic caption config."""
    return CaptionConfig(
        provider="openai",
        api_key="test-key",
        model="gpt-4o",
    )
