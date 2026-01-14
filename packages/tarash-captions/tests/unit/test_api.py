"""Unit tests for caption generation API."""

from __future__ import annotations

from tarash.captions.api import generate_caption


def test_generate_caption_returns_response(caption_request, caption_config):
    """Generate caption returns a valid CaptionResponse."""
    response = generate_caption(caption_request, caption_config)

    assert response.text == f"Caption for: {caption_request.prompt}"
    assert response.language == caption_request.language


def test_generate_caption_with_different_language(caption_config):
    """Generate caption respects language parameter."""
    from tarash.captions.models import CaptionRequest

    request = CaptionRequest(
        prompt="Un vidéo de chat",
        language="fr",
    )
    response = generate_caption(request, caption_config)

    assert response.language == "fr"


def test_caption_request_validation():
    """CaptionRequest validates required fields."""
    from tarash.captions.models import CaptionRequest

    request = CaptionRequest(
        prompt="Test prompt",
    )

    assert request.prompt == "Test prompt"
    assert request.language == "en"  # Default value
    assert request.max_length is None  # Default value


def test_caption_config_validation():
    """CaptionConfig validates required fields."""
    from tarash.captions.models import CaptionConfig

    config = CaptionConfig(
        provider="openai",
        api_key="test-key",
    )

    assert config.provider == "openai"
    assert config.api_key == "test-key"
    assert config.model == "gpt-4o"  # Default value
