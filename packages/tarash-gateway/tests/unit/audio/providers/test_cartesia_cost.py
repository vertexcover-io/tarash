"""Unit tests for Cartesia provider cost resolution (REQ-021)."""

from decimal import Decimal

import pytest

from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    STSRequest,
    TTSRequest,
)
from tarash.tarash_gateway.providers.cartesia import CartesiaProviderHandler


@pytest.fixture
def handler():
    return CartesiaProviderHandler()


@pytest.fixture
def cartesia_config():
    return AudioGenerationConfig(
        model="sonic",
        provider="cartesia",
        api_key="test-api-key",
        timeout=240,
    )


@pytest.fixture
def tts_request():
    return TTSRequest(text="Hello world", voice_id="test-voice-id")


# REQ-021: Cartesia TTS uses len(request.text) as quantity
def test_tts_response_cost_uses_text_length(handler, cartesia_config, tts_request):
    """Cartesia TTS cost uses len(request.text) as quantity for pricing lookup."""
    response = handler._convert_tts_response(
        cartesia_config, tts_request, "req-123", b"fake-audio"
    )
    assert response.cost is not None
    assert response.cost.raw_amount == len(tts_request.text)
    assert response.cost.raw_unit == "characters"
    expected_usd = Decimal("0.000011") * len(tts_request.text)
    assert response.cost.amount_usd == expected_usd


# EDGE-001: Unknown model returns cost=None
def test_tts_response_cost_unknown_model(handler, tts_request):
    """Unknown model returns cost=None."""
    config = AudioGenerationConfig(
        model="unknown_model",
        provider="cartesia",
        api_key="test-api-key",
        timeout=240,
    )
    response = handler._convert_tts_response(
        config, tts_request, "req-789", b"fake-audio"
    )
    assert response.cost is None


# STS: No text attribute on STSRequest, cost should be None
def test_sts_response_cost_is_none(handler, cartesia_config):
    """STS response has cost=None since STSRequest has no text attribute."""
    sts_request = STSRequest(
        audio={"content": b"fake-audio", "content_type": "audio/wav"},
        voice_id="test-voice-id",
    )
    response = handler._convert_sts_response(
        cartesia_config, sts_request, "req-sts", b"fake-audio"
    )
    assert response.cost is None
