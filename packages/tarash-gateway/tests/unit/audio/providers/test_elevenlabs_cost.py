"""Unit tests for ElevenLabs provider cost resolution (REQ-020)."""

import pytest

from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    STSRequest,
    TTSRequest,
)
from tarash.tarash_gateway.providers.elevenlabs import ElevenLabsProviderHandler


@pytest.fixture
def handler():
    return ElevenLabsProviderHandler()


@pytest.fixture
def elevenlabs_config():
    return AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="test-api-key",
        timeout=240,
    )


@pytest.fixture
def tts_request():
    return TTSRequest(text="Hello world", voice_id="test-voice-id")


# REQ-020: ElevenLabs TTS uses len(request.text) as quantity
def test_tts_response_cost_uses_text_length(handler, elevenlabs_config, tts_request):
    """ElevenLabs TTS cost uses len(request.text) as quantity for pricing lookup."""
    response = handler._convert_tts_response(
        elevenlabs_config, tts_request, "req-123", b"fake-audio"
    )
    assert response.cost is not None
    assert response.cost.raw_amount == len(tts_request.text)
    assert response.cost.raw_unit == "characters"
    expected_usd = 0.00024 * len(tts_request.text)
    assert response.cost.amount_usd == pytest.approx(expected_usd)


# REQ-020: Different model in the pricing table
def test_tts_response_cost_turbo_model(handler, tts_request):
    """ElevenLabs eleven_turbo_v2 model has correct cost."""
    config = AudioGenerationConfig(
        model="eleven_turbo_v2",
        provider="elevenlabs",
        api_key="test-api-key",
        timeout=240,
    )
    response = handler._convert_tts_response(
        config, tts_request, "req-456", b"fake-audio"
    )
    assert response.cost is not None
    assert response.cost.raw_amount == len(tts_request.text)
    assert response.cost.raw_unit == "characters"


# EDGE-001: Unknown model returns cost=None
def test_tts_response_cost_unknown_model(handler, tts_request):
    """Unknown model returns cost=None."""
    config = AudioGenerationConfig(
        model="unknown_model",
        provider="elevenlabs",
        api_key="test-api-key",
        timeout=240,
    )
    response = handler._convert_tts_response(
        config, tts_request, "req-789", b"fake-audio"
    )
    assert response.cost is None


# STS: No text attribute on STSRequest, cost should be None
def test_sts_response_cost_is_none(handler, elevenlabs_config):
    """STS response has cost=None since STSRequest has no text attribute."""
    sts_request = STSRequest(
        audio={"content": b"fake-audio", "content_type": "audio/wav"},
        voice_id="test-voice-id",
    )
    response = handler._convert_sts_response(
        elevenlabs_config, sts_request, "req-sts", b"fake-audio"
    )
    assert response.cost is None
