"""Unit tests for Sarvam provider cost resolution (REQ-022)."""

from types import SimpleNamespace

import pytest

from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    TTSRequest,
)
from tarash.tarash_gateway.providers.sarvam import SarvamProviderHandler


@pytest.fixture
def handler():
    return SarvamProviderHandler()


@pytest.fixture
def sarvam_config():
    return AudioGenerationConfig(
        model="bulbul-v2",
        provider="sarvam",
        api_key="test-api-key",
        timeout=240,
    )


@pytest.fixture
def tts_request():
    return TTSRequest(
        text="Hello world",
        voice_id="test-voice-id",
        language_code="en-IN",
    )


# REQ-022: Sarvam TTS uses len(request.text) as quantity
def test_tts_response_cost_uses_text_length(handler, sarvam_config, tts_request):
    """Sarvam TTS cost uses len(request.text) as quantity for pricing lookup."""
    sarvam_result = SimpleNamespace(audios=["base64audio"], request_id="srv-123")
    response = handler._convert_tts_response(
        sarvam_config, tts_request, "req-123", sarvam_result
    )
    assert response.cost is not None
    assert response.cost.raw_amount == len(tts_request.text)
    assert response.cost.raw_unit == "characters"
    expected_usd = 0.000018 * len(tts_request.text)
    assert response.cost.amount_usd == pytest.approx(expected_usd)


# REQ-022: Different model (bulbul-v3)
def test_tts_response_cost_v3_model(handler, tts_request):
    """Sarvam bulbul-v3 model has correct cost."""
    config = AudioGenerationConfig(
        model="bulbul-v3",
        provider="sarvam",
        api_key="test-api-key",
        timeout=240,
    )
    sarvam_result = SimpleNamespace(audios=["base64audio"], request_id="srv-456")
    response = handler._convert_tts_response(
        config, tts_request, "req-456", sarvam_result
    )
    assert response.cost is not None
    assert response.cost.raw_amount == len(tts_request.text)
    expected_usd = 0.000036 * len(tts_request.text)
    assert response.cost.amount_usd == pytest.approx(expected_usd)


# EDGE-001: Unknown model returns cost=None
def test_tts_response_cost_unknown_model(handler, tts_request):
    """Unknown model returns cost=None."""
    config = AudioGenerationConfig(
        model="unknown_model",
        provider="sarvam",
        api_key="test-api-key",
        timeout=240,
    )
    sarvam_result = SimpleNamespace(audios=["base64audio"], request_id="srv-789")
    response = handler._convert_tts_response(
        config, tts_request, "req-789", sarvam_result
    )
    assert response.cost is None
