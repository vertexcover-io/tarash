"""Unit tests for Hume provider cost resolution (REQ-023)."""

from types import SimpleNamespace

from decimal import Decimal

import pytest

from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    TTSRequest,
)
from tarash.tarash_gateway.providers.hume import HumeProviderHandler


@pytest.fixture
def handler():
    return HumeProviderHandler()


@pytest.fixture
def hume_config():
    return AudioGenerationConfig(
        model="octave",
        provider="hume",
        api_key="test-api-key",
        timeout=240,
    )


@pytest.fixture
def tts_request():
    return TTSRequest(text="Hello world", voice_id="test-voice-id")


def _make_hume_result():
    """Create a mock Hume synthesize_json result."""
    generation = SimpleNamespace(
        audio="base64audio",
        generation_id="gen-123",
        duration=1.5,
    )
    return SimpleNamespace(generations=[generation], request_id="hume-req-123")


# REQ-023: Hume TTS uses len(request.text) as quantity
def test_tts_response_cost_uses_text_length(handler, hume_config, tts_request):
    """Hume TTS cost uses len(request.text) as quantity for pricing lookup."""
    hume_result = _make_hume_result()
    response = handler._convert_tts_response(
        hume_config, tts_request, "req-123", hume_result
    )
    assert response.cost is not None
    assert response.cost.raw_amount == len(tts_request.text)
    assert response.cost.raw_unit == "characters"
    expected_usd = Decimal("0.00015") * len(tts_request.text)
    assert response.cost.amount_usd == expected_usd


# REQ-023: octave-v2 model
def test_tts_response_cost_octave_v2(handler, tts_request):
    """Hume octave-v2 model has correct cost."""
    config = AudioGenerationConfig(
        model="octave-v2",
        provider="hume",
        api_key="test-api-key",
        timeout=240,
    )
    hume_result = _make_hume_result()
    response = handler._convert_tts_response(
        config, tts_request, "req-456", hume_result
    )
    assert response.cost is not None
    assert response.cost.raw_amount == len(tts_request.text)
    expected_usd = Decimal("0.0000076") * len(tts_request.text)
    assert response.cost.amount_usd == expected_usd


# EDGE-001: Unknown model returns cost=None
def test_tts_response_cost_unknown_model(handler, tts_request):
    """Unknown model returns cost=None."""
    config = AudioGenerationConfig(
        model="unknown_model",
        provider="hume",
        api_key="test-api-key",
        timeout=240,
    )
    hume_result = _make_hume_result()
    response = handler._convert_tts_response(
        config, tts_request, "req-789", hume_result
    )
    assert response.cost is None
