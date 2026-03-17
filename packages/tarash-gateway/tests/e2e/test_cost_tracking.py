"""E2E tests for generation cost tracking.

Verifies that real API calls return cost data in responses.
Each test asserts:
  - response.cost is not None
  - response.cost.amount_usd > 0
  - response.cost.raw_unit matches expected unit
  - response.cost.raw_amount > 0
  - execution_metadata.total_cost_usd > 0
"""

import os

import pytest

from tarash.tarash_gateway import generate_image, generate_tts, generate_video
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    ImageGenerationConfig,
    ImageGenerationRequest,
    TTSRequest,
    VideoGenerationConfig,
    VideoGenerationRequest,
)


# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def fal_api_key():
    key = os.getenv("FAL_KEY")
    if not key:
        pytest.skip("FAL_KEY not set")
    return key


@pytest.fixture(scope="module")
def openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def google_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def stability_api_key():
    key = os.getenv("STABILITY_API_KEY")
    if not key:
        pytest.skip("STABILITY_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def xai_api_key():
    key = os.getenv("XAI_API_KEY")
    if not key:
        pytest.skip("XAI_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def sarvam_api_key():
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        pytest.skip("SARVAM_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def hume_api_key():
    key = os.getenv("HUME_API_KEY")
    if not key:
        pytest.skip("HUME_API_KEY not set")
    return key


# ==================== Helper ====================


def assert_cost(response, expected_unit):
    """Common cost assertions for any response type."""
    assert response.cost is not None, "response.cost should not be None"
    assert response.cost.amount_usd is not None, "amount_usd should not be None"
    assert response.cost.amount_usd > 0, (
        f"amount_usd should be positive, got {response.cost.amount_usd}"
    )
    assert response.cost.raw_amount > 0, (
        f"raw_amount should be positive, got {response.cost.raw_amount}"
    )
    assert response.cost.raw_unit == expected_unit, (
        f"expected raw_unit={expected_unit!r}, got {response.cost.raw_unit!r}"
    )

    meta = response.execution_metadata
    assert meta is not None, "execution_metadata should not be None"
    assert meta.total_cost_usd is not None, "total_cost_usd should not be None"
    assert meta.total_cost_usd > 0, (
        f"total_cost_usd should be positive, got {meta.total_cost_usd}"
    )

    # Print cost for visibility in test output
    print(
        f"  Cost: ${response.cost.amount_usd:.6f}"
        f" ({response.cost.raw_amount} {response.cost.raw_unit})"
    )
    print(f"  Total cost: ${meta.total_cost_usd:.6f}")


# ==================== Image Cost Tests ====================


def test_fal_image_cost(fal_api_key):
    """Fal flux/schnell image generation returns cost in megapixels."""
    config = ImageGenerationConfig(
        provider="fal",
        model="fal-ai/flux/schnell",
        api_key=fal_api_key,
        timeout=120,
        max_poll_attempts=60,
        poll_interval=2,
    )
    request = ImageGenerationRequest(prompt="A red circle on white background")

    response = generate_image(config, request)

    assert response.status == "completed"
    assert_cost(response, expected_unit="megapixels")


def test_openai_dalle3_image_cost(openai_api_key):
    """OpenAI DALL-E 3 image generation returns cost in images (flat rate)."""
    config = ImageGenerationConfig(
        provider="openai",
        model="dall-e-3",
        api_key=openai_api_key,
        timeout=120,
    )
    request = ImageGenerationRequest(
        prompt="A simple blue square",
        size="1024x1024",
    )

    response = generate_image(config, request)

    assert response.status == "completed"
    assert_cost(response, expected_unit="images")


def test_google_imagen_image_cost(google_api_key):
    """Google Imagen image generation returns cost in images."""
    config = ImageGenerationConfig(
        provider="google",
        model="imagen-3.0-generate-002",
        api_key=google_api_key,
        timeout=120,
    )
    request = ImageGenerationRequest(prompt="A green triangle")

    response = generate_image(config, request)

    assert response.status == "completed"
    assert_cost(response, expected_unit="images")


def test_stability_image_cost(stability_api_key):
    """Stability sd3.5-medium image generation returns cost in images."""
    config = ImageGenerationConfig(
        provider="stability",
        model="sd3.5-medium",
        api_key=stability_api_key,
        timeout=120,
    )
    request = ImageGenerationRequest(prompt="A yellow star")

    response = generate_image(config, request)

    assert response.status == "completed"
    assert_cost(response, expected_unit="images")


def test_xai_image_cost(xai_api_key):
    """xAI grok-imagine-image returns cost in images."""
    config = ImageGenerationConfig(
        provider="xai",
        model="grok-imagine-image",
        api_key=xai_api_key,
        timeout=120,
    )
    request = ImageGenerationRequest(prompt="A purple diamond shape")

    response = generate_image(config, request)

    assert response.status == "completed"
    assert_cost(response, expected_unit="images")


# ==================== Video Cost Tests ====================


def test_fal_video_cost(fal_api_key):
    """Fal minimax video generation returns cost in videos (flat rate)."""
    config = VideoGenerationConfig(
        provider="fal",
        model="fal-ai/minimax",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )
    request = VideoGenerationRequest(
        prompt="A ball rolling across a table",
        duration_seconds=6,
        aspect_ratio="16:9",
    )

    response = generate_video(config, request)

    assert response.status == "completed"
    assert_cost(response, expected_unit="videos")


# ==================== TTS Cost Tests ====================


def test_sarvam_tts_cost(sarvam_api_key):
    """Sarvam TTS returns cost in characters."""
    config = AudioGenerationConfig(
        provider="sarvam",
        model="bulbul:v2",
        api_key=sarvam_api_key,
        timeout=60,
    )
    text = "This is a cost tracking test."
    request = TTSRequest(text=text, language_code="en-IN")

    response = generate_tts(config, request)

    assert response.status == "completed"
    assert_cost(response, expected_unit="characters")
    # Verify raw_amount matches text length
    assert response.cost.raw_amount == len(text)


def test_hume_tts_cost(hume_api_key):
    """Hume TTS returns cost in characters."""
    config = AudioGenerationConfig(
        provider="hume",
        model="octave",
        api_key=hume_api_key,
        timeout=60,
    )
    text = "This is a cost tracking test."
    request = TTSRequest(text=text)

    response = generate_tts(config, request)

    assert response.status == "completed"
    assert_cost(response, expected_unit="characters")
    # Verify raw_amount matches text length
    assert response.cost.raw_amount == len(text)
