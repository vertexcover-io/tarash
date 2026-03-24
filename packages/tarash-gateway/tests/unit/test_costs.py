"""Tests for cost estimation module."""

import tarash.tarash_gateway.mock  # noqa: F401
from tarash.tarash_gateway.costs import CostEstimate, estimate_cost
from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    VideoGenerationConfig,
)


# ==================== CostEstimate Tests ====================


def test_cost_estimate_avg():
    est = CostEstimate(min_usd=0.10, max_usd=0.30, model="m", provider="p")
    assert est.avg_usd == 0.20


def test_cost_estimate_frozen():
    est = CostEstimate(min_usd=0.10, max_usd=0.30, model="m", provider="p")
    try:
        est.min_usd = 0.50  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass


# ==================== Video Cost Estimation Tests ====================


def test_fal_video_exact_match():
    config = VideoGenerationConfig(provider="fal", model="fal-ai/minimax")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.21
    assert result.max_usd == 0.21
    assert result.model == "fal-ai/minimax"
    assert result.provider == "fal"


def test_fal_video_prefix_match():
    config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3.1/fast")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.25
    assert result.max_usd == 0.50


def test_fal_video_longest_prefix_wins():
    """veo3.1 prefix should match over veo3 prefix."""
    config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3.1/image-to-video")
    result = estimate_cost(config)
    assert result is not None
    assert result.model == "fal-ai/veo3.1/image-to-video"
    # Should match veo3.1 cost, not veo3
    assert result.min_usd == 0.25


def test_fal_veo3_prefix_match():
    config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3/preview")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.25


def test_openai_video():
    config = VideoGenerationConfig(provider="openai", model="sora")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.20
    assert result.max_usd == 0.60


def test_google_video():
    config = VideoGenerationConfig(provider="google", model="veo-3")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.25
    assert result.max_usd == 0.50


# ==================== Image Cost Estimation Tests ====================


def test_fal_image_exact_match():
    config = ImageGenerationConfig(provider="fal", model="fal-ai/flux-2")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.03
    assert result.max_usd == 0.06


def test_fal_image_prefix_match():
    config = ImageGenerationConfig(provider="fal", model="fal-ai/flux/dev")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.01
    assert result.max_usd == 0.05


def test_openai_image_dall_e_3():
    config = ImageGenerationConfig(provider="openai", model="dall-e-3")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.04
    assert result.max_usd == 0.12


def test_openai_image_gpt_image():
    config = ImageGenerationConfig(provider="openai", model="gpt-image-1.5")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.02
    assert result.max_usd == 0.19


def test_google_image_imagen3():
    config = ImageGenerationConfig(provider="google", model="imagen-3.0-generate-002")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.02
    assert result.max_usd == 0.04


def test_google_image_prefix_match():
    config = ImageGenerationConfig(provider="google", model="imagen-3")
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.02


def test_google_image_gemini_flash():
    config = ImageGenerationConfig(
        provider="google", model="gemini-2.5-flash-image-preview"
    )
    result = estimate_cost(config)
    assert result is not None
    assert result.min_usd == 0.01
    assert result.max_usd == 0.04


# ==================== Unknown Model/Provider Tests ====================


def test_unknown_provider_returns_none():
    config = VideoGenerationConfig(provider="unknown-provider", model="some-model")
    result = estimate_cost(config)
    assert result is None


def test_unknown_model_returns_none():
    config = VideoGenerationConfig(provider="fal", model="fal-ai/nonexistent-model")
    result = estimate_cost(config)
    assert result is None


def test_unknown_image_model_returns_none():
    config = ImageGenerationConfig(provider="openai", model="nonexistent-model")
    result = estimate_cost(config)
    assert result is None
