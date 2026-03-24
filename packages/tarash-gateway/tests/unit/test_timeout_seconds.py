"""Tests for the configurable timeout_seconds field on config models."""

import tarash.tarash_gateway.mock  # noqa: F401

from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    ImageGenerationConfig,
    VideoGenerationConfig,
)


# ==================== VideoGenerationConfig ====================


def test_video_config_default_timeout_seconds() -> None:
    """Video config defaults timeout_seconds to 300."""
    config = VideoGenerationConfig(model="fal-ai/veo3", provider="fal", api_key="k")
    assert config.timeout_seconds == 300


def test_video_config_custom_timeout_seconds() -> None:
    """Explicit timeout_seconds overrides default."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3", provider="fal", api_key="k", timeout_seconds=600
    )
    assert config.timeout_seconds == 600


def test_video_config_legacy_timeout_populates_timeout_seconds() -> None:
    """When only legacy timeout is set, timeout_seconds inherits its value."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3", provider="fal", api_key="k", timeout=900
    )
    assert config.timeout_seconds == 900
    assert config.timeout == 900


def test_video_config_timeout_seconds_takes_precedence() -> None:
    """When both timeout and timeout_seconds are set, timeout_seconds wins."""
    config = VideoGenerationConfig(
        model="fal-ai/veo3",
        provider="fal",
        api_key="k",
        timeout=900,
        timeout_seconds=450,
    )
    assert config.timeout_seconds == 450
    assert config.timeout == 900


def test_video_config_fallback_inherits_timeout_seconds() -> None:
    """Fallback configs also get their own timeout_seconds."""
    fallback = VideoGenerationConfig(
        model="openai/sora-2", provider="openai", api_key="k", timeout_seconds=180
    )
    config = VideoGenerationConfig(
        model="fal-ai/veo3",
        provider="fal",
        api_key="k",
        timeout_seconds=500,
        fallback_configs=[fallback],
    )
    assert config.timeout_seconds == 500
    assert config.fallback_configs is not None
    assert config.fallback_configs[0].timeout_seconds == 180


# ==================== ImageGenerationConfig ====================


def test_image_config_default_timeout_seconds() -> None:
    """Image config defaults timeout_seconds to 60."""
    config = ImageGenerationConfig(model="fal-ai/flux/dev", provider="fal", api_key="k")
    assert config.timeout_seconds == 60


def test_image_config_custom_timeout_seconds() -> None:
    """Explicit timeout_seconds overrides default."""
    config = ImageGenerationConfig(
        model="fal-ai/flux/dev", provider="fal", api_key="k", timeout_seconds=30
    )
    assert config.timeout_seconds == 30


def test_image_config_legacy_timeout_populates_timeout_seconds() -> None:
    """When only legacy timeout is set, timeout_seconds inherits its value."""
    config = ImageGenerationConfig(
        model="fal-ai/flux/dev", provider="fal", api_key="k", timeout=90
    )
    assert config.timeout_seconds == 90
    assert config.timeout == 90


def test_image_config_timeout_seconds_takes_precedence() -> None:
    """When both timeout and timeout_seconds are set, timeout_seconds wins."""
    config = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key="k",
        timeout=90,
        timeout_seconds=45,
    )
    assert config.timeout_seconds == 45
    assert config.timeout == 90


# ==================== AudioGenerationConfig ====================


def test_audio_config_default_timeout_seconds() -> None:
    """Audio config defaults timeout_seconds to 120."""
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2", provider="elevenlabs", api_key="k"
    )
    assert config.timeout_seconds == 120


def test_audio_config_custom_timeout_seconds() -> None:
    """Explicit timeout_seconds overrides default."""
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="k",
        timeout_seconds=60,
    )
    assert config.timeout_seconds == 60


def test_audio_config_legacy_timeout_populates_timeout_seconds() -> None:
    """When only legacy timeout is set, timeout_seconds inherits its value."""
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="k",
        timeout=180,
    )
    assert config.timeout_seconds == 180
    assert config.timeout == 180


def test_audio_config_timeout_seconds_takes_precedence() -> None:
    """When both timeout and timeout_seconds are set, timeout_seconds wins."""
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="k",
        timeout=180,
        timeout_seconds=90,
    )
    assert config.timeout_seconds == 90
    assert config.timeout == 180


def test_audio_config_fallback_inherits_timeout_seconds() -> None:
    """Fallback configs also get their own timeout_seconds."""
    fallback = AudioGenerationConfig(
        model="sonic-3", provider="cartesia", api_key="k", timeout_seconds=60
    )
    config = AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
        api_key="k",
        timeout_seconds=100,
        fallback_configs=[fallback],
    )
    assert config.timeout_seconds == 100
    assert config.fallback_configs is not None
    assert config.fallback_configs[0].timeout_seconds == 60
