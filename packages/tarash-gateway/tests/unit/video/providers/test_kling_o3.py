"""Tests for Kling O3 field mappers and request conversion."""

import pytest

from tarash.tarash_gateway.exceptions import ValidationError
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest
from tarash.tarash_gateway.providers.fal import (
    FalProviderHandler,
    KLING_O3_FIELD_MAPPERS,
    get_field_mappers,
)


@pytest.fixture
def handler():
    return FalProviderHandler()


# ==================== Field Mapper Selection Tests ====================


def test_get_field_mappers_kling_o3_all_variants():
    """Test Kling O3 mapper selection for all six variants."""
    # Pro variants
    assert (
        get_field_mappers("fal-ai/kling-video/o3/pro/text-to-video")
        is KLING_O3_FIELD_MAPPERS
    )
    assert (
        get_field_mappers("fal-ai/kling-video/o3/pro/image-to-video")
        is KLING_O3_FIELD_MAPPERS
    )
    assert (
        get_field_mappers("fal-ai/kling-video/o3/pro/reference-to-video")
        is KLING_O3_FIELD_MAPPERS
    )

    # Standard variants
    assert (
        get_field_mappers("fal-ai/kling-video/o3/standard/text-to-video")
        is KLING_O3_FIELD_MAPPERS
    )
    assert (
        get_field_mappers("fal-ai/kling-video/o3/standard/image-to-video")
        is KLING_O3_FIELD_MAPPERS
    )
    assert (
        get_field_mappers("fal-ai/kling-video/o3/standard/reference-to-video")
        is KLING_O3_FIELD_MAPPERS
    )

    # Prefix match
    assert get_field_mappers("fal-ai/kling-video/o3/") is KLING_O3_FIELD_MAPPERS


# ==================== Text-to-Video Conversion Tests ====================


def test_kling_o3_text_to_video_minimal(handler):
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/pro/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A cinematic scene of a warrior in battle",
    )

    result = handler._convert_request(config, request)

    assert result == {"prompt": "A cinematic scene of a warrior in battle"}


def test_kling_o3_text_to_video_full(handler):
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/standard/text-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="A character says <<<voice_1>>>Hello world<<<voice_1>>>",
        duration_seconds=10,
        aspect_ratio="16:9",
        generate_audio=True,
        extra_params={
            "voice_ids": ["voice_abc123"],
            "multi_prompt": [
                {"prompt": "Scene 1", "duration": "5"},
                {"prompt": "Scene 2", "duration": "5"},
            ],
            "shot_type": "customize",
        },
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "A character says <<<voice_1>>>Hello world<<<voice_1>>>"
    assert result["duration"] == "10"
    assert result["aspect_ratio"] == "16:9"
    assert result["generate_audio"] is True
    assert result["voice_ids"] == ["voice_abc123"]
    assert result["multi_prompt"] == [
        {"prompt": "Scene 1", "duration": "5"},
        {"prompt": "Scene 2", "duration": "5"},
    ]
    assert result["shot_type"] == "customize"


# ==================== Image-to-Video Conversion Tests ====================


def test_kling_o3_image_to_video_conversion(handler):
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/pro/image-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Animate this scene with gentle motion",
        image_list=[
            {"image": "https://example.com/start.jpg", "type": "reference"},
        ],
        duration_seconds=5,
        generate_audio=True,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Animate this scene with gentle motion"
    assert result["image_url"] == "https://example.com/start.jpg"
    assert result["duration"] == "5"
    assert result["generate_audio"] is True
    assert "start_image_url" not in result


def test_kling_o3_image_to_video_with_end_frame(handler):
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/standard/image-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Smooth transition from start to end",
        image_list=[
            {"image": "https://example.com/start.jpg", "type": "reference"},
            {"image": "https://example.com/end.jpg", "type": "last_frame"},
        ],
        duration_seconds=8,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Smooth transition from start to end"
    assert result["image_url"] == "https://example.com/start.jpg"
    assert result["end_image_url"] == "https://example.com/end.jpg"
    assert result["duration"] == "8"


# ==================== Reference-to-Video Conversion Tests ====================


def test_kling_o3_reference_to_video_conversion(handler):
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/pro/reference-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="@Element1 walks through a forest",
        image_list=[
            {"image": "https://example.com/first.jpg", "type": "first_frame"},
        ],
        duration_seconds=10,
        aspect_ratio="16:9",
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "@Element1 walks through a forest"
    assert result["start_image_url"] == "https://example.com/first.jpg"
    assert result["duration"] == "10"
    assert result["aspect_ratio"] == "16:9"


def test_kling_o3_reference_to_video_with_elements(handler):
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/standard/reference-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="@Element1 and @Element2 having a conversation",
        image_list=[
            {"image": "https://example.com/start.jpg", "type": "first_frame"},
            {"image": "https://example.com/end.jpg", "type": "last_frame"},
        ],
        duration_seconds=15,
        aspect_ratio="9:16",
        extra_params={
            "elements": [
                {
                    "frontal_image_url": "https://example.com/char1-front.jpg",
                    "reference_image_urls": ["https://example.com/char1-side.jpg"],
                },
                {
                    "frontal_image_url": "https://example.com/char2-front.jpg",
                },
            ]
        },
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "@Element1 and @Element2 having a conversation"
    assert result["start_image_url"] == "https://example.com/start.jpg"
    assert result["end_image_url"] == "https://example.com/end.jpg"
    assert result["duration"] == "15"
    assert result["aspect_ratio"] == "9:16"
    assert len(result["elements"]) == 2
    assert (
        result["elements"][0]["frontal_image_url"]
        == "https://example.com/char1-front.jpg"
    )
    assert result["elements"][0]["reference_image_urls"] == [
        "https://example.com/char1-side.jpg"
    ]
    assert (
        result["elements"][1]["frontal_image_url"]
        == "https://example.com/char2-front.jpg"
    )


def test_kling_o3_reference_to_video_with_image_urls(handler):
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/pro/reference-to-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="Scene in style of @Image1 and @Image2",
        image_list=[
            {"image": "https://example.com/style1.jpg", "type": "reference"},
            {"image": "https://example.com/style2.jpg", "type": "reference"},
        ],
        duration_seconds=5,
    )

    result = handler._convert_request(config, request)

    assert result["prompt"] == "Scene in style of @Image1 and @Image2"
    assert result["image_urls"] == [
        "https://example.com/style1.jpg",
        "https://example.com/style2.jpg",
    ]
    assert result["duration"] == "5"


# ==================== Duration Validation Tests ====================


def test_kling_o3_duration_validation_valid(handler):
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/pro/text-to-video",
        provider="fal",
        api_key="test-key",
    )

    for valid_duration in range(3, 16):
        request = VideoGenerationRequest(
            prompt="Test prompt",
            duration_seconds=valid_duration,
        )
        result = handler._convert_request(config, request)
        assert result["duration"] == str(valid_duration)


def test_kling_o3_duration_validation_invalid(handler):
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/standard/text-to-video",
        provider="fal",
        api_key="test-key",
    )

    for invalid_duration in [2, 16, 20, 30]:
        request = VideoGenerationRequest(
            prompt="Test prompt",
            duration_seconds=invalid_duration,
        )

        with pytest.raises(ValidationError) as exc_info:
            handler._convert_request(config, request)

        assert "Invalid duration" in str(exc_info.value)
        assert f"{invalid_duration} seconds" in str(exc_info.value)
