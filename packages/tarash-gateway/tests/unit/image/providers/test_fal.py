"""Tests for Fal image generation field mappers and registry."""

from tarash.tarash_gateway.models import ImageGenerationRequest
from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers
from tarash.tarash_gateway.providers.fal import (
    FAL_IMAGE_MODEL_REGISTRY,
    NANO_BANANA_2_FIELD_MAPPERS,
    get_image_field_mappers,
)


# ==================== Nano Banana 2 Registry Tests ====================


def test_get_field_mappers_nano_banana_2_all_variants():
    """Test unified mapper for all Nano Banana 2 variants via prefix matching."""
    variants = [
        "fal-ai/nano-banana-2",
        "fal-ai/nano-banana-2/edit",
    ]
    for variant in variants:
        mappers = get_image_field_mappers(variant)
        assert mappers is NANO_BANANA_2_FIELD_MAPPERS, (
            f"Expected NANO_BANANA_2_FIELD_MAPPERS for {variant}"
        )


def test_nano_banana_2_in_registry():
    """Nano Banana 2 is registered in the image model registry."""
    assert "fal-ai/nano-banana-2" in FAL_IMAGE_MODEL_REGISTRY


# ==================== Nano Banana 2 Conversion Tests ====================


def test_nano_banana_2_text_to_image_conversion():
    """Test Nano Banana 2 text-to-image request conversion (prompt only, no images).

    Tests: prompt, seed, n, aspect_ratio, output_format, resolution, safety_tolerance.
    """
    request = ImageGenerationRequest(
        prompt="A futuristic cityscape at sunset",
        seed=42,
        n=2,
        aspect_ratio="16:9",
        extra_params={
            "output_format": "png",
            "resolution": "2K",
            "safety_tolerance": "4",
        },
    )

    result = apply_field_mappers(NANO_BANANA_2_FIELD_MAPPERS, request)

    assert result["prompt"] == "A futuristic cityscape at sunset"
    assert result["seed"] == 42
    assert result["num_images"] == 2
    assert result["aspect_ratio"] == "16:9"
    assert result["output_format"] == "png"
    assert result["resolution"] == "2K"
    assert result["safety_tolerance"] == "4"
    # No image fields should be present
    assert "image_urls" not in result


def test_nano_banana_2_edit_conversion_with_images():
    """Test Nano Banana 2 edit conversion with reference images.

    Tests: image_urls from image_list, enable_web_search, limit_generations.
    """
    request = ImageGenerationRequest(
        prompt="Add a rainbow over the mountains",
        image_list=[
            {"type": "reference", "image": "https://example.com/img1.jpg"},
            {"type": "reference", "image": "https://example.com/img2.jpg"},
        ],
        extra_params={
            "enable_web_search": True,
            "limit_generations": False,
        },
    )

    result = apply_field_mappers(NANO_BANANA_2_FIELD_MAPPERS, request)

    assert result["prompt"] == "Add a rainbow over the mountains"
    assert result["image_urls"] == [
        "https://example.com/img1.jpg",
        "https://example.com/img2.jpg",
    ]
    assert result["enable_web_search"] is True
    assert result["limit_generations"] is False


def test_nano_banana_2_optional_fields_excluded_when_none():
    """Optional fields are excluded from conversion when not provided."""
    request = ImageGenerationRequest(prompt="Simple prompt")

    result = apply_field_mappers(NANO_BANANA_2_FIELD_MAPPERS, request)

    assert result["prompt"] == "Simple prompt"
    assert "seed" not in result
    assert "num_images" not in result
    assert "aspect_ratio" not in result
    assert "image_urls" not in result
    assert "output_format" not in result
    assert "safety_tolerance" not in result
    assert "resolution" not in result
    assert "enable_web_search" not in result
    assert "limit_generations" not in result
