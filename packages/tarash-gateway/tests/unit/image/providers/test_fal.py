"""Tests for Fal image generation field mappers and registry."""

from tarash.tarash_gateway.models import ImageGenerationRequest
from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers
from tarash.tarash_gateway.providers.fal import (
    FAL_IMAGE_MODEL_REGISTRY,
    GPT_IMAGE_2_FIELD_MAPPERS,
    NANO_BANANA_2_FIELD_MAPPERS,
    get_image_field_mappers,
)


# ==================== GPT Image 2 Registry Tests ====================


def test_get_field_mappers_gpt_image_2_all_variants():
    """Test unified mapper for all GPT Image 2 variants via prefix matching."""
    variants = [
        "openai/gpt-image-2",
        "openai/gpt-image-2/edit",
    ]
    for variant in variants:
        mappers = get_image_field_mappers(variant)
        assert mappers is GPT_IMAGE_2_FIELD_MAPPERS, (
            f"Expected GPT_IMAGE_2_FIELD_MAPPERS for {variant}"
        )


def test_gpt_image_2_in_registry():
    """GPT Image 2 is registered in the image model registry."""
    assert "openai/gpt-image-2" in FAL_IMAGE_MODEL_REGISTRY


# ==================== GPT Image 2 Conversion Tests ====================


def test_gpt_image_2_edit_conversion_with_images():
    """Test GPT Image 2 edit request conversion with reference images.

    Tests: prompt, image_urls from image_list, n (num_images), quality, output_format.
    """
    request = ImageGenerationRequest(
        prompt="Add a rainbow over the mountains",
        image_list=[
            {"type": "reference", "image": "https://example.com/img1.jpg"},
            {"type": "reference", "image": "https://example.com/img2.jpg"},
        ],
        n=2,
        quality="high",
        extra_params={
            "output_format": "png",
        },
    )

    result = apply_field_mappers(GPT_IMAGE_2_FIELD_MAPPERS, request)

    assert result["prompt"] == "Add a rainbow over the mountains"
    assert result["image_urls"] == [
        "https://example.com/img1.jpg",
        "https://example.com/img2.jpg",
    ]
    assert result["num_images"] == 2
    assert result["quality"] == "high"
    assert result["output_format"] == "png"


def test_gpt_image_2_edit_conversion_with_mask():
    """Test GPT Image 2 edit with mask_url for targeted region editing.

    Tests: mask_url via extra_params, image_size, single reference image.
    """
    request = ImageGenerationRequest(
        prompt="Replace the sky with a stormy night",
        image_list=[
            {"type": "reference", "image": "https://example.com/photo.jpg"},
        ],
        size="square_hd",
        extra_params={
            "mask_url": "https://example.com/mask.png",
            "output_format": "webp",
        },
    )

    result = apply_field_mappers(GPT_IMAGE_2_FIELD_MAPPERS, request)

    assert result["prompt"] == "Replace the sky with a stormy night"
    assert result["image_urls"] == ["https://example.com/photo.jpg"]
    assert result["image_size"] == "square_hd"
    assert result["mask_url"] == "https://example.com/mask.png"
    assert result["output_format"] == "webp"


def test_gpt_image_2_optional_fields_excluded_when_none():
    """Optional fields are excluded from conversion when not provided."""
    request = ImageGenerationRequest(
        prompt="Edit the image",
        image_list=[
            {"type": "reference", "image": "https://example.com/img.jpg"},
        ],
    )

    result = apply_field_mappers(GPT_IMAGE_2_FIELD_MAPPERS, request)

    assert result["prompt"] == "Edit the image"
    assert result["image_urls"] == ["https://example.com/img.jpg"]
    assert "num_images" not in result
    assert "quality" not in result
    assert "image_size" not in result
    assert "output_format" not in result
    assert "mask_url" not in result


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
