"""Unit tests for Stability AI provider handler."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers
from tarash.tarash_gateway.providers.stability import (
    SD35_LARGE_FIELD_MAPPERS,
    STABLE_IMAGE_ULTRA_FIELD_MAPPERS,
    STABILITY_IMAGE_MODEL_REGISTRY,
    StabilityProviderHandler,
    get_stability_image_field_mappers,
)


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    """Create handler instance and clear caches."""
    h = StabilityProviderHandler()
    return h


@pytest.fixture
def base_config():
    """Create basic config for testing."""
    return ImageGenerationConfig(
        model="sd3.5-large",
        provider="stability",
        api_key="test-api-key",
        timeout=120,
    )


@pytest.fixture
def base_request():
    """Create basic request for testing."""
    return ImageGenerationRequest(
        prompt="A serene mountain landscape",
    )


# ==================== Client Creation Tests ====================


def test_get_client_creates_sync_client(handler, base_config):
    """Test that _get_client creates sync httpx.Client with correct auth."""
    with patch("httpx.Client") as mock_client_class:
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        client = handler._get_client(base_config, "sync")

        assert client is mock_instance
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["base_url"] == "https://api.stability.ai"
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"
        assert call_kwargs["headers"]["Accept"] == "image/*"


def test_get_client_creates_async_client():
    """Test that _get_client creates async httpx.AsyncClient."""
    handler = StabilityProviderHandler()
    config = ImageGenerationConfig(
        model="sd3.5-large",
        provider="stability",
        api_key="test-key",
        timeout=120,
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_instance = AsyncMock()
        mock_client_class.return_value = mock_instance

        client = handler._get_client(config, "async")

        assert client is mock_instance
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["base_url"] == "https://api.stability.ai"
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"


# ==================== Field Mapper Tests (SD 3.5 Large) ====================


def test_sd35_large_field_mappers_prompt():
    """Test SD 3.5 Large prompt mapping."""
    request = ImageGenerationRequest(prompt="A beautiful sunset")

    result = apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)

    assert result["prompt"] == "A beautiful sunset"


def test_sd35_large_field_mappers_negative_prompt():
    """Test SD 3.5 Large negative_prompt mapping."""
    request = ImageGenerationRequest(
        prompt="A sunset",
        negative_prompt="blurry, low quality",
    )

    result = apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)

    assert result["negative_prompt"] == "blurry, low quality"


def test_sd35_large_field_mappers_aspect_ratio():
    """Test SD 3.5 Large aspect_ratio mapping with validation."""
    request = ImageGenerationRequest(
        prompt="A sunset",
        aspect_ratio="16:9",
    )

    result = apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)

    assert result["aspect_ratio"] == "16:9"


def test_sd35_large_field_mappers_invalid_aspect_ratio():
    """Test SD 3.5 Large rejects invalid aspect ratios."""
    # Use extra_params to bypass Pydantic validation
    request = ImageGenerationRequest(
        prompt="A sunset",
        extra_params={
            "aspect_ratio": "32:9"
        },  # Invalid, will be caught by field mapper
    )

    # Manually set aspect_ratio attribute to test field mapper validation
    request.aspect_ratio = "32:9"  # type: ignore

    with pytest.raises(ValueError, match="Invalid aspect_ratio"):
        apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)


def test_sd35_large_field_mappers_seed():
    """Test SD 3.5 Large seed mapping with validation."""
    request = ImageGenerationRequest(
        prompt="A sunset",
        seed=42,
    )

    result = apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)

    assert result["seed"] == 42


def test_sd35_large_field_mappers_seed_out_of_range():
    """Test SD 3.5 Large rejects seed out of range."""
    request = ImageGenerationRequest(
        prompt="A sunset",
        seed=5000000000,  # > 4294967294
    )

    with pytest.raises(ValueError, match="seed must be between"):
        apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)


def test_sd35_large_field_mappers_output_format():
    """Test SD 3.5 Large output_format from extra_params."""
    request = ImageGenerationRequest(
        prompt="A sunset",
        extra_params={"output_format": "jpeg"},
    )

    result = apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)

    assert result["output_format"] == "jpeg"


def test_sd35_large_field_mappers_cfg_scale():
    """Test SD 3.5 Large cfg_scale from extra_params."""
    request = ImageGenerationRequest(
        prompt="A sunset",
        extra_params={"cfg_scale": 7.5},
    )

    result = apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)

    assert result["cfg_scale"] == 7.5


def test_sd35_large_field_mappers_cfg_scale_validation():
    """Test SD 3.5 Large cfg_scale validation."""
    request = ImageGenerationRequest(
        prompt="A sunset",
        extra_params={"cfg_scale": 50.0},  # > 35.0
    )

    with pytest.raises(ValueError, match="cfg_scale must be between"):
        apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)


def test_sd35_large_field_mappers_steps():
    """Test SD 3.5 Large steps from extra_params."""
    request = ImageGenerationRequest(
        prompt="A sunset",
        extra_params={"steps": 30},
    )

    result = apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)

    assert result["steps"] == 30


def test_sd35_large_field_mappers_steps_validation():
    """Test SD 3.5 Large steps validation."""
    request = ImageGenerationRequest(
        prompt="A sunset",
        extra_params={"steps": 100},  # > 50
    )

    with pytest.raises(ValueError, match="steps must be between"):
        apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)


# ==================== Field Mapper Tests (Stable Image Ultra) ====================


def test_stable_image_ultra_field_mappers_prompt():
    """Test Stable Image Ultra prompt mapping."""
    request = ImageGenerationRequest(prompt="A cat on a windowsill")

    result = apply_field_mappers(STABLE_IMAGE_ULTRA_FIELD_MAPPERS, request)

    assert result["prompt"] == "A cat on a windowsill"


def test_stable_image_ultra_field_mappers_aspect_ratio():
    """Test Stable Image Ultra aspect_ratio mapping."""
    request = ImageGenerationRequest(
        prompt="A cat",
        aspect_ratio="1:1",
    )

    result = apply_field_mappers(STABLE_IMAGE_ULTRA_FIELD_MAPPERS, request)

    assert result["aspect_ratio"] == "1:1"


def test_stable_image_ultra_field_mappers_seed():
    """Test Stable Image Ultra seed mapping."""
    request = ImageGenerationRequest(
        prompt="A cat",
        seed=12345,
    )

    result = apply_field_mappers(STABLE_IMAGE_ULTRA_FIELD_MAPPERS, request)

    assert result["seed"] == 12345


def test_stable_image_ultra_field_mappers_output_format():
    """Test Stable Image Ultra output_format mapping."""
    request = ImageGenerationRequest(
        prompt="A cat",
        extra_params={"output_format": "webp"},
    )

    result = apply_field_mappers(STABLE_IMAGE_ULTRA_FIELD_MAPPERS, request)

    assert result["output_format"] == "webp"


# ==================== Model Registry Tests ====================


def test_get_stability_image_field_mappers_sd35_large():
    """Test registry lookup for SD 3.5 Large."""
    mappers = get_stability_image_field_mappers("sd3.5-large")
    assert mappers is SD35_LARGE_FIELD_MAPPERS


def test_get_stability_image_field_mappers_sd35_large_turbo():
    """Test registry lookup for SD 3.5 Large Turbo (prefix match)."""
    mappers = get_stability_image_field_mappers("sd3.5-large-turbo")
    # Should match "sd3.5-large" prefix
    assert mappers is SD35_LARGE_FIELD_MAPPERS


def test_get_stability_image_field_mappers_stable_image_ultra():
    """Test registry lookup for Stable Image Ultra."""
    mappers = get_stability_image_field_mappers("stable-image-ultra")
    assert mappers is STABLE_IMAGE_ULTRA_FIELD_MAPPERS


def test_get_stability_image_field_mappers_stable_image_core():
    """Test registry lookup for Stable Image Core (prefix match)."""
    mappers = get_stability_image_field_mappers("stable-image-core")
    # Should match "stable-image" prefix
    assert mappers is STABLE_IMAGE_ULTRA_FIELD_MAPPERS


def test_stability_registry_contains_expected_models():
    """Test that registry contains expected model keys."""
    expected_keys = {"sd3.5-large", "stable-image-ultra", "stable-image"}
    assert expected_keys.issubset(set(STABILITY_IMAGE_MODEL_REGISTRY.keys()))


# ==================== Request Conversion Tests ====================


def test_convert_image_request_basic(handler, base_config, base_request):
    """Test basic request conversion."""
    result = handler._convert_image_request(base_config, base_request)

    assert result["prompt"] == "A serene mountain landscape"
    assert "negative_prompt" not in result  # Not provided


def test_convert_image_request_with_all_params(handler, base_config):
    """Test request conversion with all parameters."""
    request = ImageGenerationRequest(
        prompt="A mountain",
        negative_prompt="ugly, blurry",
        aspect_ratio="16:9",
        seed=999,
        extra_params={"output_format": "png", "cfg_scale": 8.0, "steps": 40},
    )

    result = handler._convert_image_request(base_config, request)

    assert result["prompt"] == "A mountain"
    assert result["negative_prompt"] == "ugly, blurry"
    assert result["aspect_ratio"] == "16:9"
    assert result["seed"] == 999
    assert result["output_format"] == "png"
    assert result["cfg_scale"] == 8.0
    assert result["steps"] == 40


# ==================== Response Conversion Tests ====================


def test_convert_image_response_basic(handler, base_config, base_request):
    """Test basic response conversion with binary data."""
    # Simulate binary image data
    fake_image_bytes = b"fake-png-data"
    base64_image = base64.b64encode(fake_image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{base64_image}"

    response = handler._convert_image_response(
        base_config,
        base_request,
        "req-123",
        fake_image_bytes,
        "image/png",
    )

    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id == "req-123"
    assert response.status == "completed"
    assert len(response.images) == 1
    assert response.images[0] == data_url
    assert response.content_type == "image/png"


def test_convert_image_response_jpeg(handler, base_config, base_request):
    """Test response conversion with JPEG format."""
    fake_image_bytes = b"fake-jpeg-data"
    base64_image = base64.b64encode(fake_image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{base64_image}"

    response = handler._convert_image_response(
        base_config,
        base_request,
        "req-456",
        fake_image_bytes,
        "image/jpeg",
    )

    assert response.images[0] == data_url
    assert response.content_type == "image/jpeg"


# ==================== Async Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_image_async_success(handler, base_config, base_request):
    """Test successful async image generation."""
    fake_image_bytes = b"fake-image-data"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = fake_image_bytes
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        response = await handler.generate_image_async(base_config, base_request)

        assert isinstance(response, ImageGenerationResponse)
        assert response.status == "completed"
        assert len(response.images) == 1
        assert response.images[0].startswith("data:image/png;base64,")

        # Verify API call
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/v2beta/stable-image/generate/" in call_args[0][0]


# ==================== Sync Generation Tests ====================


def test_generate_image_sync_success(handler, base_config, base_request):
    """Test successful sync image generation."""
    fake_image_bytes = b"fake-image-data"

    with patch("httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_image_bytes
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        response = handler.generate_image(base_config, base_request)

        assert isinstance(response, ImageGenerationResponse)
        assert response.status == "completed"
        assert len(response.images) == 1
        assert response.images[0].startswith("data:image/png;base64,")

        # Verify API call
        mock_client.post.assert_called_once()


# ==================== Video Generation Not Supported ====================


@pytest.mark.asyncio
async def test_generate_video_async_not_supported(handler):
    """Test that video generation raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="does not support video"):
        await handler.generate_video_async(None, None)


def test_generate_video_sync_not_supported(handler):
    """Test that sync video generation raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="does not support video"):
        handler.generate_video(None, None)


# ==================== Error Handling Tests ====================


def test_handle_error_tarash_exception_passes_through(
    handler, base_config, base_request
):
    """Test TarashException passes through unchanged."""
    from tarash.tarash_gateway.exceptions import ValidationError

    ex = ValidationError("test", provider="stability")
    result = handler._handle_error(base_config, base_request, "req-1", ex)
    assert result is ex


def test_handle_error_http_status_400(handler, base_config, base_request):
    """Test HTTPStatusError 400 maps to ValidationError."""
    import httpx

    mock_response = httpx.Response(
        status_code=400,
        text="Bad request: invalid prompt",
        request=httpx.Request("POST", "https://api.stability.ai/v2beta/test"),
    )
    ex = httpx.HTTPStatusError(
        "400 Bad Request", request=mock_response.request, response=mock_response
    )

    result = handler._handle_error(base_config, base_request, "req-400", ex)

    from tarash.tarash_gateway.exceptions import ValidationError

    assert isinstance(result, ValidationError)
    assert "Invalid request" in result.message
    assert result.raw_response["status_code"] == 400


def test_handle_error_http_status_422(handler, base_config, base_request):
    """Test HTTPStatusError 422 maps to ValidationError."""
    import httpx

    mock_response = httpx.Response(
        status_code=422,
        text="Unprocessable entity",
        request=httpx.Request("POST", "https://api.stability.ai/v2beta/test"),
    )
    ex = httpx.HTTPStatusError(
        "422 Unprocessable", request=mock_response.request, response=mock_response
    )

    result = handler._handle_error(base_config, base_request, "req-422", ex)

    from tarash.tarash_gateway.exceptions import ValidationError

    assert isinstance(result, ValidationError)


def test_handle_error_http_status_500(handler, base_config, base_request):
    """Test HTTPStatusError 500 maps to HTTPError."""
    import httpx

    mock_response = httpx.Response(
        status_code=500,
        text="Internal server error",
        request=httpx.Request("POST", "https://api.stability.ai/v2beta/test"),
    )
    ex = httpx.HTTPStatusError(
        "500 Server Error", request=mock_response.request, response=mock_response
    )

    result = handler._handle_error(base_config, base_request, "req-500", ex)

    from tarash.tarash_gateway.exceptions import HTTPError

    assert isinstance(result, HTTPError)
    assert result.status_code == 500


def test_handle_error_unknown_exception(handler, base_config, base_request):
    """Test unknown exception maps to TarashException."""
    from tarash.tarash_gateway.exceptions import TarashException

    ex = RuntimeError("something unexpected")
    result = handler._handle_error(base_config, base_request, "req-unk", ex)

    assert isinstance(result, TarashException)
    assert "Error during image generation" in result.message


# ==================== Endpoint Routing Tests ====================


def test_get_endpoint_sd3_model(handler):
    """Test SD3 model routes to SD3 endpoint."""
    result = handler._get_endpoint_for_model("sd3.5-large")
    assert result == "/v2beta/stable-image/generate/sd3"


def test_get_endpoint_sd3_medium(handler):
    """Test SD3 medium model routes to SD3 endpoint."""
    result = handler._get_endpoint_for_model("sd3.5-medium")
    assert result == "/v2beta/stable-image/generate/sd3"


def test_get_endpoint_ultra(handler):
    """Test stable-image-ultra routes to ultra endpoint."""
    result = handler._get_endpoint_for_model("stable-image-ultra")
    assert result == "/v2beta/stable-image/generate/ultra"


def test_get_endpoint_core(handler):
    """Test stable-image-core routes to core endpoint."""
    result = handler._get_endpoint_for_model("stable-image-core")
    assert result == "/v2beta/stable-image/generate/core"


def test_get_endpoint_unknown_defaults_to_sd3(handler):
    """Test unknown model defaults to SD3 endpoint."""
    result = handler._get_endpoint_for_model("unknown-model")
    assert result == "/v2beta/stable-image/generate/sd3"


# ==================== SD3 Model Variant Selection ====================


@pytest.mark.asyncio
async def test_generate_image_async_sd3_turbo_variant(handler):
    """Test SD3 turbo variant sets correct model name."""
    config = ImageGenerationConfig(
        model="sd3.5-large-turbo",
        provider="stability",
        api_key="test-api-key",
        timeout=120,
    )
    request = ImageGenerationRequest(prompt="A sunset")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-image"
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        await handler.generate_image_async(config, request)

        call_kwargs = mock_client.post.call_args
        # Verify model param was set to sd3.5-large-turbo
        assert call_kwargs[1]["data"]["model"] == "sd3.5-large-turbo"


@pytest.mark.asyncio
async def test_generate_image_async_sd3_medium_variant(handler):
    """Test SD3 medium variant sets correct model name."""
    config = ImageGenerationConfig(
        model="sd3.5-medium",
        provider="stability",
        api_key="test-api-key",
        timeout=120,
    )
    request = ImageGenerationRequest(prompt="A mountain")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-image"
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        await handler.generate_image_async(config, request)

        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["data"]["model"] == "sd3.5-medium"


@pytest.mark.asyncio
async def test_generate_image_async_error_handling(handler, base_config, base_request):
    """Test async image generation error handling."""
    import httpx

    mock_response = httpx.Response(
        status_code=500,
        text="Server error",
        request=httpx.Request("POST", "https://api.stability.ai/v2beta/test"),
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=mock_response.request, response=mock_response
            )
        )
        mock_client_class.return_value = mock_client

        from tarash.tarash_gateway.exceptions import HTTPError

        with pytest.raises(HTTPError):
            await handler.generate_image_async(base_config, base_request)


def test_generate_image_sync_error_handling(handler, base_config, base_request):
    """Test sync image generation error handling."""
    import httpx

    mock_response = httpx.Response(
        status_code=400,
        text="Bad request",
        request=httpx.Request("POST", "https://api.stability.ai/v2beta/test"),
    )

    with patch("httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.post = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "400", request=mock_response.request, response=mock_response
            )
        )
        mock_client_class.return_value = mock_client

        from tarash.tarash_gateway.exceptions import ValidationError

        with pytest.raises(ValidationError):
            handler.generate_image(base_config, base_request)


# ==================== Field Mapper Non-Dict Input ====================


def test_cfg_scale_field_mapper_non_dict_returns_none():
    """Test cfg_scale field mapper returns None for non-dict input."""
    request = ImageGenerationRequest(prompt="A sunset")
    # When extra_params is None, the converter receives None
    result = apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)
    # cfg_scale should not be in result when extra_params has no cfg_scale
    assert "cfg_scale" not in result


def test_steps_field_mapper_non_dict_returns_none():
    """Test steps field mapper returns None for non-dict input."""
    request = ImageGenerationRequest(prompt="A sunset")
    result = apply_field_mappers(SD35_LARGE_FIELD_MAPPERS, request)
    assert "steps" not in result
