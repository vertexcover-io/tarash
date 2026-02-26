"""Stability AI provider handler for direct REST API.

Supports Stability AI image generation models via direct REST API using httpx.
Does not support video generation.

Supported models:
- SD 3.5 Large (sd3.5-large, sd3.5-large-turbo, sd3.5-medium)
- Stable Image Ultra (stable-image-ultra)
- Stable Image Core (stable-image-core)
"""

import base64
from typing import Literal

import httpx

from tarash.tarash_gateway.exceptions import (
    HTTPError,
    TarashException,
    ValidationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.logging import ProviderLogger
from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageProgressCallback,
    SyncImageProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
    extra_params_field_mapper,
    get_field_mappers_from_registry,
    passthrough_field_mapper,
)

# ==================== Constants ====================

# Logger name constant
_LOGGER_NAME = "tarash.tarash_gateway.providers.stability"

# Provider name constant
_PROVIDER_NAME = "stability"

# Base URL for Stability AI API
STABILITY_API_BASE = "https://api.stability.ai"

# ==================== Field Mappers ====================


def _aspect_ratio_field_mapper(
    allowed_values: list[str],
) -> FieldMapper:
    """Create FieldMapper for aspect_ratio with validation.

    Args:
        allowed_values: List of allowed aspect ratio values

    Returns:
        FieldMapper for aspect_ratio field
    """

    def converter(_request: ImageGenerationRequest, value: object) -> str | None:
        if value is None:
            return None

        aspect_ratio = str(value)
        if aspect_ratio not in allowed_values:
            allowed_str = ", ".join(allowed_values)
            raise ValueError(
                f"Invalid aspect_ratio '{aspect_ratio}'. Allowed: {allowed_str} (stability)"
            )
        return aspect_ratio

    return FieldMapper(source_field="aspect_ratio", converter=converter)


def _seed_field_mapper(
    min_value: int = 0,
    max_value: int = 4294967294,
) -> FieldMapper:
    """Create FieldMapper for seed with validation.

    Args:
        min_value: Minimum allowed seed value
        max_value: Maximum allowed seed value

    Returns:
        FieldMapper for seed field
    """

    def converter(_request: ImageGenerationRequest, value: object) -> int | None:
        if value is None:
            return None

        seed = int(value)
        if seed < min_value or seed > max_value:
            raise ValueError(
                f"seed must be between {min_value} and {max_value}, got {seed}"
            )
        return seed

    return FieldMapper(source_field="seed", converter=converter)


def _cfg_scale_field_mapper(
    min_value: float = 1.0,
    max_value: float = 35.0,
) -> FieldMapper:
    """Create FieldMapper for cfg_scale with validation.

    Args:
        min_value: Minimum allowed cfg_scale value
        max_value: Maximum allowed cfg_scale value

    Returns:
        FieldMapper for cfg_scale from extra_params
    """

    def converter(_request: ImageGenerationRequest, value: object) -> float | None:
        if not isinstance(value, dict):
            return None

        extra_params = value
        if "cfg_scale" not in extra_params:
            return None

        cfg_scale = float(extra_params["cfg_scale"])
        if cfg_scale < min_value or cfg_scale > max_value:
            raise ValueError(
                f"cfg_scale must be between {min_value} and {max_value}, got {cfg_scale}"
            )
        return cfg_scale

    return FieldMapper(source_field="extra_params", converter=converter)


def _steps_field_mapper(
    min_value: int = 1,
    max_value: int = 50,
) -> FieldMapper:
    """Create FieldMapper for steps with validation.

    Args:
        min_value: Minimum allowed steps value
        max_value: Maximum allowed steps value

    Returns:
        FieldMapper for steps from extra_params
    """

    def converter(_request: ImageGenerationRequest, value: object) -> int | None:
        if not isinstance(value, dict):
            return None

        extra_params = value
        if "steps" not in extra_params:
            return None

        steps = int(extra_params["steps"])
        if steps < min_value or steps > max_value:
            raise ValueError(
                f"steps must be between {min_value} and {max_value}, got {steps}"
            )
        return steps

    return FieldMapper(source_field="extra_params", converter=converter)


# SD 3.5 Large field mappers (also used for sd3.5-large-turbo, sd3.5-medium)
SD35_LARGE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "aspect_ratio": _aspect_ratio_field_mapper(
        allowed_values=[
            "1:1",
            "16:9",
            "21:9",
            "2:3",
            "3:2",
            "4:5",
            "5:4",
            "9:16",
            "9:21",
        ]
    ),
    "seed": _seed_field_mapper(min_value=0, max_value=4294967294),
    "output_format": extra_params_field_mapper("output_format"),
    "cfg_scale": _cfg_scale_field_mapper(min_value=1.0, max_value=35.0),
    "steps": _steps_field_mapper(min_value=1, max_value=50),
}

# Stable Image Ultra field mappers (also used for stable-image-core)
# Fewer parameters than SD 3.5 Large - automatic quality optimization
STABLE_IMAGE_ULTRA_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "aspect_ratio": _aspect_ratio_field_mapper(
        allowed_values=[
            "1:1",
            "16:9",
            "21:9",
            "2:3",
            "3:2",
            "4:5",
            "5:4",
            "9:16",
            "9:21",
        ]
    ),
    "seed": _seed_field_mapper(min_value=0, max_value=4294967294),
    "output_format": extra_params_field_mapper("output_format"),
}

# Image model registry for Stability AI
STABILITY_IMAGE_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    "sd3.5-large": SD35_LARGE_FIELD_MAPPERS,
    "sd3.5-medium": SD35_LARGE_FIELD_MAPPERS,
    "stable-image-ultra": STABLE_IMAGE_ULTRA_FIELD_MAPPERS,
    "stable-image": STABLE_IMAGE_ULTRA_FIELD_MAPPERS,  # Prefix match for stable-image-*
}


def get_stability_image_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get field mappers for a Stability AI image model.

    Uses registry lookup with prefix matching.

    Args:
        model_name: Model name (e.g., "sd3.5-large", "stable-image-ultra")

    Returns:
        Dict mapping API field names to FieldMapper objects
    """
    return get_field_mappers_from_registry(
        model_name,
        STABILITY_IMAGE_MODEL_REGISTRY,
        SD35_LARGE_FIELD_MAPPERS,  # Fallback to SD 3.5 Large
    )


# ==================== Provider Handler ====================


class StabilityProviderHandler:
    """Handler for Stability AI image generation via REST API.

    Uses httpx for direct REST API calls. Does not support video generation.
    """

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""

    def _get_client(
        self,
        config: ImageGenerationConfig,
        client_type: Literal["sync", "async"],
    ) -> httpx.Client | httpx.AsyncClient:
        """Create httpx client for Stability API.

        Args:
            config: Image generation configuration
            client_type: "sync" or "async"

        Returns:
            httpx.Client or httpx.AsyncClient configured with auth headers
        """
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Accept": "image/*",
        }

        if client_type == "async":
            return httpx.AsyncClient(
                base_url=STABILITY_API_BASE,
                headers=headers,
                timeout=config.timeout,
            )
        else:
            return httpx.Client(
                base_url=STABILITY_API_BASE,
                headers=headers,
                timeout=config.timeout,
            )

    def _convert_image_request(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
    ) -> dict[str, object]:
        """Convert ImageGenerationRequest to Stability API format.

        Args:
            config: Image generation configuration
            request: Image generation request

        Returns:
            Dict with converted parameters for Stability API
        """
        field_mappers = get_stability_image_field_mappers(config.model)
        return apply_field_mappers(field_mappers, request)

    def _convert_image_response(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        request_id: str,
        image_bytes: bytes,
        content_type: str,
    ) -> ImageGenerationResponse:
        """Convert Stability API binary response to ImageGenerationResponse.

        Args:
            config: Image generation configuration
            request: Original image generation request
            request_id: Request ID
            image_bytes: Binary image data from API
            content_type: Content type from response headers

        Returns:
            ImageGenerationResponse with data URL
        """
        # Convert binary data to data URL
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{content_type};base64,{base64_image}"

        return ImageGenerationResponse(
            request_id=request_id,
            images=[data_url],
            content_type=content_type,
            status="completed",
            is_mock=False,
            raw_response={"content_length": len(image_bytes)},
            provider_metadata={
                "model": config.model,
                "provider": config.provider,
            },
        )

    def _handle_error(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Handle errors during image generation.

        Args:
            config: Image generation configuration
            request: Image generation request
            request_id: Request ID
            ex: Exception that occurred

        Returns:
            TarashException with error details
        """
        if isinstance(ex, TarashException):
            return ex

        # httpx HTTP errors
        if isinstance(ex, httpx.HTTPStatusError):
            status_code = ex.response.status_code
            raw_response: dict[str, object] = {
                "status_code": status_code,
                "response": ex.response.text,
            }

            # Validation errors (400, 422)
            if status_code in (400, 422):
                return ValidationError(
                    f"Invalid request: {ex.response.text}",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                )

            # All other HTTP errors
            return HTTPError(
                f"HTTP {status_code}: {ex.response.text}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=raw_response,
                status_code=status_code,
            )

        # Unknown errors - wrap in TarashException
        return TarashException(
            f"Error during image generation: {str(ex)}",
            provider=config.provider,
            model=config.model,
            request_id=request_id,
            raw_response={"error": str(ex), "error_type": type(ex).__name__},
        )

    def _get_endpoint_for_model(self, model: str) -> str:
        """Get API endpoint path for model.

        Args:
            model: Model name

        Returns:
            Endpoint path for the model
        """
        if model.startswith("sd3"):
            return "/v2beta/stable-image/generate/sd3"
        elif model.startswith("stable-image-ultra"):
            return "/v2beta/stable-image/generate/ultra"
        elif model.startswith("stable-image-core"):
            return "/v2beta/stable-image/generate/core"
        else:
            # Default to SD3 endpoint
            return "/v2beta/stable-image/generate/sd3"

    @handle_video_generation_errors
    async def generate_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate image asynchronously with Stability AI.

        Args:
            config: Image generation configuration
            request: Image generation request
            on_progress: Optional callback for progress updates (not used - Stability API is synchronous)

        Returns:
            ImageGenerationResponse with generated image as data URL
        """
        client = self._get_client(config, "async")
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)

        # Convert request to Stability API format
        stability_params = self._convert_image_request(config, request)

        logger.info(
            "Mapped request to provider format",
            {"converted_request": stability_params},
        )

        request_id = f"stability-{id(request)}"

        try:
            # Get endpoint for model
            endpoint = self._get_endpoint_for_model(config.model)

            # Add model to params if using SD3 endpoint
            if "sd3" in endpoint:
                if config.model.startswith("sd3.5-large-turbo"):
                    stability_params["model"] = "sd3.5-large-turbo"
                elif config.model.startswith("sd3.5-medium"):
                    stability_params["model"] = "sd3.5-medium"
                else:
                    stability_params["model"] = "sd3.5-large"

            logger.info("Submitting request to Stability API", {"endpoint": endpoint})

            # Submit request using multipart/form-data
            response = await client.post(
                endpoint,
                data=stability_params,
            )

            # Raise for HTTP errors
            response.raise_for_status()

            # Extract content type
            content_type = response.headers.get("content-type", "image/png")

            logger.debug(
                "Image request complete",
                {
                    "content_type": content_type,
                    "content_length": len(response.content),
                },
            )

            # Convert response
            result = self._convert_image_response(
                config,
                request,
                request_id,
                response.content,
                content_type,
            )

            logger.info("Final generated response", {"response": result}, redact=True)

            return result

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)

    @handle_video_generation_errors
    def generate_image(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: SyncImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate image synchronously with Stability AI.

        Args:
            config: Image generation configuration
            request: Image generation request
            on_progress: Optional callback for progress updates (not used - Stability API is synchronous)

        Returns:
            ImageGenerationResponse with generated image as data URL
        """
        client = self._get_client(config, "sync")
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)

        # Convert request to Stability API format
        stability_params = self._convert_image_request(config, request)

        logger.info(
            "Mapped request to provider format",
            {"converted_request": stability_params},
        )

        request_id = f"stability-{id(request)}"

        try:
            # Get endpoint for model
            endpoint = self._get_endpoint_for_model(config.model)

            # Add model to params if using SD3 endpoint
            if "sd3" in endpoint:
                if config.model.startswith("sd3.5-large-turbo"):
                    stability_params["model"] = "sd3.5-large-turbo"
                elif config.model.startswith("sd3.5-medium"):
                    stability_params["model"] = "sd3.5-medium"
                else:
                    stability_params["model"] = "sd3.5-large"

            logger.info("Submitting request to Stability API", {"endpoint": endpoint})

            # Submit request using multipart/form-data
            response = client.post(
                endpoint,
                data=stability_params,
            )

            # Raise for HTTP errors
            response.raise_for_status()

            # Extract content type
            content_type = response.headers.get("content-type", "image/png")

            logger.debug(
                "Image request complete",
                {
                    "content_type": content_type,
                    "content_length": len(response.content),
                },
            )

            # Convert response
            result = self._convert_image_response(
                config,
                request,
                request_id,
                response.content,
                content_type,
            )

            logger.info("Final generated response", {"response": result}, redact=True)

            return result

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)

    # ==================== Video Generation Not Supported ====================

    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress=None,
    ) -> VideoGenerationResponse:
        """Video generation not supported by Stability AI."""
        raise NotImplementedError("Stability AI does not support video generation")

    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress=None,
    ) -> VideoGenerationResponse:
        """Video generation not supported by Stability AI."""
        raise NotImplementedError("Stability AI does not support video generation")
