"""Replicate provider handler."""

import asyncio
import traceback
from typing import TYPE_CHECKING, Literal, cast, overload

from tarash.tarash_gateway.logging import log_debug, log_info
from tarash.tarash_gateway.video.exceptions import (
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.video.models import (
    AnyDict,
    MediaContent,
    ProgressCallback,
    SyncProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.video.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
    duration_field_mapper,
    extra_params_field_mapper,
    get_field_mappers_from_registry,
    passthrough_field_mapper,
    single_image_field_mapper,
)

has_replicate = True
try:
    import replicate
    from replicate import AsyncReplicate, Replicate

    # Import Replicate exception types
    APITimeoutError = replicate.APITimeoutError
    APIConnectionError = replicate.APIConnectionError
    APIStatusError = replicate.APIStatusError
    BadRequestError = replicate.BadRequestError
    UnprocessableEntityError = replicate.UnprocessableEntityError
except (ImportError, Exception):
    has_replicate = False

if TYPE_CHECKING:
    import replicate
    from replicate import AsyncReplicate, Replicate
    from replicate.types import Prediction

# Logger name constant
_LOGGER_NAME = "tarash.tarash_gateway.video.providers.replicate"

# Provider name constant
_PROVIDER_NAME = "replicate"

# ==================== Model Field Mappings ====================


# Kling v2.1 field mappings (kwaivgi/kling-v2.1)
# Input schema: prompt (required), image (optional), duration (5 or 10), aspect_ratio, negative_prompt
KLING_V21_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "image": single_image_field_mapper(required=False, image_type="reference"),
    "duration": duration_field_mapper(
        field_type="int", allowed_values=[5, 10], provider="replicate", model="kling"
    ),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "cfg_scale": extra_params_field_mapper("cfg_scale"),
    "seed": passthrough_field_mapper("seed"),
}


# Minimax (Hailuo) field mappings
# Based on fal's MINIMAX_FIELD_MAPPERS pattern
MINIMAX_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=["6s", "10s"],
        provider="replicate",
        model="minimax",
    ),
    "image_url": single_image_field_mapper(required=False, image_type="reference"),
    "prompt_optimizer": passthrough_field_mapper("enhance_prompt"),
}


# Luma Dream Machine field mappings (luma/dream-machine)
LUMA_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "start_image_url": single_image_field_mapper(
        required=False, image_type="first_frame"
    ),
    "end_image_url": single_image_field_mapper(required=False, image_type="last_frame"),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "loop": extra_params_field_mapper("loop"),
}


# Wan (Alibaba) video models field mappings
WAN_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "image": single_image_field_mapper(required=False, image_type="reference"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "num_frames": FieldMapper(
        source_field="duration_seconds",
        converter=lambda req, val: cast(int, val) * 24
        if val
        else None,  # Approximate fps
        required=False,
    ),
    "seed": passthrough_field_mapper("seed"),
}


# Helper function to create a filtered image list mapper
def _create_reference_images_mapper() -> FieldMapper:
    """Create a FieldMapper for reference_images (filters by type='reference')."""
    from tarash.tarash_gateway.video.utils import convert_to_data_url

    def converter(_request: VideoGenerationRequest, value: object) -> list[str] | None:
        if not value or not isinstance(value, list):
            return None

        # Type narrow the list to list[object]
        value_list = cast(list[object], value)

        # Filter images with type="reference"
        urls: list[str] = []
        for item in value_list:
            # Type narrow to dict
            if not isinstance(item, dict):
                continue

            item_dict = cast(dict[str, object], item)

            # Check if this is a reference image
            if item_dict.get("type") == "reference" and "image" in item_dict:
                media = item_dict["image"]
                if isinstance(media, dict) and "content" in media:
                    urls.append(convert_to_data_url(cast(MediaContent, media)))
                else:
                    # Media is a URL string or convertible to string
                    if isinstance(media, str):
                        urls.append(media)
                    else:
                        # Cast to object for str() conversion
                        urls.append(str(cast(object, media)))

        return urls if urls else None

    return FieldMapper(source_field="image_list", converter=converter, required=False)


# Veo 3.1 field mappings (google/veo-3.1)
# Input schema: prompt (required), aspect_ratio, duration (int seconds),
# image, last_frame, reference_images (1-3 images), negative_prompt, resolution, generate_audio, seed
VEO31_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "duration": duration_field_mapper(
        field_type="int", allowed_values=[4, 6, 8], provider="replicate", model="veo3.1"
    ),
    "image": single_image_field_mapper(required=False, image_type="first_frame"),
    "last_frame": single_image_field_mapper(required=False, image_type="last_frame"),
    "reference_images": _create_reference_images_mapper(),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "resolution": passthrough_field_mapper("resolution"),
    "generate_audio": passthrough_field_mapper("generate_audio"),
    "seed": passthrough_field_mapper("seed"),
}


# Generic fallback field mappings for unknown Replicate models
GENERIC_REPLICATE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper(field_type="int"),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "image": single_image_field_mapper(required=False, image_type="reference"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "seed": passthrough_field_mapper("seed"),
}


# ==================== Model Registry ====================


# Registry maps model name prefixes to their field mappers
REPLICATE_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    # Kling models
    "kwaivgi/kling": KLING_V21_FIELD_MAPPERS,
    # Minimax / Hailuo models
    "minimax/": MINIMAX_FIELD_MAPPERS,
    "hailuo/": MINIMAX_FIELD_MAPPERS,
    # Luma Dream Machine
    "luma/": LUMA_FIELD_MAPPERS,
    "luma/dream-machine": LUMA_FIELD_MAPPERS,
    # Wan (Alibaba) models
    "wan-video/": WAN_FIELD_MAPPERS,
    "alibaba/wan": WAN_FIELD_MAPPERS,
    # Google Veo 3 and 3.1 (same API)
    "google/veo-3": VEO31_FIELD_MAPPERS,
}


def get_replicate_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get the field mappers for a given Replicate model.

    Lookup Strategy:
    1. Try exact match first
    2. If not found, try prefix matching - find registry keys that are prefixes
       of the model_name, and use the longest matching prefix
    3. If no match found, return generic field mappers as fallback

    Args:
        model_name: Full model name (e.g., "kwaivgi/kling-v2.1")

    Returns:
        Dict mapping API field names to FieldMapper objects
    """
    # Normalize model name (remove version suffix if present)
    base_model = model_name.split(":")[0]

    return get_field_mappers_from_registry(
        base_model, REPLICATE_MODEL_REGISTRY, GENERIC_REPLICATE_FIELD_MAPPERS
    )


# ==================== Status Parsing ====================


def parse_replicate_status(prediction: Prediction) -> VideoGenerationUpdate:
    """Parse Replicate prediction status into VideoGenerationUpdate.

    Replicate prediction statuses:
    - starting: Prediction is starting up
    - processing: Model is running
    - succeeded: Prediction completed successfully
    - failed: Prediction failed
    - canceled: Prediction was canceled

    Args:
        prediction: Replicate Prediction object

    Returns:
        VideoGenerationUpdate with normalized status
    """
    from tarash.tarash_gateway.video.models import StatusType

    status = prediction.status
    prediction_id = prediction.id

    # Map Replicate status to our status
    status_map: dict[str, StatusType] = {
        "starting": "queued",
        "processing": "processing",
        "succeeded": "completed",
        "failed": "failed",
        "canceled": "failed",
    }

    normalized_status: StatusType = status_map.get(status, "processing")

    # Build update dict with available info
    update_data: AnyDict = {
        "replicate_status": status,
    }

    # Add progress if available (progress is a dynamic attribute, not in type definition)
    progress: object | None = getattr(prediction, "progress", None)
    progress_percent: int | None = None
    if progress is not None:
        if hasattr(progress, "percentage"):
            percentage = getattr(progress, "percentage")
            progress_percent = int(cast(float, percentage))
            update_data["progress"] = percentage
        if hasattr(progress, "current") and hasattr(progress, "total"):
            update_data["current"] = getattr(progress, "current")
            update_data["total"] = getattr(progress, "total")

    # Add logs if available
    logs = prediction.logs
    if logs:
        # Only include last portion to avoid huge updates
        update_data["logs"] = logs[-500:] if len(logs) > 500 else logs

    # Add error if failed
    error = prediction.error

    return VideoGenerationUpdate(
        request_id=prediction_id,
        status=normalized_status,
        progress_percent=progress_percent,
        update=update_data,
        error=error,
    )


# ==================== Provider Handler ====================


class ReplicateProviderHandler:
    """Handler for Replicate provider."""

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""
        if not has_replicate:
            raise ImportError(
                "replicate is required for Replicate provider. "
                + "Install with: pip install tarash-gateway[replicate]"
            )

        self._sync_client_cache: dict[str, Replicate] = {}
        self._async_client_cache: dict[str, AsyncReplicate] = {}

    @overload
    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["async"]
    ) -> AsyncReplicate: ...

    @overload
    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["sync"]
    ) -> Replicate: ...

    def _get_client(
        self,
        config: VideoGenerationConfig,
        client_type: Literal["sync", "async"] = "sync",
    ) -> Replicate | AsyncReplicate:
        """Get or create Replicate client for the given config.

        Args:
            config: Provider configuration
            client_type: Type of client to create ("sync" or "async")

        Returns:
            Replicate or AsyncReplicate instance (v2.0.0 API)
        """
        if not has_replicate:
            raise ImportError(
                "replicate is required for Replicate provider. "
                + "Install with: pip install tarash-gateway[replicate]"
            )

        # Use API key as cache key
        cache_key = config.api_key

        if client_type == "async":
            if cache_key not in self._async_client_cache:
                self._async_client_cache[cache_key] = AsyncReplicate(
                    bearer_token=config.api_key
                )
            return self._async_client_cache[cache_key]
        else:
            if cache_key not in self._sync_client_cache:
                self._sync_client_cache[cache_key] = Replicate(
                    bearer_token=config.api_key
                )
            return self._sync_client_cache[cache_key]

    def _convert_request(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
    ) -> AnyDict:
        """Convert VideoGenerationRequest to Replicate model-specific format.

        Args:
            config: Provider configuration
            request: Generic video generation request

        Returns:
            Model-specific validated request dictionary
        """
        # Get model-specific field mappers
        field_mappers = get_replicate_field_mappers(config.model)

        # Apply field mappers to convert request
        api_payload = apply_field_mappers(field_mappers, request)

        # Merge with extra_params (allows manual overrides)
        api_payload.update(request.extra_params)

        log_info(
            "Mapped request to provider format",
            context={
                "provider": config.provider,
                "model": config.model,
                "converted_request": api_payload,
            },
            logger_name=_LOGGER_NAME,
            redact=True,
        )

        return api_payload

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        _request: VideoGenerationRequest,
        prediction_id: str,
        prediction_output: object,
    ) -> VideoGenerationResponse:
        """Convert Replicate prediction output to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request: Original video generation request
            prediction_id: Replicate prediction ID
            prediction_output: Raw Replicate prediction output

        Returns:
            Normalized VideoGenerationResponse
        """
        # Extract video URL from prediction output
        # Replicate outputs can be:
        # - A single URL string
        # - A list of URLs
        # - A dict with 'video' or 'output' key
        # - A FileOutput object with URL
        video_url: str | None = None

        if prediction_output is None:
            raise GenerationFailedError(
                "Prediction completed but no output was returned",
                provider=config.provider,
                raw_response={},
            )

        if isinstance(prediction_output, str):
            video_url = prediction_output
        elif isinstance(prediction_output, list):
            # Type narrow to list
            output_list = cast(list[object], prediction_output)
            if len(output_list) > 0:
                # Take the first item if it's a list
                first_item = output_list[0]
                if isinstance(first_item, str):
                    video_url = first_item
                elif hasattr(first_item, "url"):
                    video_url = cast(str, getattr(first_item, "url"))
                elif hasattr(first_item, "read"):
                    # FileOutput object - get URL
                    video_url = str(first_item)
        elif isinstance(prediction_output, dict):
            output_dict = cast(AnyDict, prediction_output)
            video_url = cast(
                str | None, output_dict.get("video") or output_dict.get("output")
            )
        elif hasattr(prediction_output, "url"):
            video_url = cast(str, getattr(prediction_output, "url"))
        elif hasattr(prediction_output, "read"):
            # FileOutput object
            video_url = str(prediction_output)

        if not video_url:
            # Convert output to string for error message - handle complex types
            if isinstance(prediction_output, str):
                output_str = prediction_output
            elif isinstance(prediction_output, dict):
                output_str = repr(cast(dict[str, object], prediction_output))
            elif isinstance(prediction_output, list):
                output_str = repr(cast(list[object], prediction_output))
            elif prediction_output is None:
                output_str = "None"
            else:
                output_str = str(prediction_output)

            raise GenerationFailedError(
                f"Could not extract video URL from Replicate output: {output_str}",
                provider=config.provider,
                raw_response={"output": output_str},
            )

        # Convert output to string for raw_response - handle complex types
        if isinstance(prediction_output, str):
            output_str = prediction_output
        elif isinstance(prediction_output, dict):
            output_str = repr(cast(dict[str, object], prediction_output))
        elif isinstance(prediction_output, list):
            output_str = repr(cast(list[object], prediction_output))
        elif prediction_output is None:
            output_str = "None"
        else:
            output_str = str(prediction_output)

        return VideoGenerationResponse(
            request_id=prediction_id,
            video=video_url,
            status="completed",
            raw_response={"output": output_str},
            provider_metadata={"replicate_prediction_id": prediction_id},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        _request: VideoGenerationRequest,
        prediction_id: str,
        ex: Exception,
    ) -> TarashException:
        """Handle errors during video generation.

        Args:
            config: Provider configuration
            request: Original video generation request
            prediction_id: Replicate prediction ID (if available)
            ex: The exception that occurred

        Returns:
            TarashException with details
        """
        if isinstance(ex, TarashException):
            return ex

        # API timeout errors
        if isinstance(ex, APITimeoutError):
            return TimeoutError(
                f"Request timed out: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=prediction_id,
                raw_response={"error": str(ex), "prediction_id": prediction_id},
                timeout_seconds=config.timeout,
            )

        # API connection errors
        if isinstance(ex, APIConnectionError):
            return HTTPConnectionError(
                f"Connection error: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=prediction_id,
                raw_response={"error": str(ex), "prediction_id": prediction_id},
            )

        # API status errors (4xx, 5xx)
        if isinstance(ex, APIStatusError):
            raw_response = {
                "status_code": ex.status_code,
                "message": ex.message,
                "body": ex.body,
                "prediction_id": prediction_id,
            }

            # Validation errors (400, 422)
            if isinstance(ex, (BadRequestError, UnprocessableEntityError)):
                return ValidationError(
                    ex.message,
                    provider=config.provider,
                    model=config.model,
                    request_id=prediction_id,
                    raw_response=raw_response,
                )

            # All other HTTP errors (401, 403, 429, 500, etc.)
            return HTTPError(
                ex.message,
                provider=config.provider,
                model=config.model,
                request_id=prediction_id,
                raw_response=raw_response,
                status_code=ex.status_code,
            )

        # Unknown errors
        return GenerationFailedError(
            f"Replicate API error: {str(ex)}",
            provider=config.provider,
            model=config.model,
            request_id=prediction_id,
            raw_response={
                "error": str(ex),
                "error_type": type(ex).__name__,
                "prediction_id": prediction_id,
                "traceback": traceback.format_exc(),
            },
        )

    @handle_video_generation_errors
    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate video asynchronously via Replicate with async progress callback.

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional async callback for progress updates

        Returns:
            Final VideoGenerationResponse when complete
        """
        # Build Replicate input
        replicate_input = self._convert_request(config, request)

        log_debug(
            "Starting API call",
            context={
                "provider": config.provider,
                "model": config.model,
            },
            logger_name=_LOGGER_NAME,
        )

        prediction_id = ""

        try:
            # Get async client (v2.0.0 API)
            async_client = self._get_client(config, "async")

            # Use run for simple case without progress
            if on_progress is None:
                output = cast(
                    object,
                    await async_client.run(
                        config.model,
                        input=replicate_input,
                    ),
                )
                # Generate a pseudo prediction ID since run doesn't return one
                prediction_id = f"replicate-{id(output)}"
                return self._convert_response(config, request, prediction_id, output)

            # For progress tracking, we need to use predictions.create and poll
            prediction = await async_client.predictions.create(
                version=config.model,
                input=replicate_input,
            )
            prediction_id = prediction.id

            log_debug(
                "Request submitted",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": prediction_id,
                },
                logger_name=_LOGGER_NAME,
            )

            # Poll for status updates
            poll_interval = config.poll_interval
            max_attempts = config.max_poll_attempts

            for _ in range(max_attempts):
                # Reload prediction status (v2.0.0 API)
                prediction = await async_client.predictions.get(
                    prediction_id=prediction.id
                )

                # Send progress update
                update = parse_replicate_status(prediction)
                result = on_progress(update)
                if asyncio.iscoroutine(result):
                    await result

                # Log progress
                log_info(
                    "Progress status update",
                    context={
                        "provider": config.provider,
                        "model": config.model,
                        "request_id": prediction_id,
                        "status": prediction.status,
                    },
                    logger_name=_LOGGER_NAME,
                )

                # Check if completed
                if prediction.status == "succeeded":
                    log_debug(
                        "Request complete",
                        context={
                            "provider": config.provider,
                            "model": config.model,
                            "request_id": prediction_id,
                            "response": prediction.output,
                        },
                        logger_name=_LOGGER_NAME,
                        redact=True,
                    )

                    response = self._convert_response(
                        config, request, prediction_id, prediction.output
                    )

                    log_info(
                        "Final generated response",
                        context={
                            "provider": config.provider,
                            "model": config.model,
                            "request_id": prediction_id,
                            "response": response,
                        },
                        logger_name=_LOGGER_NAME,
                        redact=True,
                    )

                    return response
                elif prediction.status in ("failed", "canceled"):
                    error_msg = prediction.error or f"Prediction {prediction.status}"
                    raise GenerationFailedError(
                        error_msg,
                        provider=config.provider,
                        raw_response={
                            "status": prediction.status,
                            "error": prediction.error,
                            "logs": prediction.logs,
                        },
                        request_id=prediction_id,
                        model=config.model,
                    )

                # Wait before next poll
                await asyncio.sleep(poll_interval)

            # Max attempts reached
            timeout_seconds = max_attempts * poll_interval
            raise TimeoutError(
                f"Prediction timed out after {max_attempts} attempts ({timeout_seconds}s)",
                provider=config.provider,
                raw_response={
                    "prediction_id": prediction_id,
                    "poll_attempts": max_attempts,
                },
                request_id=prediction_id,
                model=config.model,
                timeout_seconds=timeout_seconds,
            )

        except Exception as ex:
            raise self._handle_error(config, request, prediction_id, ex)

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: SyncProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate video synchronously (blocking).

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional callback for progress updates

        Returns:
            Final VideoGenerationResponse
        """
        client = self._get_client(config, "sync")

        # Build Replicate input
        replicate_input = self._convert_request(config, request)

        log_debug(
            "Starting API call",
            context={
                "provider": config.provider,
                "model": config.model,
            },
            logger_name=_LOGGER_NAME,
        )

        prediction_id = ""

        try:
            # For simple case without progress, use run() (v2.0.0 API)
            if on_progress is None:
                output = cast(
                    object,
                    client.run(
                        config.model,
                        input=replicate_input,
                    ),
                )
                prediction_id = f"replicate-{id(output)}"
                return self._convert_response(config, request, prediction_id, output)

            # For progress tracking, use predictions.create and poll (v2.0.0 API)
            prediction = client.predictions.create(
                version=config.model,
                input=replicate_input,
            )
            prediction_id = prediction.id

            log_debug(
                "Request submitted",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": prediction_id,
                },
                logger_name=_LOGGER_NAME,
            )

            # Poll for status updates
            poll_interval = config.poll_interval
            max_attempts = config.max_poll_attempts

            import time

            for _ in range(max_attempts):
                # Reload prediction status (v2.0.0 API)
                prediction = client.predictions.get(prediction_id=prediction.id)

                update = parse_replicate_status(prediction)
                on_progress(update)

                # Log progress
                log_info(
                    "Progress status update",
                    context={
                        "provider": config.provider,
                        "model": config.model,
                        "request_id": prediction_id,
                        "status": prediction.status,
                    },
                    logger_name=_LOGGER_NAME,
                )

                # Check if completed
                if prediction.status == "succeeded":
                    log_debug(
                        "Request complete",
                        context={
                            "provider": config.provider,
                            "model": config.model,
                            "request_id": prediction_id,
                            "response": prediction.output,
                        },
                        logger_name=_LOGGER_NAME,
                        redact=True,
                    )

                    response = self._convert_response(
                        config, request, prediction_id, prediction.output
                    )

                    log_info(
                        "Final generated response",
                        context={
                            "provider": config.provider,
                            "model": config.model,
                            "request_id": prediction_id,
                            "response": response,
                        },
                        logger_name=_LOGGER_NAME,
                        redact=True,
                    )

                    return response
                elif prediction.status in ("failed", "canceled"):
                    error_msg = prediction.error or f"Prediction {prediction.status}"
                    raise GenerationFailedError(
                        error_msg,
                        provider=config.provider,
                        raw_response={
                            "status": prediction.status,
                            "error": prediction.error,
                            "logs": prediction.logs,
                        },
                        request_id=prediction_id,
                        model=config.model,
                    )

                # Wait before next poll
                time.sleep(poll_interval)

            # Max attempts reached
            timeout_seconds = max_attempts * poll_interval
            raise TimeoutError(
                f"Prediction timed out after {max_attempts} attempts ({timeout_seconds}s)",
                provider=config.provider,
                raw_response={
                    "prediction_id": prediction_id,
                    "poll_attempts": max_attempts,
                },
                request_id=prediction_id,
                model=config.model,
                timeout_seconds=timeout_seconds,
            )

        except Exception as ex:
            raise self._handle_error(config, request, prediction_id, ex)
