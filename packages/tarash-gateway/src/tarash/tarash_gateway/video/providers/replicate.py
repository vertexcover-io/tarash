"""Replicate provider handler."""

import asyncio
import traceback
from typing import Any

from tarash.tarash_gateway.logging import log_debug, log_info
from tarash.tarash_gateway.video.exceptions import (
    GenerationFailedError,
    TarashException,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.video.models import (
    ProgressCallback,
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
    passthrough_field_mapper,
    single_image_field_mapper,
)

try:
    import replicate
    from replicate import Client
    from replicate.prediction import Prediction
except (ImportError, Exception):
    # Handle both ImportError and potential pydantic compatibility issues
    replicate = None  # type: ignore
    Client = None  # type: ignore
    Prediction = None  # type: ignore

# Logger name constant
_LOGGER_NAME = "tarash.tarash_gateway.video.providers.replicate"

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
        converter=lambda req, val: val * 24 if val else None,  # Approximate fps
        required=False,
    ),
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

    # Try exact match first
    if base_model in REPLICATE_MODEL_REGISTRY:
        return REPLICATE_MODEL_REGISTRY[base_model]

    # Try prefix matching - find all registry keys that are prefixes of model_name
    matching_prefix = None
    for registry_key in REPLICATE_MODEL_REGISTRY:
        if base_model.startswith(registry_key):
            if matching_prefix is None or len(registry_key) > len(matching_prefix):
                matching_prefix = registry_key

    if matching_prefix:
        return REPLICATE_MODEL_REGISTRY[matching_prefix]

    # No match found - use generic fallback
    return GENERIC_REPLICATE_FIELD_MAPPERS


# ==================== Status Parsing ====================


def parse_replicate_status(prediction: Any) -> VideoGenerationUpdate:
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
    status = getattr(prediction, "status", "unknown")
    prediction_id = getattr(prediction, "id", "unknown")

    # Map Replicate status to our status
    status_map = {
        "starting": "queued",
        "processing": "processing",
        "succeeded": "completed",
        "failed": "failed",
        "canceled": "failed",
    }

    normalized_status = status_map.get(status, "processing")

    # Build update dict with available info
    update_data: dict[str, Any] = {
        "replicate_status": status,
    }

    # Add progress if available
    progress = getattr(prediction, "progress", None)
    progress_percent = None
    if progress:
        if hasattr(progress, "percentage"):
            progress_percent = int(progress.percentage)
            update_data["progress"] = progress.percentage
        if hasattr(progress, "current") and hasattr(progress, "total"):
            update_data["current"] = progress.current
            update_data["total"] = progress.total

    # Add logs if available
    logs = getattr(prediction, "logs", None)
    if logs:
        # Only include last portion to avoid huge updates
        update_data["logs"] = logs[-500:] if len(logs) > 500 else logs

    # Add error if failed
    error = getattr(prediction, "error", None)

    return VideoGenerationUpdate(
        request_id=prediction_id,
        status=normalized_status,  # type: ignore
        progress_percent=progress_percent,
        update=update_data,
        error=str(error) if error else None,
    )


# ==================== Provider Handler ====================


class ReplicateProviderHandler:
    """Handler for Replicate provider."""

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""
        if replicate is None:
            raise ImportError(
                "replicate is required for Replicate provider. "
                "Install with: pip install tarash-gateway[replicate]"
            )

        self._client_cache: dict[str, Client] = {}

    def _get_client(self, config: VideoGenerationConfig) -> Client:
        """Get or create Replicate client for the given config.

        Args:
            config: Provider configuration

        Returns:
            replicate.Client instance
        """
        # Use API key as cache key
        cache_key = config.api_key

        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = Client(api_token=config.api_key)

        return self._client_cache[cache_key]

    def _convert_request(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
    ) -> dict[str, Any]:
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
        request: VideoGenerationRequest,
        prediction_id: str,
        prediction_output: Any,
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
        video_url = None

        if prediction_output is None:
            raise GenerationFailedError(
                "Prediction completed but no output was returned",
                provider=config.provider,
                raw_response={},
            )

        if isinstance(prediction_output, str):
            video_url = prediction_output
        elif isinstance(prediction_output, list) and len(prediction_output) > 0:
            # Take the first item if it's a list
            first_item = prediction_output[0]
            if isinstance(first_item, str):
                video_url = first_item
            elif hasattr(first_item, "url"):
                video_url = first_item.url
            elif hasattr(first_item, "read"):
                # FileOutput object - get URL
                video_url = str(first_item)
        elif isinstance(prediction_output, dict):
            video_url = prediction_output.get("video") or prediction_output.get(
                "output"
            )
        elif hasattr(prediction_output, "url"):
            video_url = prediction_output.url
        elif hasattr(prediction_output, "read"):
            # FileOutput object
            video_url = str(prediction_output)

        if not video_url:
            raise GenerationFailedError(
                f"Could not extract video URL from Replicate output: {prediction_output}",
                provider=config.provider,
                raw_response={"output": str(prediction_output)},
            )

        return VideoGenerationResponse(
            request_id=prediction_id,
            video=video_url,
            status="completed",
            raw_response={"output": str(prediction_output)},
            provider_metadata={"replicate_prediction_id": prediction_id},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
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

        # Handle Replicate-specific errors
        error_message = str(ex)
        raw_response: dict[str, Any] = {
            "error": error_message,
            "error_type": type(ex).__name__,
            "traceback": traceback.format_exc(),
        }

        if prediction_id:
            raw_response["prediction_id"] = prediction_id

        return GenerationFailedError(
            f"Replicate API error: {error_message}",
            provider=config.provider,
            raw_response=raw_response,
            request_id=prediction_id,
            model=config.model,
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
            # Use async_run for simple case without progress
            if on_progress is None:
                output = await replicate.async_run(
                    config.model,
                    input=replicate_input,
                )
                # Generate a pseudo prediction ID since async_run doesn't return one
                prediction_id = f"replicate-{id(output)}"
                return self._convert_response(config, request, prediction_id, output)

            # For progress tracking, we need to use predictions.create and poll
            # Run the sync prediction creation in a thread pool
            loop = asyncio.get_event_loop()

            def create_prediction():
                client = self._get_client(config)
                return client.predictions.create(
                    model=config.model,
                    input=replicate_input,
                )

            prediction = await loop.run_in_executor(None, create_prediction)
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
                # Reload prediction status
                await loop.run_in_executor(None, prediction.reload)

                # Send progress update
                if on_progress:
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
            raise GenerationFailedError(
                f"Prediction timed out after {max_attempts * poll_interval} seconds",
                provider=config.provider,
                raw_response={"prediction_id": prediction_id},
                request_id=prediction_id,
                model=config.model,
            )

        except Exception as ex:
            raise self._handle_error(config, request, prediction_id, ex)

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate video synchronously (blocking).

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional callback for progress updates

        Returns:
            Final VideoGenerationResponse
        """
        client = self._get_client(config)

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
            # For simple case without progress, use run()
            if on_progress is None:
                output = replicate.run(
                    config.model,
                    input=replicate_input,
                )
                prediction_id = f"replicate-{id(output)}"
                return self._convert_response(config, request, prediction_id, output)

            # For progress tracking, use predictions.create and poll
            prediction = client.predictions.create(
                model=config.model,
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
                # Reload prediction status
                prediction.reload()

                # Send progress update
                if on_progress:
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
            raise GenerationFailedError(
                f"Prediction timed out after {max_attempts * poll_interval} seconds",
                provider=config.provider,
                raw_response={"prediction_id": prediction_id},
                request_id=prediction_id,
                model=config.model,
            )

        except Exception as ex:
            raise self._handle_error(config, request, prediction_id, ex)
