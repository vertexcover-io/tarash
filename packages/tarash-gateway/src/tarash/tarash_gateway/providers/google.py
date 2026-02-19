"""Google AI provider handler using google-genai.

Supports:
- Video generation: Veo 3 models (veo-3.0-generate-preview, etc.)
- Image generation: Imagen 3, Gemini 2.5 Flash Image (Nano Banana)
"""

import asyncio
import base64
import time
import traceback
import uuid
from typing import TYPE_CHECKING, Any, Literal, cast, overload
import os

import httpx
from google.genai.types import GenerateVideosOperation, VideoGenerationReferenceType
from typing_extensions import TypedDict

from tarash.tarash_gateway.logging import ProviderLogger, log_error
from tarash.tarash_gateway.exceptions import (
    ContentModerationError,
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.models import (
    AnyDict,
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageProgressCallback,
    MediaContent,
    MediaType,
    ProgressCallback,
    SyncImageProgressCallback,
    SyncProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
    extra_params_field_mapper,
    get_field_mappers_from_registry,
    passthrough_field_mapper,
)
from tarash.tarash_gateway.utils import validate_model_params

has_genai = True
has_aiohttp = False
try:
    from google.genai.client import AsyncClient, Client
    from google.genai.errors import ClientError
    from google.genai.types import (
        GenerateContentConfig,
        GenerateContentResponse,
        GenerateImagesConfig,
        GenerateImagesResponse,
        GenerateVideosConfig,
        ImageConfig,
    )
    from google.oauth2 import service_account

    # google.genai can use either httpx or aiohttp for async
    try:
        import aiohttp  # type: ignore[import-not-found] # noqa: F401

        has_aiohttp = True
    except ImportError:
        pass
except ImportError:
    has_genai = False

if TYPE_CHECKING:
    from google.genai.client import AsyncClient, Client
    from google.genai.errors import ClientError
    from google.genai.types import (
        GenerateContentConfig,
        GenerateContentResponse,
        GenerateImagesConfig,
        GenerateImagesResponse,
        GenerateVideosConfig,
        ImageConfig,
    )

# Logger name constant
_LOGGER_NAME = "tarash.tarash_gateway.providers.google"


class Veo3VideoParams(TypedDict, total=False):
    """Veo3-specific parameters."""

    person_generation: Literal["allow_all", "dont_allow", "allow_adult"]
    output_gcs_uri: str


def _convert_to_image(image: MediaType) -> AnyDict:
    """Parse last frame image."""
    if isinstance(image, dict) and "content" in image and "content_type" in image:
        return {"image_bytes": image["content"], "mime_type": image["content_type"]}
    elif isinstance(image, str):
        return {"gcs_uri": image}
    return {}


def _convert_to_video(video: MediaType) -> AnyDict:
    """Convert video to google-genai format."""
    if isinstance(video, dict) and "content" in video and "content_type" in video:
        return {"video_bytes": video["content"], "mime_type": video["content_type"]}
    elif isinstance(video, str):
        return {"uri": video}
    return {}


def parse_veo3_operation(operation: GenerateVideosOperation) -> VideoGenerationUpdate:
    """Parse google-genai operation to VideoGenerationUpdate."""
    # Check operation metadata for status
    status_str = "processing"
    metadata = operation.metadata
    if metadata and "state" in metadata:
        state = metadata["state"]
        if state in ["PENDING", "QUEUED"]:
            status_str = "queued"
        elif state in ["RUNNING", "PROCESSING"]:
            status_str = "processing"
    return VideoGenerationUpdate(
        request_id=operation.name or "unknown",
        status=status_str,
        progress_percent=None,
        update={"metadata": metadata},
    )


def _bytes_to_data_uri(
    img_bytes: bytes | bytearray | str, mime_type: str = "image/png"
) -> str:
    """Encode image bytes as a base64 data URI."""
    if isinstance(img_bytes, str):
        img_bytes = img_bytes.encode()
    return f"data:{mime_type};base64,{base64.b64encode(img_bytes).decode()}"


def _is_gemini_image_model(model: str) -> bool:
    """Check if model uses generate_content API (Gemini) vs generate_images API (Imagen).

    Gemini image models (gemini-2.5-flash-image, gemini-3-pro-image) use the
    generate_content API with response_modalities=["IMAGE"].

    Imagen models (imagen-3.0-generate-*) use the dedicated generate_images API.

    Args:
        model: Model name

    Returns:
        True if model is a Gemini image model, False for Imagen models
    """
    return model.startswith("gemini-") and "image" in model.lower()


class GoogleProviderHandler:
    """Handler for Google AI provider using google-genai.

    Supports both video generation (Veo 3) and image generation (Imagen 3, Nano Banana).
    """

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""
        if not has_genai:
            raise ImportError(
                "google-genai is required for Google provider. "
                + "Install with: pip install tarash-gateway[google]"
            )

    @overload
    def _get_client(
        self,
        config: VideoGenerationConfig | ImageGenerationConfig,
        client_type: Literal["async"],
    ) -> AsyncClient: ...

    @overload
    def _get_client(
        self,
        config: VideoGenerationConfig | ImageGenerationConfig,
        client_type: Literal["sync"],
    ) -> Client: ...

    def _get_client(
        self,
        config: VideoGenerationConfig | ImageGenerationConfig,
        client_type: Literal["sync", "async"],
    ) -> AsyncClient | Client:
        """
        Create google-genai client for the given config.

        Supports two modes:
        1. Gemini Developer API: When api_key is provided in config
        2. Vertex AI: When api_key is None (uses provider_config)

        For Vertex AI, set these in provider_config:
            project: GCP project ID (required)
            location: GCP region (default: "us-central1")
            credentials_path: Path to service account JSON file (optional)

        Args:
            config: Provider configuration
            client_type: Type of client to return ("sync" or "async")

        Returns:
            genai.Client instance (sync) or genai.Client.aio (async)
        """
        if not has_genai:
            raise ImportError(
                "google-genai is required for Google provider. "
                + "Install with: pip install tarash-gateway[google]"
            )

        client = self._create_client(config)
        if client_type == "async":
            return client.aio
        else:  # sync
            return client

    def _create_client(
        self, config: VideoGenerationConfig | ImageGenerationConfig
    ) -> Client:
        """Create a new google-genai Client instance."""
        if config.api_key:
            return Client(api_key=config.api_key)

        # Vertex AI mode - get config from provider_config

        pc = config.provider_config or {}
        project = pc.get("project")
        location = pc.get("location")
        credentials_path = pc.get("credentials_path")

        if not project:
            raise ValidationError(
                "Vertex AI requires 'project' in provider_config. "
                "Set provider_config={'project': 'your-project-id'}, "
                "or provide api_key for Gemini Developer API.",
                provider=config.provider,
                model=config.model,
            )

        if not location:
            raise ValidationError(
                "Vertex AI requires 'location' in provider_config. "
                "Set provider_config={'project': '...', 'location': 'us-central1'}, "
                "or provide api_key for Gemini Developer API.",
                provider=config.provider,
                model=config.model,
            )

        credentials = None
        if credentials_path and not os.path.isfile(credentials_path):
            raise ValidationError(
                f"Credentials file not found: {credentials_path}",
                provider=config.provider,
                model=config.model,
            )

        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        return Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
        )

    def _validate_params(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> AnyDict:
        """
        Validate model_params using Veo3-specific schema.

        Args:
            config: Provider configuration
            request: Video generation request

        Returns:
            Validated model parameters dict
        """
        if not request.extra_params:
            return {}

        # Use Veo3VideoParams schema for validation
        return validate_model_params(
            schema=Veo3VideoParams,
            data=request.extra_params,
            provider=config.provider,
            model=config.model,
        )

    def _convert_request(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> AnyDict:
        """
        Convert VideoGenerationRequest to google-genai format.

        Args:
            config: Provider configuration
            request: Video generation request

        Returns:
            Dict with prompt, image, video, and config keys ready for generate_videos(**kwargs)

        Raises:
            ValidationError: If more than 1 reference/first_frame or last_frame image is provided
        """
        # Start with validated model_params
        model_params = self._validate_params(config, request)

        # Build GenerateVideosConfig params
        config_params: AnyDict = {
            **model_params,
        }

        # Separate parameters for generate_videos call
        prompt = request.prompt
        image = None
        video = None

        # Duration: convert to duration_seconds if it's a number or parse from string
        if request.duration_seconds is not None:
            config_params["duration_seconds"] = int(request.duration_seconds)

        # Aspect ratio
        if request.aspect_ratio is not None:
            config_params["aspect_ratio"] = request.aspect_ratio

        # Resolution (720p, 1080p, 4k - Veo 3+ only)
        if request.resolution is not None:
            config_params["resolution"] = request.resolution

        # number_of_videos has default value of 1, always set if different from 1
        if request.number_of_videos != 1:
            config_params["number_of_videos"] = request.number_of_videos

        if request.generate_audio is not None:
            config_params["generate_audio"] = request.generate_audio

        if request.seed is not None:
            config_params["seed"] = int(request.seed)

        if request.negative_prompt is not None:
            config_params["negative_prompt"] = request.negative_prompt

        if request.enhance_prompt is not None:
            config_params["enhance_prompt"] = request.enhance_prompt

        if request.video is not None:
            video = _convert_to_video(request.video)

        # Validate image constraints before processing
        if request.image_list:
            reference_count = sum(
                1
                for img in request.image_list
                if img["type"] in ("first_frame", "reference")
            )
            last_frame_count = sum(
                1 for img in request.image_list if img["type"] == "last_frame"
            )
            reference_images_count = sum(
                1 for img in request.image_list if img["type"] in ("asset", "style")
            )
            has_first_frame = any(
                img["type"] == "first_frame" for img in request.image_list
            )

            # Validate only 1 reference/first_frame image
            if reference_count > 1:
                raise ValidationError(
                    f"Veo3 only supports 1 reference/first_frame image, got {reference_count}",
                    provider=config.provider,
                )

            # Validate only 1 last_frame image
            if last_frame_count > 1:
                raise ValidationError(
                    f"Veo3 only supports 1 last_frame image, got {last_frame_count}",
                    provider=config.provider,
                )

            # Validate last_frame requires first_frame
            if last_frame_count > 0 and not has_first_frame:
                raise ValidationError(
                    "Veo3 requires first_frame when using last_frame (interpolation mode)",
                    provider=config.provider,
                )

            # Validate first_frame and reference images are mutually exclusive
            if has_first_frame and reference_images_count > 0:
                raise ValidationError(
                    "Veo3 does not allow reference images (asset/style) when using first_frame (interpolation mode)",
                    provider=config.provider,
                )

            # Validate maximum 3 reference images
            if reference_images_count > 3:
                raise ValidationError(
                    f"Veo3 only supports up to 3 reference images (asset/style), got {reference_images_count}",
                    provider=config.provider,
                )

        for image_item in request.image_list or []:
            match image_item["type"]:
                case "first_frame" | "reference":
                    image = _convert_to_image(image_item["image"])
                case "last_frame":
                    config_params["last_frame"] = _convert_to_image(image_item["image"])
                case "asset":
                    if "reference_images" not in config_params:
                        config_params["reference_images"] = []

                    reference_images = cast(
                        list[AnyDict], config_params["reference_images"]
                    )
                    reference_images.append(
                        {
                            "image": _convert_to_image(image_item["image"]),
                            "reference_type": VideoGenerationReferenceType.ASSET,
                        }
                    )
                case "style":
                    if "reference_images" not in config_params:
                        config_params["reference_images"] = []

                    reference_images = cast(
                        list[AnyDict], config_params["reference_images"]
                    )
                    reference_images.append(
                        {
                            "image": _convert_to_image(image_item["image"]),
                            "reference_type": VideoGenerationReferenceType.STYLE,
                        }
                    )

        # Create GenerateVideosConfig and return kwargs dict
        generation_config = GenerateVideosConfig(**config_params)

        api_payload = {
            "prompt": prompt,
            "image": image,
            "video": video,
            "config": generation_config,
        }

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.info(
            "Mapped request to provider format",
            {"converted_request": api_payload},
            redact=True,
        )

        return api_payload

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        request_id: str,
        operation: GenerateVideosOperation,
    ) -> VideoGenerationResponse:
        """
        Convert google-genai operation response to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request_id: Our request ID (operation name)
            operation: Completed operation from google-genai

        Returns:
            Normalized VideoGenerationResponse
        """
        if not operation.done:
            raise GenerationFailedError(
                "Operation is not completed",
                provider=config.provider,
                raw_response={"operation": str(operation)},
            )

        if operation.error:
            raise GenerationFailedError(
                f"Video generation failed: {operation.error}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=operation.model_dump(),
            )

        # Extract video from operation.response
        if not operation.response:
            raise GenerationFailedError(
                "No response in completed operation",
                provider=config.provider,
                raw_response={"operation": str(operation)},
            )

        generated_videos = operation.response.generated_videos
        if not generated_videos or len(generated_videos) == 0:
            raise GenerationFailedError(
                "No generated videos in response",
                provider=config.provider,
                raw_response={"operation": str(operation)},
            )

        video_obj = generated_videos[0].video
        if video_obj is None:
            raise GenerationFailedError(
                "Video object is None in generated_videos",
                provider=config.provider,
                raw_response={"operation": str(operation)},
            )

        video: MediaType
        if video_obj.uri:
            video = video_obj.uri
        else:
            # Ensure video_bytes is bytes for MediaContent
            video_bytes = video_obj.video_bytes
            mime_type = video_obj.mime_type

            if video_bytes is None or mime_type is None:
                raise GenerationFailedError(
                    "Video object has no URI and no video_bytes/mime_type",
                    provider=config.provider,
                    raw_response={"operation": str(operation)},
                )

            if isinstance(video_bytes, str):
                video_bytes = video_bytes.encode()

            video = MediaContent(
                content=video_bytes,
                content_type=mime_type,
            )

        return VideoGenerationResponse(
            request_id=request_id,
            video=video,
            content_type=video_obj.mime_type,
            status="completed",
            raw_response=operation.model_dump(),
            provider_metadata={},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Handle errors from google-genai API."""
        if isinstance(ex, TarashException):
            return ex

        # httpx timeout errors
        if isinstance(ex, httpx.TimeoutException):
            return TimeoutError(
                f"Request timed out: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
                timeout_seconds=config.timeout,
            )

        # httpx connection errors
        if isinstance(ex, (httpx.ConnectError, httpx.NetworkError)):
            return HTTPConnectionError(
                f"Connection error: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
            )

        # aiohttp errors (if aiohttp is used instead of httpx)
        # Check exception type names for aiohttp errors to avoid import issues
        if has_aiohttp:
            ex_type_name = type(ex).__name__
            ex_module = type(ex).__module__

            # aiohttp timeout errors
            if ex_module.startswith("aiohttp") and ex_type_name in (
                "ServerTimeoutError",
                "ClientTimeout",
            ):
                return TimeoutError(
                    f"Request timed out: {str(ex)}",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={"error": str(ex)},
                    timeout_seconds=config.timeout,
                )

            # aiohttp connection errors
            if ex_module.startswith("aiohttp") and ex_type_name in (
                "ClientConnectionError",
                "ClientConnectorError",
            ):
                return HTTPConnectionError(
                    f"Connection error: {str(ex)}",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={"error": str(ex)},
                )

        # Google GenAI ClientError (includes validation and HTTP errors)
        if has_genai and isinstance(ex, ClientError):
            error_code = ex.code
            error_message: str = ex.message or str(ex)
            # ex.details is a dict from google-genai SDK, but type is not exported
            # Use type: ignore since SDK doesn't export details type
            error_details = cast(AnyDict, ex.details if ex.details else {})  # type: ignore[arg-type]
            raw_response: AnyDict = {
                "status_code": error_code,
                "message": error_message,
                "error_details": error_details,
                "error_type": ex.status,
            }

            # Validation errors (400, 422)
            if error_code == 400 or error_code == 422:
                return ValidationError(
                    error_message,
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                )

            # Content moderation errors (403)
            if error_code == 403:
                return ContentModerationError(
                    error_message,
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                )

            # All other HTTP errors
            return HTTPError(
                error_message,
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=raw_response,
                status_code=error_code,
            )

        # Unknown errors
        log_error(
            f"Veo3 unknown error: {str(ex)}",
            context={
                "provider": config.provider,
                "model": config.model,
                "request_id": request_id,
                "error_type": type(ex).__name__,
            },
            logger_name=_LOGGER_NAME,
            exc_info=True,
        )
        return GenerationFailedError(
            f"Error while generating video: {str(ex)}",
            provider=config.provider,
            model=config.model,
            request_id=request_id,
            raw_response={
                "error": str(ex),
                "error_type": type(ex).__name__,
                "traceback": traceback.format_exc(),
            },
        )

    # ==================== Image Generation Methods ====================

    def _convert_image_request(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
    ) -> AnyDict:
        """Convert ImageGenerationRequest to Google GenAI format.

        Args:
            config: Image generation configuration
            request: Image generation request

        Returns:
            Dict with parameters ready for google-genai generate_images API
        """
        field_mappers = get_google_image_field_mappers(config.model)
        api_params = apply_field_mappers(field_mappers, request)

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.info(
            "Mapped request to provider format",
            {"converted_request": api_params},
            redact=True,
        )

        return api_params

    def _convert_image_response(
        self,
        config: ImageGenerationConfig,
        request_id: str,
        genai_response: AnyDict,
    ) -> ImageGenerationResponse:
        """Convert Google GenAI Imagen response to ImageGenerationResponse.

        Args:
            config: Image generation configuration
            request_id: Our request ID
            genai_response: Response from google-genai with generated_images

        Returns:
            Normalized ImageGenerationResponse
        """
        # Extract image URLs from response
        # Google GenAI returns {"generated_images": [GeneratedImage, ...]}
        generated_images = genai_response.get("generated_images", [])
        image_urls: list[str] = []

        for gen_img in generated_images:
            img = getattr(gen_img, "image", None)
            if img is None:
                continue

            if getattr(img, "gcs_uri", None):
                image_urls.append(str(img.gcs_uri))
                continue

            img_bytes = getattr(img, "image_bytes", None)
            if not isinstance(img_bytes, (bytes, bytearray, str)):
                continue

            mime_type = getattr(img, "mime_type", None) or "image/png"
            image_urls.append(_bytes_to_data_uri(img_bytes, mime_type))

        return ImageGenerationResponse(
            request_id=request_id,
            images=image_urls,
            content_type="image/png",
            status="completed",
            is_mock=False,
            raw_response=genai_response,
            provider_metadata={
                "model": config.model,
                "provider": config.provider,
            },
        )

    def _convert_gemini_image_response(
        self,
        config: ImageGenerationConfig,
        request_id: str,
        response: Any,
    ) -> ImageGenerationResponse:
        """Convert Gemini generate_content response to ImageGenerationResponse.

        Gemini image models return images via response.parts[].inline_data.

        Args:
            config: Image generation configuration
            request_id: Our request ID
            response: Response from google-genai generate_content

        Returns:
            Normalized ImageGenerationResponse with base64 data URIs
        """
        images: list[str] = []

        for part in getattr(response, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if not inline_data:
                continue

            img_bytes = getattr(inline_data, "data", None)
            if not isinstance(img_bytes, (bytes, bytearray, str)):
                continue

            mime_type = getattr(inline_data, "mime_type", None) or "image/png"
            images.append(_bytes_to_data_uri(img_bytes, mime_type))

        # Try model_dump for raw_response, fallback to string representation
        raw_response: AnyDict
        if hasattr(response, "model_dump"):
            raw_response = response.model_dump()
        else:
            raw_response = {"response": str(response)}

        return ImageGenerationResponse(
            request_id=request_id,
            images=images,
            content_type="image/png",
            status="completed",
            is_mock=False,
            raw_response=raw_response,
            provider_metadata={
                "model": config.model,
                "provider": config.provider,
            },
        )

    # ==================== Image Generation Helpers ====================

    def _build_gemini_content_kwargs(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
    ) -> AnyDict:
        """Build kwargs for client.models.generate_content (Gemini image models)."""
        api_params = self._convert_image_request(config, request)

        image_config_params: AnyDict = {}
        if api_params.get("aspect_ratio"):
            image_config_params["aspect_ratio"] = api_params["aspect_ratio"]

        return {
            "model": config.model,
            "contents": api_params.get("prompt", request.prompt),
            "config": GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=ImageConfig(**image_config_params)
                if image_config_params
                else None,
            ),
        }

    def _build_imagen_kwargs(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
    ) -> AnyDict:
        """Build kwargs for client.models.generate_images (Imagen models)."""
        api_params = self._convert_image_request(config, request)
        prompt = api_params.pop("prompt", request.prompt)

        return {
            "model": config.model,
            "prompt": prompt,
            "config": GenerateImagesConfig(**api_params) if api_params else None,
        }

    def _handle_image_error(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Adapt image config to video error handler."""
        video_config = VideoGenerationConfig(
            model=config.model,
            provider=config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            provider_config=config.provider_config,
        )
        return self._handle_error(video_config, request_id, ex)

    # ==================== Image Generation Methods ====================

    @handle_video_generation_errors
    async def generate_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate image asynchronously via Google GenAI.

        Args:
            config: Image generation configuration
            request: Image generation request
            on_progress: Optional async callback for progress updates (unused for images)

        Returns:
            ImageGenerationResponse with generated images
        """
        client = self._get_client(config, "async")
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        request_id = f"google-image-{uuid.uuid4()}"
        logger = logger.with_request_id(request_id)

        logger.debug("Starting image generation API call")

        try:
            if _is_gemini_image_model(config.model):
                kwargs = self._build_gemini_content_kwargs(config, request)
                gemini_response: GenerateContentResponse = (
                    await client.models.generate_content(**kwargs)
                )
                result = self._convert_gemini_image_response(
                    config, request_id, gemini_response
                )
            else:
                kwargs = self._build_imagen_kwargs(config, request)
                imagen_response: GenerateImagesResponse = (
                    await client.models.generate_images(**kwargs)
                )
                result = self._convert_image_response(
                    config,
                    request_id,
                    {"generated_images": imagen_response.generated_images},
                )

            logger.info(
                "Final generated image response",
                {"response": result},
                redact=True,
            )

            return result

        except Exception as ex:
            raise self._handle_image_error(config, request, request_id, ex) from ex

    @handle_video_generation_errors
    def generate_image(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: SyncImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate image synchronously via Google GenAI.

        Args:
            config: Image generation configuration
            request: Image generation request
            on_progress: Optional sync callback for progress updates (unused for images)

        Returns:
            ImageGenerationResponse with generated images
        """
        client = self._get_client(config, "sync")
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        request_id = f"google-image-{uuid.uuid4()}"
        logger = logger.with_request_id(request_id)

        logger.debug("Starting image generation API call")

        try:
            if _is_gemini_image_model(config.model):
                kwargs = self._build_gemini_content_kwargs(config, request)
                gemini_response: GenerateContentResponse = (
                    client.models.generate_content(**kwargs)
                )
                result = self._convert_gemini_image_response(
                    config, request_id, gemini_response
                )
            else:
                kwargs = self._build_imagen_kwargs(config, request)
                imagen_response: GenerateImagesResponse = client.models.generate_images(
                    **kwargs
                )
                result = self._convert_image_response(
                    config,
                    request_id,
                    {"generated_images": imagen_response.generated_images},
                )

            logger.info(
                "Final generated image response",
                {"response": result},
                redact=True,
            )

            return result

        except Exception as ex:
            raise self._handle_image_error(config, request, request_id, ex) from ex

    # ==================== Video Generation Methods ====================

    @handle_video_generation_errors
    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video asynchronously via Veo3 with async progress callback.

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional async callback for progress updates

        Returns:
            Final VideoGenerationResponse when complete
        """
        client = self._get_client(config, "async")
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)

        # Build Veo3 input (let validation errors propagate)
        veo3_kwargs = self._convert_request(config, request)

        logger.debug("Starting API call")

        request_id = "unknown"
        try:
            operation = await client.models.generate_videos(
                model=config.model,
                **veo3_kwargs,
            )

            request_id = operation.name or "unknown"
            logger = logger.with_request_id(request_id)

            logger.debug("Request submitted")

            # Poll for completion
            poll_attempts = 0
            while poll_attempts < config.max_poll_attempts:
                if operation.done:
                    break

                start = time.time()
                if on_progress:
                    update = parse_veo3_operation(operation)
                    result = on_progress(update)
                    if asyncio.iscoroutine(result):
                        await result

                # Log progress
                logger.info(
                    "Progress status update",
                    {"status": "processing" if not operation.done else "completed"},
                )

                end_time = time.time()
                if end_time - start < config.poll_interval:
                    await asyncio.sleep(config.poll_interval - (end_time - start))

                # Get updated operation status
                operation = await client.operations.get(operation)
                poll_attempts += 1

            if not operation.done:
                timeout_seconds = config.max_poll_attempts * config.poll_interval
                raise TimeoutError(
                    f"Video generation timed out after {config.max_poll_attempts} attempts ({timeout_seconds}s)",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={
                        "status": "timeout",
                        "poll_attempts": config.max_poll_attempts,
                    },
                    timeout_seconds=timeout_seconds,
                )

            logger.debug(
                "Request complete",
                {"response": operation},
                redact=True,
            )

            # Convert final response
            response = self._convert_response(config, request_id, operation)

            logger.info(
                "Final generated response",
                {"response": response},
                redact=True,
            )

            return response

        except Exception as ex:
            raise self._handle_error(config, request_id, ex) from ex

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: SyncProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video synchronously (blocking).

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional callback for progress updates

        Returns:
            Final VideoGenerationResponse
        """
        client = self._get_client(config, "sync")
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)

        # Build Veo3 input (let validation errors propagate)
        veo3_kwargs = self._convert_request(config, request)

        logger.debug("Starting API call")

        request_id = "unknown"
        try:
            operation = client.models.generate_videos(
                model=config.model,
                **veo3_kwargs,
            )

            request_id = operation.name or "unknown"
            logger = logger.with_request_id(request_id)

            logger.debug("Request submitted")

            # Poll for completion
            poll_attempts = 0
            while poll_attempts < config.max_poll_attempts:
                # Parse and report progress
                if on_progress:
                    update = parse_veo3_operation(operation)
                    on_progress(update)

                # Log progress
                logger.info(
                    "Progress status update",
                    {"status": "processing" if not operation.done else "completed"},
                )

                # Check if done
                if operation.done:
                    break

                # Wait before next poll
                time.sleep(config.poll_interval)

                # Get updated operation status
                operation = client.operations.get(operation)
                poll_attempts += 1

            if not operation.done:
                timeout_seconds = config.max_poll_attempts * config.poll_interval
                raise TimeoutError(
                    f"Video generation timed out after {config.max_poll_attempts} attempts ({timeout_seconds}s)",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={
                        "status": "timeout",
                        "poll_attempts": config.max_poll_attempts,
                    },
                    timeout_seconds=timeout_seconds,
                )

            logger.debug(
                "Request complete",
                {"response": operation},
                redact=True,
            )

            # Convert final response
            response = self._convert_response(config, request_id, operation)

            logger.info(
                "Final generated response",
                {"response": response},
                redact=True,
            )

            return response

        except Exception as ex:
            raise self._handle_error(config, request_id, ex) from ex


# ==================== Image Generation Field Mappings ====================

# Gemini 2.5 Flash Image ("Nano Banana") field mappings
NANO_BANANA_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "number_of_images": passthrough_field_mapper("n"),
    "safety_filter_level": extra_params_field_mapper("safety_filter_level"),
    "person_generation": extra_params_field_mapper("person_generation"),
}

# Imagen 3 field mappings
IMAGEN3_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "number_of_images": passthrough_field_mapper("n"),
    "safety_filter_level": extra_params_field_mapper("safety_filter_level"),
    "person_generation": extra_params_field_mapper("person_generation"),
    "language": extra_params_field_mapper("language"),
    "output_mime_type": extra_params_field_mapper("output_mime_type"),
    "output_compression_quality": extra_params_field_mapper(
        "output_compression_quality"
    ),
}

# Generic image field mappings (fallback)
GENERIC_IMAGE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "number_of_images": passthrough_field_mapper("n"),
}

# Image model registry for Google GenAI
GOOGLE_IMAGE_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    # Gemini Flash Image (Nano Banana)
    "gemini-2.5-flash-image": NANO_BANANA_FIELD_MAPPERS,  # Prefix match
    "gemini-2.5-flash-image-preview": NANO_BANANA_FIELD_MAPPERS,
    "gemini-3-pro-image-preview": NANO_BANANA_FIELD_MAPPERS,
    # Imagen 3
    "imagen-3": IMAGEN3_FIELD_MAPPERS,  # Prefix match
    "imagen-3.0-generate-001": IMAGEN3_FIELD_MAPPERS,
    "imagen-3.0-generate-002": IMAGEN3_FIELD_MAPPERS,
    "imagen-3.0-fast-generate-001": IMAGEN3_FIELD_MAPPERS,
}


def get_google_image_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get field mappers for Google GenAI image model.

    Args:
        model_name: Model name (e.g., "imagen-3.0-generate-002")

    Returns:
        Dict mapping API field names to FieldMapper objects
    """
    return get_field_mappers_from_registry(
        model_name, GOOGLE_IMAGE_MODEL_REGISTRY, GENERIC_IMAGE_FIELD_MAPPERS
    )
