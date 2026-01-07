# Image Gateway Design Plan

## Executive Summary

This document outlines the architecture and implementation plan for the **Image Gateway**, a unified interface for AI image generation models. The design reuses the proven architecture from the Video Gateway while adapting it for image-specific requirements.

**Target Providers:**
- **Fal** (FLUX.2 Pro/Dev, Flux 1.1 Pro Ultra, Recraft V3, Ideogram V3, Z-Image-Turbo)
- **OpenAI** (GPT-Image-1.5, GPT-Image-1, DALL-E 3) - GPT-Image-1.5 is latest (Dec 2025)
- **Replicate** (FLUX.2, Flux 1.1 Pro Ultra, SD3.5, Z-Image, community models)
- **Google** (Gemini 2.5 Flash Image "Nano Banana", Gemini 3 Pro Image "Nano Banana Pro", Imagen 3)
- **Stability AI** (Stable Diffusion 3.5, Stable Image Core/Ultra) - Direct API
- **Alibaba** (Z-Image-Turbo, Z-Image-Edit) - New efficient models

**Key Design Goals:**
1. Maximum code reuse from Video Gateway
2. Shared infrastructure for polling, error handling, orchestration
3. Image-specific parameter handling (size, quality, style)
4. Consistent API across all providers

---

## 1. Architecture Overview

### 1.1 Directory Structure

```
packages/tarash-gateway/
├── src/tarash/tarash_gateway/
│   ├── logging.py                    # ✅ REUSE AS-IS
│   ├── common/                       # 🆕 NEW - Shared utilities
│   │   ├── __init__.py
│   │   ├── exceptions.py             # ♻️ REFACTORED - Base exceptions
│   │   ├── orchestrator.py           # ♻️ REFACTORED - Generic orchestrator
│   │   ├── registry.py               # ♻️ REFACTORED - Generic registry
│   │   ├── utils.py                  # ♻️ REFACTORED - Shared utils
│   │   └── field_mappers.py          # ♻️ REFACTORED - Base field mapper framework
│   ├── video/
│   │   ├── __init__.py
│   │   ├── api.py                    # ✅ KEEP - Video-specific API
│   │   ├── models.py                 # ✅ KEEP - Video models
│   │   ├── exceptions.py             # ✅ KEEP - Video exceptions (extend common)
│   │   ├── orchestrator.py           # ✅ KEEP - Video orchestrator (use common)
│   │   ├── registry.py               # ✅ KEEP - Video registry (use common)
│   │   ├── utils.py                  # ✅ KEEP - Video utils (use common)
│   │   ├── mock.py                   # ✅ KEEP - Video mock
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── field_mappers.py      # ✅ KEEP - Video field mappers
│   │       ├── fal.py                # ✅ KEEP
│   │       ├── openai.py             # ✅ KEEP
│   │       ├── replicate.py          # ✅ KEEP
│   │       └── ...
│   └── image/                        # 🆕 NEW - Image gateway
│       ├── __init__.py
│       ├── api.py                    # Public API entry point
│       ├── models.py                 # Image-specific models
│       ├── exceptions.py             # Image exceptions (extend common)
│       ├── orchestrator.py           # Image orchestrator (use common)
│       ├── registry.py               # Image registry (use common)
│       ├── utils.py                  # Image utils (use common)
│       ├── mock.py                   # Image mock system
│       └── providers/
│           ├── __init__.py
│           ├── field_mappers.py      # Image field mapper registry
│           ├── fal.py                # Fal image provider
│           ├── openai.py             # OpenAI DALL-E provider
│           ├── replicate.py          # Replicate image provider
│           └── gemini.py             # Gemini Imagen provider
└── tests/
    ├── conftest.py                   # ♻️ UPDATE - Support both video + image
    ├── unit/
    │   ├── video/                    # ✅ KEEP
    │   ├── image/                    # 🆕 NEW
    │   └── common/                   # 🆕 NEW - Test shared utilities
    └── e2e/
        ├── video/                    # ✅ KEEP
        └── image/                    # 🆕 NEW
```

**Legend:**
- ✅ **REUSE AS-IS**: No changes needed
- ✅ **KEEP**: Keep existing, may need minor updates to use common/
- ♻️ **REFACTORED**: Extract to common/, keep domain-specific in video/image
- 🆕 **NEW**: Create for image gateway

---

## 2. Code Refactoring Strategy

### 2.1 Phase 1: Extract Common Infrastructure

#### 2.1.1 Common Exceptions (`common/exceptions.py`)

**What to extract:**
- Base `TarashException` class
- Generic exceptions: `ValidationError`, `ContentModerationError`, `HTTPError`, `GenerationFailedError`, `HTTPConnectionError`, `TimeoutError`
- Error classification utilities: `is_retryable_error()`
- Error handling decorator factory (parameterized for video/image)

**What stays domain-specific:**
- Domain-specific error messages (but use same exception types)

**Implementation:**

```python
# common/exceptions.py
class TarashException(Exception):
    """Base exception for all Tarash generation errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ):
        self.message = message
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self.raw_response = raw_response
        super().__init__(message)

class ValidationError(TarashException):
    """Raised when request validation fails (400-level errors)."""
    pass

class ContentModerationError(TarashException):
    """Raised when content policy is violated (403-level errors)."""
    pass

# ... other exceptions

def is_retryable_error(error: Exception) -> bool:
    """Determine if error is retryable for fallback."""
    if isinstance(error, ValidationError):
        return False
    if isinstance(error, ContentModerationError):
        return False
    if isinstance(error, HTTPError) and 400 <= error.status_code < 500:
        return error.status_code == 429  # Only 429 is retryable
    return True  # Network errors, 5xx, timeouts are retryable

def create_error_handler(domain: str):
    """Factory for domain-specific error handlers."""
    def handle_errors(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except (TarashException, PydanticValidationError):
                raise
            except Exception as ex:
                # Log and wrap unknown exceptions
                logger.exception(f"{domain.title()} generation error")
                raise TarashException(
                    f"Unexpected {domain} generation error: {str(ex)}",
                    raw_response={"error": str(ex), "traceback": traceback.format_exc()}
                ) from ex

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar for sync
            ...

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return handle_errors
```

**Migration:**
- `video/exceptions.py`: Import from `common.exceptions`, add alias for backward compatibility
- `image/exceptions.py`: Import from `common.exceptions`

#### 2.1.2 Common Orchestrator (`common/orchestrator.py`)

**What to extract:**
- Generic `ExecutionOrchestrator` class (parameterized by config/request/response types)
- Fallback chain collection logic
- Retry logic with error classification
- Execution metadata tracking
- Attempt metadata models

**What stays domain-specific:**
- Domain-specific type hints (but use generics)

**Implementation:**

```python
# common/orchestrator.py
from typing import TypeVar, Generic, Protocol, Callable, Awaitable

TConfig = TypeVar("TConfig")  # VideoGenerationConfig | ImageGenerationConfig
TRequest = TypeVar("TRequest")  # VideoGenerationRequest | ImageGenerationRequest
TResponse = TypeVar("TResponse")  # VideoGenerationResponse | ImageGenerationResponse
TUpdate = TypeVar("TUpdate")  # VideoGenerationUpdate | ImageGenerationUpdate

class GenerationConfig(Protocol):
    """Protocol for generation configs."""
    provider: str
    model: str
    fallback_configs: list["GenerationConfig"] | None

class GenerationResponse(Protocol):
    """Protocol for generation responses."""
    request_id: str
    status: str
    execution_metadata: ExecutionMetadata | None

@dataclass
class AttemptMetadata:
    """Metadata for a single generation attempt."""
    attempt_number: int
    config_index: int
    provider: str
    model: str
    status: Literal["success", "error"]
    start_time: float
    end_time: float
    duration_seconds: float
    error_type: str | None = None
    error_message: str | None = None
    request_id: str | None = None

@dataclass
class ExecutionMetadata:
    """Metadata about execution across fallback chain."""
    total_attempts: int
    successful_attempt: int | None
    attempts: list[AttemptMetadata]
    fallback_triggered: bool
    configs_in_chain: int

class ExecutionOrchestrator(Generic[TConfig, TRequest, TResponse, TUpdate]):
    """Generic execution orchestrator for fallback chains."""

    def __init__(
        self,
        domain: str,  # "video" or "image"
        handler_getter: Callable[[TConfig], Any],  # Function to get provider handler
    ):
        self.domain = domain
        self.handler_getter = handler_getter

    def _collect_fallback_chain(self, config: TConfig) -> list[TConfig]:
        """Collect all configs in fallback chain (depth-first)."""
        chain = [config]
        if hasattr(config, "fallback_configs") and config.fallback_configs:
            for fallback in config.fallback_configs:
                chain.extend(self._collect_fallback_chain(fallback))
        return chain

    async def execute_with_fallback_async(
        self,
        config: TConfig,
        request: TRequest,
        on_progress: Callable[[TUpdate], Awaitable[None] | None] | None = None,
    ) -> TResponse:
        """Execute generation with fallback chain (async)."""
        chain = self._collect_fallback_chain(config)
        attempts = []

        for idx, cfg in enumerate(chain):
            attempt_start = time.time()

            try:
                handler = self.handler_getter(cfg)

                # Call handler's generate method (video/image specific)
                if self.domain == "video":
                    response = await handler.generate_video_async(cfg, request, on_progress)
                elif self.domain == "image":
                    response = await handler.generate_image_async(cfg, request, on_progress)

                # Success - record attempt and return
                attempt_end = time.time()
                attempts.append(AttemptMetadata(
                    attempt_number=len(attempts) + 1,
                    config_index=idx,
                    provider=cfg.provider,
                    model=cfg.model,
                    status="success",
                    start_time=attempt_start,
                    end_time=attempt_end,
                    duration_seconds=attempt_end - attempt_start,
                    request_id=response.request_id,
                ))

                # Attach metadata
                response.execution_metadata = ExecutionMetadata(
                    total_attempts=len(attempts),
                    successful_attempt=len(attempts),
                    attempts=attempts,
                    fallback_triggered=len(attempts) > 1,
                    configs_in_chain=len(chain),
                )

                return response

            except Exception as ex:
                attempt_end = time.time()
                attempts.append(AttemptMetadata(
                    attempt_number=len(attempts) + 1,
                    config_index=idx,
                    provider=cfg.provider,
                    model=cfg.model,
                    status="error",
                    start_time=attempt_start,
                    end_time=attempt_end,
                    duration_seconds=attempt_end - attempt_start,
                    error_type=type(ex).__name__,
                    error_message=str(ex),
                    request_id=getattr(ex, "request_id", None),
                ))

                # Decide if we should retry
                if not is_retryable_error(ex):
                    # Non-retryable error - stop immediately
                    logger.error(f"Non-retryable error, stopping fallback chain")
                    raise

                # Retryable error - continue to next config if available
                if idx < len(chain) - 1:
                    logger.info(f"Retryable error, trying next config in chain")
                    continue
                else:
                    # Last config failed - raise
                    logger.error(f"All configs failed")
                    raise

        # Should never reach here
        raise RuntimeError("Fallback chain exhausted without raising")

    def execute_with_fallback(
        self,
        config: TConfig,
        request: TRequest,
        on_progress: Callable[[TUpdate], None] | None = None,
    ) -> TResponse:
        """Execute generation with fallback chain (sync)."""
        # Similar to async but without await
        ...
```

**Migration:**
- `video/orchestrator.py`: Import from `common.orchestrator`, instantiate with domain="video"
- `image/orchestrator.py`: Import from `common.orchestrator`, instantiate with domain="image"

#### 2.1.3 Common Registry (`common/registry.py`)

**What to extract:**
- Generic provider registry pattern
- Singleton handler management
- Custom provider registration

**What stays domain-specific:**
- Domain-specific handler initialization (but use callbacks)

**Implementation:**

```python
# common/registry.py
TConfig = TypeVar("TConfig")
THandler = TypeVar("THandler")

class ProviderRegistry(Generic[TConfig, THandler]):
    """Generic provider handler registry."""

    def __init__(self):
        self._handler_instances: dict[str, THandler] = {}
        self._custom_providers: dict[str, THandler] = {}

    def register_provider(self, provider_name: str, handler: THandler) -> None:
        """Register custom provider handler."""
        self._custom_providers[provider_name] = handler

    def get_handler(
        self,
        config: TConfig,
        builtin_handlers: dict[str, Callable[[], THandler]],
    ) -> THandler:
        """Get or create provider handler (singleton pattern)."""

        # Check mock mode
        if hasattr(config, "mock") and config.mock and config.mock.enabled:
            # Return mock handler (domain-specific)
            return builtin_handlers.get("mock", lambda: None)()

        provider = config.provider

        # Check custom providers first
        if provider in self._custom_providers:
            return self._custom_providers[provider]

        # Check built-in providers
        if provider not in self._handler_instances:
            if provider in builtin_handlers:
                self._handler_instances[provider] = builtin_handlers[provider]()
            else:
                raise ValueError(f"Unknown provider: {provider}")

        return self._handler_instances[provider]
```

**Migration:**
- `video/registry.py`: Use `ProviderRegistry` with video handlers
- `image/registry.py`: Use `ProviderRegistry` with image handlers

#### 2.1.4 Common Utils (`common/utils.py`)

**What to extract:**
- `download_media_from_url()` / `download_media_from_url_async()`
- `convert_to_data_url()`
- `get_filename_from_url()`
- MediaType handling utilities

**What stays domain-specific:**
- Domain-specific validation (e.g., `validate_duration()` stays in video/)

**Implementation:**

```python
# common/utils.py
def convert_to_data_url(media: MediaContent) -> str:
    """Convert MediaContent to data URL."""
    if media.data_url:
        return media.data_url
    elif media.url:
        data = download_media_from_url(media.url)
        # Detect media type and encode
        import base64
        media_type = media.media_type or "image/jpeg"  # Default
        b64_data = base64.b64encode(data).decode("utf-8")
        return f"data:{media_type};base64,{b64_data}"
    else:
        raise ValueError("MediaContent must have either data_url or url")

async def download_media_from_url_async(url: str, timeout: int = 30) -> bytes:
    """Download media from URL asynchronously."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content

def download_media_from_url(url: str, timeout: int = 30) -> bytes:
    """Download media from URL synchronously."""
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content

def get_filename_from_url(url: str) -> str | None:
    """Extract filename from URL."""
    from urllib.parse import urlparse, unquote
    path = urlparse(url).path
    filename = unquote(path.split("/")[-1])
    return filename if filename else None
```

**Migration:**
- `video/utils.py`: Import from `common.utils`, keep `validate_duration()` and video-specific utils
- `image/utils.py`: Import from `common.utils`, add image-specific utils (e.g., `validate_size()`)

#### 2.1.5 Common Field Mapper Framework (`common/field_mappers.py`)

**What to extract:**
- `FieldMapper` dataclass
- `apply_field_mappers()` function
- Registry lookup utilities: `get_field_mappers_from_registry()`
- Generic field mapper factories: `passthrough_field_mapper()`, `extra_params_field_mapper()`

**What stays domain-specific:**
- Domain-specific mappers: `duration_field_mapper()`, `video_url_field_mapper()` (stay in video)
- Image-specific mappers: `size_field_mapper()`, `quality_field_mapper()` (new in image)

**Implementation:**

```python
# common/field_mappers.py
@dataclass
class FieldMapper:
    """Maps a request field to an API field with conversion."""
    source_field: str
    converter: Callable[[Any, Any], Any]  # (request, value) -> converted_value
    required: bool = False

def apply_field_mappers(
    field_mappers: dict[str, FieldMapper],
    request: Any,  # VideoGenerationRequest | ImageGenerationRequest
) -> dict[str, Any]:
    """Apply field mappers to convert request to API format."""
    result = {}

    for api_field_name, mapper in field_mappers.items():
        source_value = getattr(request, mapper.source_field, None)

        if mapper.required and source_value is None:
            raise ValueError(f"Required field '{mapper.source_field}' is missing")

        converted_value = mapper.converter(request, source_value)

        if mapper.required and converted_value is None:
            raise ValueError(f"Required field '{api_field_name}' cannot be None")

        if converted_value is not None and converted_value != [] and converted_value != {}:
            result[api_field_name] = converted_value

    return result

def get_field_mappers_from_registry(
    model_name: str,
    registry: dict[str, dict[str, FieldMapper]],
    fallback_mappers: dict[str, FieldMapper],
) -> dict[str, FieldMapper]:
    """Get field mappers with prefix matching support."""
    # Exact match
    if model_name in registry:
        return registry[model_name]

    # Prefix matching - longest match wins
    matching_prefix = None
    for registry_key in registry:
        if model_name.startswith(registry_key):
            if matching_prefix is None or len(registry_key) > len(matching_prefix):
                matching_prefix = registry_key

    if matching_prefix:
        return registry[matching_prefix]

    # Fallback
    return fallback_mappers

def passthrough_field_mapper(source_field: str, required: bool = False) -> FieldMapper:
    """Create field mapper that passes value through unchanged."""
    return FieldMapper(
        source_field=source_field,
        converter=lambda req, val: val,
        required=required,
    )

def extra_params_field_mapper(extra_param_key_name: str) -> FieldMapper:
    """Create field mapper that extracts from extra_params."""
    def converter(request, _):
        if not hasattr(request, "extra_params") or not request.extra_params:
            return None
        return request.extra_params.get(extra_param_key_name)

    return FieldMapper(
        source_field="extra_params",
        converter=converter,
        required=False,
    )
```

**Migration:**
- `video/providers/field_mappers.py`: Import base from `common.field_mappers`, keep video-specific mappers
- `image/providers/field_mappers.py`: Import base from `common.field_mappers`, add image-specific mappers

---

### 2.2 Phase 2: Create Image Gateway

#### 2.2.1 Image Models (`image/models.py`)

**Key Differences from Video:**

| Video | Image |
|-------|-------|
| `duration_seconds` | ~~removed~~ |
| `generate_audio` | ~~removed~~ |
| `first_frame`, `last_frame` | ~~removed~~ |
| ~~N/A~~ | `size` (e.g., "1024x1024", "landscape", "portrait") |
| ~~N/A~~ | `quality` (e.g., "standard", "hd") |
| ~~N/A~~ | `style` (e.g., "natural", "vivid") |
| ~~N/A~~ | `n` (number of images to generate) |
| `video: str` (response) | `images: list[str]` (response) |

**Implementation:**

```python
# image/models.py
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Any
from common.field_mappers import FieldMapper

class ImageGenerationConfig(BaseModel):
    """Configuration for image generation."""

    model: str
    provider: str
    api_key: str
    base_url: str | None = None
    api_version: str | None = None
    timeout: int = 120  # Images typically faster than video
    max_poll_attempts: int = 60  # Fewer attempts needed
    poll_interval: float = 2.0  # Shorter interval
    mock: MockConfig | None = None
    fallback_configs: list["ImageGenerationConfig"] | None = None

    model_config = {"frozen": True}

class ImageGenerationRequest(BaseModel):
    """Flexible image generation request."""

    # Core fields
    prompt: str | None = None
    negative_prompt: str | None = None

    # Image parameters
    size: str | None = None  # "1024x1024", "1024x1792", "landscape_4_3", etc.
    quality: str | None = None  # "standard", "hd"
    style: str | None = None  # "natural", "vivid"
    n: int | None = None  # Number of images

    # Input images (for img2img, inpainting, etc.)
    image_list: list[MediaContent] | None = None
    mask_image: MediaContent | None = None  # For inpainting

    # Control parameters
    seed: int | None = None
    enhance_prompt: bool = False
    aspect_ratio: str | None = None  # Alternative to size
    resolution: str | None = None  # "1080p", "4k", etc.

    # Extra params for provider-specific features
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def capture_extra_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Capture unknown fields into extra_params."""
        # Same pattern as video
        ...

class ImageGenerationResponse(BaseModel):
    """Normalized image generation response."""

    request_id: str
    images: list[str]  # List of URLs
    status: Literal["pending", "processing", "completed", "failed"]
    is_mock: bool = False
    raw_response: dict[str, Any] | None = None
    execution_metadata: ExecutionMetadata | None = None
    revised_prompt: str | None = None  # Some providers revise prompts

class ImageGenerationUpdate(BaseModel):
    """Progress update during image generation."""

    request_id: str
    status: str
    message: str | None = None
    progress_percentage: float | None = None
    elapsed_time_seconds: float | None = None

# Protocol for image provider handlers
class ProviderHandler(Protocol):
    """Protocol that all image provider handlers must implement."""

    def _get_client(
        self,
        config: ImageGenerationConfig,
        client_type: Literal["sync", "async"],
    ) -> Any:
        """Get or create provider client."""
        ...

    def _validate_params(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
    ) -> dict[str, Any]:
        """Validate and return parameters."""
        ...

    def _convert_request(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
    ) -> dict[str, Any]:
        """Convert request to provider-specific format."""
        ...

    def _convert_response(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        request_id: str,
        provider_response: Any,
    ) -> ImageGenerationResponse:
        """Convert provider response to normalized format."""
        ...

    async def generate_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: Callable[[ImageGenerationUpdate], Awaitable[None] | None] | None = None,
    ) -> ImageGenerationResponse:
        """Generate image asynchronously."""
        ...

    def generate_image(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: Callable[[ImageGenerationUpdate], None] | None = None,
    ) -> ImageGenerationResponse:
        """Generate image synchronously."""
        ...
```

#### 2.2.2 Image Field Mappers (`image/providers/field_mappers.py`)

**Image-Specific Mappers:**

```python
# image/providers/field_mappers.py
from common.field_mappers import (
    FieldMapper,
    passthrough_field_mapper,
    extra_params_field_mapper,
)

def size_field_mapper(
    provider: str,
    model: str,
    allowed_sizes: list[str] | None = None,
) -> FieldMapper:
    """Create field mapper for image size with validation."""

    def converter(request: ImageGenerationRequest, val: Any) -> str | None:
        if val is None:
            return None

        # Validate against allowed sizes
        if allowed_sizes and val not in allowed_sizes:
            raise ValueError(
                f"Invalid size '{val}' for {provider}/{model}. "
                f"Allowed sizes: {allowed_sizes}"
            )

        return val

    return FieldMapper(
        source_field="size",
        converter=converter,
        required=False,
    )

def quality_field_mapper(
    allowed_qualities: list[str] | None = None,
) -> FieldMapper:
    """Create field mapper for image quality."""

    def converter(request: ImageGenerationRequest, val: Any) -> str | None:
        if val is None:
            return None

        if allowed_qualities and val not in allowed_qualities:
            raise ValueError(f"Invalid quality '{val}'. Allowed: {allowed_qualities}")

        return val

    return FieldMapper(
        source_field="quality",
        converter=converter,
        required=False,
    )

def style_field_mapper(
    allowed_styles: list[str] | None = None,
) -> FieldMapper:
    """Create field mapper for image style."""

    def converter(request: ImageGenerationRequest, val: Any) -> str | None:
        if val is None:
            return None

        if allowed_styles and val not in allowed_styles:
            raise ValueError(f"Invalid style '{val}'. Allowed: {allowed_styles}")

        return val

    return FieldMapper(
        source_field="style",
        converter=converter,
        required=False,
    )

def single_image_field_mapper(image_type: str | None = None) -> FieldMapper:
    """Extract single image from image_list."""

    def converter(request: ImageGenerationRequest, val: Any) -> str | None:
        if not request.image_list:
            return None

        # Filter by type if specified
        if image_type:
            images = [img for img in request.image_list if img.type == image_type]
        else:
            images = request.image_list

        if not images:
            return None

        if len(images) > 1:
            raise ValueError(
                f"Multiple images with type '{image_type}' found. "
                "Expected single image."
            )

        # Convert to URL
        return _convert_to_url(images[0])

    return FieldMapper(
        source_field="image_list",
        converter=converter,
        required=False,
    )

def image_list_field_mapper() -> FieldMapper:
    """Convert all images in image_list to URLs."""

    def converter(request: ImageGenerationRequest, val: Any) -> list[str] | None:
        if not request.image_list:
            return None

        return [_convert_to_url(img) for img in request.image_list]

    return FieldMapper(
        source_field="image_list",
        converter=converter,
        required=False,
    )

def mask_image_field_mapper() -> FieldMapper:
    """Convert mask image to URL."""

    def converter(request: ImageGenerationRequest, val: Any) -> str | None:
        if not request.mask_image:
            return None

        return _convert_to_url(request.mask_image)

    return FieldMapper(
        source_field="mask_image",
        converter=converter,
        required=False,
    )

# Helper
def _convert_to_url(media: MediaContent) -> str:
    """Convert MediaContent to URL or data URL."""
    if media.url:
        return media.url
    elif media.data_url:
        return media.data_url
    else:
        raise ValueError("MediaContent must have either url or data_url")
```

#### 2.2.3 Image Provider: OpenAI (`image/providers/openai.py`)

**Example Implementation:**

```python
# image/providers/openai.py
from common.exceptions import (
    ValidationError,
    ContentModerationError,
    HTTPError,
    GenerationFailedError,
    create_error_handler,
)
from image.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ProviderHandler,
)
from image.providers.field_mappers import (
    size_field_mapper,
    quality_field_mapper,
    style_field_mapper,
    passthrough_field_mapper,
)
from common.field_mappers import (
    apply_field_mappers,
    get_field_mappers_from_registry,
)

try:
    from openai import OpenAI, AsyncOpenAI
    from openai import exceptions as openai_exceptions

    APIStatusError = openai_exceptions.APIStatusError
except ImportError:
    pass

# Field mapper registry
DALLE3_FIELD_MAPPERS = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "size": size_field_mapper(
        provider="openai",
        model="dall-e-3",
        allowed_sizes=["1024x1024", "1024x1792", "1792x1024"],
    ),
    "quality": quality_field_mapper(allowed_qualities=["standard", "hd"]),
    "style": style_field_mapper(allowed_styles=["natural", "vivid"]),
    "n": passthrough_field_mapper("n"),  # DALL-E 3 only supports n=1
}

DALLE2_FIELD_MAPPERS = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "size": size_field_mapper(
        provider="openai",
        model="dall-e-2",
        allowed_sizes=["256x256", "512x512", "1024x1024"],
    ),
    "n": passthrough_field_mapper("n"),  # DALL-E 2 supports n=1-10
}

OPENAI_MODEL_REGISTRY = {
    "dall-e-3": DALLE3_FIELD_MAPPERS,
    "dall-e-2": DALLE2_FIELD_MAPPERS,
}

GENERIC_OPENAI_FIELD_MAPPERS = {
    "prompt": passthrough_field_mapper("prompt", required=True),
}

def get_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get field mappers for OpenAI image model."""
    return get_field_mappers_from_registry(
        model_name,
        OPENAI_MODEL_REGISTRY,
        GENERIC_OPENAI_FIELD_MAPPERS,
    )

class OpenAIProviderHandler:
    """OpenAI DALL-E image generation handler."""

    def __init__(self):
        self._client_cache: dict[str, Any] = {}

    def _get_client(
        self,
        config: ImageGenerationConfig,
        client_type: Literal["sync", "async"],
    ) -> OpenAI | AsyncOpenAI:
        """Get or create OpenAI client (cached)."""
        cache_key = f"{config.api_key}:{config.base_url or 'default'}:{client_type}"

        if cache_key not in self._client_cache:
            kwargs = {
                "api_key": config.api_key,
                "timeout": config.timeout,
            }
            if config.base_url:
                kwargs["base_url"] = config.base_url

            if client_type == "async":
                self._client_cache[cache_key] = AsyncOpenAI(**kwargs)
            else:
                self._client_cache[cache_key] = OpenAI(**kwargs)

        return self._client_cache[cache_key]

    def _convert_request(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
    ) -> dict[str, Any]:
        """Convert request to OpenAI API format."""
        mappers = get_field_mappers(config.model)
        params = apply_field_mappers(mappers, request)

        # Merge with extra_params
        if request.extra_params:
            params.update(request.extra_params)

        # Add model
        params["model"] = config.model

        return params

    def _convert_response(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        request_id: str,
        provider_response: Any,
    ) -> ImageGenerationResponse:
        """Convert OpenAI response to normalized format."""

        # Extract image URLs
        images = [img.url for img in provider_response.data]

        # Extract revised prompt (DALL-E 3 feature)
        revised_prompt = None
        if hasattr(provider_response.data[0], "revised_prompt"):
            revised_prompt = provider_response.data[0].revised_prompt

        return ImageGenerationResponse(
            request_id=request_id,
            images=images,
            status="completed",
            revised_prompt=revised_prompt,
            raw_response=provider_response.model_dump() if hasattr(provider_response, "model_dump") else {},
        )

    @create_error_handler("image")
    async def generate_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress=None,
    ) -> ImageGenerationResponse:
        """Generate image asynchronously with OpenAI."""

        try:
            client = self._get_client(config, "async")
            params = self._convert_request(config, request)

            # OpenAI doesn't support streaming for images, so no progress updates
            response = await client.images.generate(**params)

            request_id = f"openai_{response.created}"

            return self._convert_response(config, request, request_id, response)

        except APIStatusError as ex:
            raise self._handle_error(config, request, "", ex)

    @create_error_handler("image")
    def generate_image(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress=None,
    ) -> ImageGenerationResponse:
        """Generate image synchronously with OpenAI."""

        try:
            client = self._get_client(config, "sync")
            params = self._convert_request(config, request)

            response = client.images.generate(**params)

            request_id = f"openai_{response.created}"

            return self._convert_response(config, request, request_id, response)

        except APIStatusError as ex:
            raise self._handle_error(config, request, "", ex)

    def _handle_error(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Map OpenAI errors to Tarash exceptions."""

        if isinstance(ex, APIStatusError):
            if ex.status_code == 400:
                return ValidationError(
                    f"Invalid request: {ex.message}",
                    provider="openai",
                    model=config.model,
                    request_id=request_id,
                    raw_response=ex.response.json() if ex.response else {},
                )
            elif ex.status_code == 403:
                return ContentModerationError(
                    f"Content policy violation: {ex.message}",
                    provider="openai",
                    model=config.model,
                    request_id=request_id,
                    raw_response=ex.response.json() if ex.response else {},
                )
            else:
                return HTTPError(
                    f"HTTP error: {ex.message}",
                    provider="openai",
                    model=config.model,
                    request_id=request_id,
                    status_code=ex.status_code,
                    raw_response=ex.response.json() if ex.response else {},
                )

        return TarashException(
            f"Unknown error: {str(ex)}",
            provider="openai",
            model=config.model,
            request_id=request_id,
            raw_response={"error": str(ex), "error_type": type(ex).__name__},
        )
```

#### 2.2.4 Image Provider: Fal (`image/providers/fal.py`)

**Polling Pattern for Fal:**

Fal supports both instant and async image generation:
- **Instant models**: No polling needed (Flux Pro, SDXL)
- **Async models**: Polling required (some high-quality models)

```python
# image/providers/fal.py
import fal_client
from common.exceptions import create_error_handler
from image.models import ImageGenerationConfig, ImageGenerationRequest, ImageGenerationResponse

# Field mappers for different Fal image models
FLUX_PRO_FIELD_MAPPERS = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "image_size": size_field_mapper(
        provider="fal",
        model="flux-pro",
        allowed_sizes=["square", "portrait_4_3", "landscape_4_3", "landscape_16_9"],
    ),
    "num_images": passthrough_field_mapper("n"),
}

SDXL_FIELD_MAPPERS = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "image_size": size_field_mapper(
        provider="fal",
        model="sdxl",
        allowed_sizes=["square", "square_hd", "portrait_4_3", "landscape_4_3"],
    ),
    "num_images": passthrough_field_mapper("n"),
    "seed": passthrough_field_mapper("seed"),
}

FAL_MODEL_REGISTRY = {
    "fal-ai/flux-pro": FLUX_PRO_FIELD_MAPPERS,
    "fal-ai/flux/": FLUX_PRO_FIELD_MAPPERS,  # Prefix match
    "fal-ai/sdxl": SDXL_FIELD_MAPPERS,
}

class FalProviderHandler:
    """Fal image generation handler."""

    def __init__(self):
        self._sync_client_cache: dict[str, Any] = {}

    def _get_client(self, config, client_type):
        """Get Fal client (sync cached, async fresh)."""
        # Same pattern as video
        ...

    @create_error_handler("image")
    async def generate_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress=None,
    ) -> ImageGenerationResponse:
        """Generate image with Fal (async)."""

        client = self._get_client(config, "async")
        params = self._convert_request(config, request)

        # Submit request
        handler = await client.submit(config.model, arguments=params)
        request_id = handler.request_id

        # Check if streaming is supported
        if hasattr(handler, "iter_events"):
            # Async model with polling
            async for event in handler.iter_events(interval=config.poll_interval):
                if on_progress:
                    update = ImageGenerationUpdate(
                        request_id=request_id,
                        status=event.get("type", "processing"),
                        message=event.get("logs", ""),
                    )
                    result = on_progress(update)
                    if asyncio.iscoroutine(result):
                        await result

            result = await handler.get()
        else:
            # Instant model - result ready immediately
            result = await handler.get()

        return self._convert_response(config, request, request_id, result)

    def _convert_response(self, config, request, request_id, provider_response):
        """Convert Fal response to normalized format."""

        # Fal returns images in response.images list
        images = []
        if isinstance(provider_response, dict):
            if "images" in provider_response:
                images = [img["url"] for img in provider_response["images"]]
            elif "image" in provider_response:
                images = [provider_response["image"]["url"]]

        return ImageGenerationResponse(
            request_id=request_id,
            images=images,
            status="completed",
            raw_response=provider_response,
        )
```

#### 2.2.5 Image API (`image/api.py`)

**Public API Entry Point:**

```python
# image/api.py
from common.registry import ProviderRegistry
from common.orchestrator import ExecutionOrchestrator
from image.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ProviderHandler,
)
from image.providers import openai, fal, replicate, gemini
from image.mock import MockProviderHandler

# Provider registry
_registry = ProviderRegistry[ImageGenerationConfig, ProviderHandler]()

# Built-in handlers
BUILTIN_HANDLERS = {
    "openai": lambda: openai.OpenAIProviderHandler(),
    "fal": lambda: fal.FalProviderHandler(),
    "replicate": lambda: replicate.ReplicateProviderHandler(),
    "gemini": lambda: gemini.GeminiProviderHandler(),
    "mock": lambda: MockProviderHandler(),
}

def _get_handler(config: ImageGenerationConfig) -> ProviderHandler:
    """Get provider handler from registry."""
    return _registry.get_handler(config, BUILTIN_HANDLERS)

# Orchestrator
_orchestrator = ExecutionOrchestrator[
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageGenerationUpdate,
](
    domain="image",
    handler_getter=_get_handler,
)

# Public API
async def generate_image_async(
    config: ImageGenerationConfig,
    request: ImageGenerationRequest,
    on_progress=None,
) -> ImageGenerationResponse:
    """Generate image asynchronously with fallback support."""
    return await _orchestrator.execute_with_fallback_async(
        config, request, on_progress
    )

def generate_image(
    config: ImageGenerationConfig,
    request: ImageGenerationRequest,
    on_progress=None,
) -> ImageGenerationResponse:
    """Generate image synchronously with fallback support."""
    return _orchestrator.execute_with_fallback(
        config, request, on_progress
    )

def register_provider(provider_name: str, handler: ProviderHandler) -> None:
    """Register custom image provider."""
    _registry.register_provider(provider_name, handler)

def register_provider_field_mapping(
    provider_name: str,
    field_mappings: dict[str, dict[str, FieldMapper]],
) -> None:
    """Register field mappings for custom provider."""
    # Store in provider-specific registry
    ...
```

---

## 3. Shared vs Domain-Specific Code Matrix

| Component | Location | Reuse Strategy |
|-----------|----------|----------------|
| **Exceptions** | `common/exceptions.py` | ✅ Full reuse |
| **Orchestrator** | `common/orchestrator.py` | ✅ Generic with type parameters |
| **Registry** | `common/registry.py` | ✅ Generic with callbacks |
| **Logging** | `logging.py` | ✅ Full reuse |
| **Media utils** | `common/utils.py` | ✅ Download, data URL conversion |
| **Field mapper framework** | `common/field_mappers.py` | ✅ Core framework reused |
| **Request/Response models** | `video/models.py`, `image/models.py` | ❌ Domain-specific |
| **Field mappers (specific)** | `video/providers/`, `image/providers/` | ❌ Domain-specific (duration vs size/quality) |
| **Provider implementations** | `video/providers/`, `image/providers/` | ❌ Domain-specific (different APIs) |
| **Mock system** | `video/mock.py`, `image/mock.py` | ⚠️ Adapt pattern, different media types |
| **Validation** | `video/utils.py`, `image/utils.py` | ❌ Domain-specific (duration vs size) |

---

## 4. Implementation Phases

### Phase 1: Extract Common Infrastructure (1-2 days)

**Tasks:**
1. Create `common/` directory
2. Extract and refactor:
   - ✅ `common/exceptions.py`
   - ✅ `common/orchestrator.py`
   - ✅ `common/registry.py`
   - ✅ `common/utils.py`
   - ✅ `common/field_mappers.py`
3. Update `video/` to use common modules
4. Run all video tests to ensure no regressions
5. Update imports, maintain backward compatibility

**Success Criteria:**
- All video tests pass
- No breaking changes to public API
- Clear separation of concerns

### Phase 2: Create Image Models and Core (1 day)

**Tasks:**
1. Create `image/` directory structure
2. Implement:
   - ✅ `image/models.py` (Config, Request, Response, Update, Protocol)
   - ✅ `image/exceptions.py` (import from common)
   - ✅ `image/orchestrator.py` (use common)
   - ✅ `image/registry.py` (use common)
   - ✅ `image/utils.py` (image-specific validation)
   - ✅ `image/api.py` (public API)
3. Write unit tests for models

**Success Criteria:**
- Image models fully defined
- Orchestrator and registry instantiated for images
- Basic API structure in place

### Phase 3: Implement Image Providers (3-4 days)

**Tasks:**
1. Implement `image/providers/field_mappers.py`
   - Size, quality, style mappers
   - Image list mappers
2. Implement providers:
   - ✅ `image/providers/openai.py` (DALL-E 2/3)
   - ✅ `image/providers/fal.py` (Flux Pro, SDXL, etc.)
   - ✅ `image/providers/replicate.py` (SDXL, Flux, etc.)
   - ✅ `image/providers/gemini.py` (Imagen 3)
3. Write unit tests for each provider
   - Client caching
   - Field mapping
   - Error handling
   - Request/response conversion

**Success Criteria:**
- All 4 providers functional
- Comprehensive unit test coverage
- Field mapper registries defined

### Phase 4: Implement Mock System (1 day)

**Tasks:**
1. Create `image/mock.py`
   - Adapt video mock pattern
   - Image library (sample images)
   - Weighted response system
2. Write unit tests for mock provider

**Success Criteria:**
- Mock provider works for all image models
- Supports testing without API keys

### Phase 5: E2E Testing (2 days)

**Tasks:**
1. Update `tests/conftest.py` for image support
2. Write E2E tests for each provider:
   - ✅ `tests/e2e/image/test_openai.py`
   - ✅ `tests/e2e/image/test_fal.py`
   - ✅ `tests/e2e/image/test_replicate.py`
   - ✅ `tests/e2e/image/test_gemini.py`
3. Test fallback chains
4. Test progress tracking

**Success Criteria:**
- E2E tests pass with real API calls
- Fallback chain validated
- Progress callbacks working

### Phase 6: Documentation and Examples (1 day)

**Tasks:**
1. Create `image/README.md`
2. Update main `README.md`
3. Add example scripts:
   - Basic text-to-image
   - Image-to-image
   - Fallback chain example
4. Update `CLAUDE.md` with image gateway patterns

**Success Criteria:**
- Comprehensive documentation
- Working examples
- Clear migration guide from video patterns

---

## 5. Provider-Specific Implementation Details

### 5.1 Fal Image Provider

**Text-to-Image Models (Latest 2025):**

| Model ID | Description | Speed | Key Features |
|----------|-------------|-------|--------------|
| `fal-ai/flux-2/pro` | **FLUX.2 Pro** - Latest flagship (Nov 2025) | Fast | 32B params, multi-reference (up to 10 images), 4MP output |
| `fal-ai/flux-2/flex` | **FLUX.2 Flex** - Tunable parameters | Fast | Adjustable steps/guidance for latency-quality tradeoffs |
| `fal-ai/flux-2/dev` | **FLUX.2 Dev** - Open weights | Medium | 32B params, combined gen+edit, non-commercial |
| `fal-ai/flux-pro/v1.1-ultra` | Flux 1.1 Pro Ultra | Fast | 4MP (4x resolution), $0.06/image |
| `fal-ai/flux-pro/v1.1-raw` | Flux 1.1 Pro Raw | Fast | Candid photography aesthetic |
| `fal-ai/flux-pro/v1.1` | Flux 1.1 Pro | Fast | Production quality |
| `fal-ai/flux/schnell` | Flux Schnell | Very Fast | 4-step, best speed/quality ratio |
| `fal-ai/z-image-turbo` | **Z-Image-Turbo** (Alibaba) | Very Fast | Sub-second inference, 6B params, bilingual text |
| `fal-ai/recraft-v3` | Recraft V3 | Fast | SVG output, brand styles |
| `fal-ai/ideogram/v3` | Ideogram V3 | Fast | Best text-in-image rendering |

**Image-to-Image Models:**

| Model ID | Description | Capabilities |
|----------|-------------|--------------|
| `fal-ai/flux-2/dev` | FLUX.2 Dev img2img | Combined gen+edit in single model |
| `fal-ai/flux-pro/v1.1/redux` | Flux Pro Redux | Image variations |
| `fal-ai/z-image-edit` | Z-Image-Edit (Alibaba) | Instruction-following edits |
| `fal-ai/creative-upscaler` | Creative Upscaler | 4x upscaling with enhancement |
| `fal-ai/clarity-upscaler` | Clarity Upscaler | Detail preservation |
| `fal-ai/inpaint` | Inpainting | Mask-based editing |
| `fal-ai/remove-background` | Background Removal | Alpha channel output |

**Polling Strategy:**
- Most Fal image models are **instant** (no polling)
- FLUX.2 models may use async pattern for multi-reference inputs
- Event-based polling via `iter_events()` when needed

**Field Mapping:**
```python
FAL_IMAGE_MODEL_REGISTRY = {
    "fal-ai/flux-2/pro": {
        "prompt": required,
        "image_size": ["square", "square_hd", "portrait_4_3", "portrait_16_9",
                       "landscape_4_3", "landscape_16_9"],
        "num_images": 1-4,
        "seed": optional,
        "guidance_scale": 1.0-20.0,
        "num_inference_steps": 1-50,
        "enable_safety_checker": bool,
        "reference_images": list[str],  # Up to 10 images for multi-reference
    },
    "fal-ai/flux-pro/v1.1-ultra": {
        "prompt": required,
        "aspect_ratio": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"],
        "raw": bool,  # Enable raw mode for natural aesthetic
        "seed": optional,
        "safety_tolerance": 1-6,
        "output_format": ["jpeg", "png"],
    },
    "fal-ai/z-image-turbo": {
        "prompt": required,
        "negative_prompt": optional,
        "image_size": ["square", "portrait_4_3", "landscape_4_3", "landscape_16_9"],
        "num_inference_steps": 8,  # Distilled, only 8 NFEs needed
        "seed": optional,
        "enable_safety_checker": bool,
    },
    "fal-ai/recraft-v3": {
        "prompt": required,
        "style": ["realistic_image", "digital_illustration", "vector_illustration",
                  "icon", "logo"],
        "size": ["1024x1024", "1365x1024", "1024x1365", ...],
        "colors": list[str],  # Brand colors
    },
    "fal-ai/ideogram/v3": {
        "prompt": required,
        "aspect_ratio": ["1:1", "16:9", "9:16", "4:3", "3:4"],
        "style_type": ["Auto", "General", "Realistic", "Design", "3D", "Anime"],
        "magic_prompt_option": ["Auto", "On", "Off"],
    },
}
```

### 5.2 OpenAI Image Provider

**Supported Models (Latest Dec 2025):**

| Model ID | Description | Capabilities | Status |
|----------|-------------|--------------|--------|
| `gpt-image-1.5` | **GPT Image 1.5** - Latest (Dec 16, 2025) | Text-to-image, editing, 4x faster | **CURRENT** |
| `gpt-image-1` | GPT Image 1 - Native multimodal | Text-to-image, editing | Stable |
| `dall-e-3` | DALL-E 3 | Text-to-image only | Legacy |
| `dall-e-2` | DALL-E 2 | Text-to-image, edit, variation | Legacy |

**GPT-Image-1.5 Key Features:**
- **4x faster** generation (5-15 seconds vs 20-60 seconds)
- **20% lower pricing** than GPT-Image-1
- **Native multimodal** architecture (unified text+image processing)
- **Better editing** - precise localized edits, identity preservation
- **Improved text rendering** - crisp lettering
- **Complex compositions** - infographics, diagrams, multi-panel layouts

**API Endpoints:**

| Operation | Endpoint | Models Supported |
|-----------|----------|------------------|
| Text-to-Image | `POST /v1/images/generations` | gpt-image-1.5, gpt-image-1, dall-e-3, dall-e-2 |
| Image Editing | `POST /v1/images/edits` | gpt-image-1.5, gpt-image-1, dall-e-2 |
| Variations | `POST /v1/images/variations` | dall-e-2 only |

**Polling Strategy:**
- **No polling** - Synchronous API
- Response immediate after generation completes

**Field Mapping:**
```python
OPENAI_MODEL_REGISTRY = {
    "gpt-image-1.5": {
        "prompt": required,
        "size": ["1024x1024", "1024x1792", "1792x1024", "auto"],
        "quality": ["low", "medium", "high", "auto"],  # Flexible quality-latency tradeoff
        "n": 1-4,
        "output_format": ["png", "jpeg", "webp"],
        "background": ["transparent", "opaque", "auto"],
        # Editing parameters
        "image": file,  # For edits - input image
        "mask": file,   # For edits - optional mask
    },
    "gpt-image-1": {
        "prompt": required,
        "size": ["1024x1024", "1024x1792", "1792x1024", "auto"],
        "quality": ["low", "medium", "high", "auto"],
        "n": 1-4,
        "output_format": ["png", "jpeg", "webp"],
        "background": ["transparent", "opaque", "auto"],
    },
    "dall-e-3": {
        "prompt": required,
        "size": ["1024x1024", "1024x1792", "1792x1024"],
        "quality": ["standard", "hd"],
        "style": ["natural", "vivid"],
        "n": 1,  # Only n=1 supported
        "revised_prompt": returns,  # Response includes revised prompt
    },
    "dall-e-2": {
        "prompt": required,
        "size": ["256x256", "512x512", "1024x1024"],
        "n": 1-10,
        # No quality/style parameters
    },
}
```

**Key Differences:**
- `gpt-image-1.5`: Latest, fastest, best for production, native multimodal architecture
- `gpt-image-1`: Previous flagship, stable, still excellent quality
- `dall-e-3`: Good prompt understanding, auto-revises prompts
- `dall-e-2`: Legacy, only model supporting image variations

### 5.3 Replicate Image Provider

**Top Text-to-Image Models (Latest 2025):**

| Model ID | Description | Speed | Notes |
|----------|-------------|-------|-------|
| `black-forest-labs/flux.2-pro` | **FLUX.2 Pro** - Latest flagship | Fast | 32B params, multi-reference |
| `black-forest-labs/flux.2-dev` | **FLUX.2 Dev** - Open weights | Medium | Combined gen+edit |
| `black-forest-labs/flux-1.1-pro-ultra` | Flux 1.1 Pro Ultra | Fast | 4MP resolution |
| `black-forest-labs/flux-1.1-pro` | Flux 1.1 Pro | Fast | Production quality |
| `black-forest-labs/flux-schnell` | Flux Schnell | Very Fast | Best speed/quality |
| `tongyi-mai/z-image-turbo` | **Z-Image-Turbo** (Alibaba) | Very Fast | #1 open source on benchmarks |
| `stability-ai/stable-diffusion-3.5-large` | SD 3.5 Large | Medium | Open weights |
| `stability-ai/stable-diffusion-3.5-large-turbo` | SD 3.5 Turbo | Fast | Speed optimized |

**Image-to-Image Models:**

| Model ID | Description |
|----------|-------------|
| `black-forest-labs/flux.2-dev` | FLUX.2 Dev img2img (combined model) |
| `tongyi-mai/z-image-edit` | Z-Image-Edit (instruction-following) |
| `black-forest-labs/flux-dev-controlnet` | Flux ControlNet |
| `stability-ai/stable-diffusion-inpainting` | Inpainting |
| `nightmareai/real-esrgan` | 4x Upscaling |
| `cjwbw/rembg` | Background removal |

**Polling Strategy:**
- **Manual polling loop** (same as Replicate video)
- Poll `predictions.get()` until status = `succeeded`
- Use `config.poll_interval` and `max_poll_attempts`
- Typical statuses: `starting` → `processing` → `succeeded`/`failed`

**Field Mapping:**
```python
REPLICATE_MODEL_REGISTRY = {
    "black-forest-labs/flux.2-pro": {
        "prompt": required,
        "aspect_ratio": ["1:1", "16:9", "9:16", "21:9", "9:21", "4:3", "3:4", "4:5", "5:4", "3:2", "2:3"],
        "output_format": ["webp", "jpg", "png"],
        "output_quality": 1-100,
        "safety_tolerance": 1-6,
        "reference_images": list[str],  # Up to 10 for multi-reference
        "num_inference_steps": 1-50,
    },
    "black-forest-labs/flux-1.1-pro-ultra": {
        "prompt": required,
        "aspect_ratio": ["1:1", "16:9", "9:16", "21:9", "9:21", "4:3", "3:4", "4:5", "5:4", "3:2", "2:3"],
        "output_format": ["webp", "jpg", "png"],
        "output_quality": 1-100,
        "raw": bool,  # Enable raw mode
        "safety_tolerance": 1-6,
    },
    "tongyi-mai/z-image-turbo": {
        "prompt": required,
        "negative_prompt": optional,
        "aspect_ratio": ["1:1", "16:9", "9:16", "4:3", "3:4"],
        "num_inference_steps": 8,  # Only 8 NFEs needed
        "seed": optional,
        "output_format": ["webp", "jpg", "png"],
    },
    "stability-ai/stable-diffusion-3.5-large": {
        "prompt": required,
        "negative_prompt": optional,
        "aspect_ratio": ["1:1", "16:9", "21:9", "3:2", "2:3", "4:5", "5:4", "9:16", "9:21"],
        "cfg_scale": 1.0-20.0,
        "num_inference_steps": 1-50,
        "seed": optional,
        "output_format": ["webp", "jpg", "png"],
    },
}
```

### 5.4 Google Image Provider (Gemini + Imagen)

**Supported Models (Latest 2025):**

| Model ID | Description | Speed | Status |
|----------|-------------|-------|--------|
| `gemini-2.5-flash-image-preview` | **"Nano Banana"** - SOTA image gen (Aug 2025) | Fast | **#1 on LMArena** |
| `gemini-3-pro-image-preview` | **"Nano Banana Pro"** - Professional | Medium | Advanced reasoning |
| `gemini-2.0-flash-exp` | Gemini 2.0 Flash Native Image | Fast | Experimental |
| `imagen-3.0-generate-002` | Imagen 3 Latest | Medium | Production |
| `imagen-3.0-fast-generate-001` | Imagen 3 Fast | Fast | Speed optimized |

**"Nano Banana" (Gemini 2.5 Flash Image) Key Features:**
- **#1 on LMArena** Text-to-Image and Image Edit leaderboards (Aug 2025)
- **Native multimodal** - image gen inside same transformer (not separate diffusion)
- **Multi-turn editing** - conversational image refinement
- **Character consistency** - maintains identity across edits
- **World knowledge** - leverages Gemini's reasoning for realistic details
- **Better text rendering** - improved long text sequences
- **Image fusion** - combine multiple input images
- Pricing: ~$0.039 per image (1290 output tokens at $30/1M tokens)

**"Nano Banana Pro" (Gemini 3 Pro Image):**
- Advanced "Thinking" for complex instructions
- High-fidelity text rendering
- Professional asset production

**SDK Access:**
- Via `google-genai` SDK (same as Veo 3.1 video)
- Via Vertex AI (`google-cloud-aiplatform`)
- Via Google AI Studio

**Polling Strategy:**
- **No polling** - Synchronous API for most models
- Native image output integrated into response

**Field Mapping:**
```python
GOOGLE_IMAGE_MODEL_REGISTRY = {
    "gemini-2.5-flash-image-preview": {  # "Nano Banana"
        "prompt": required,
        "aspect_ratio": ["1:1", "3:4", "4:3", "9:16", "16:9"],
        "number_of_images": 1-4,
        "safety_filter_level": ["block_low_and_above", "block_medium_and_above",
                                 "block_only_high", "block_none"],
        "person_generation": ["allow_all", "allow_adult", "dont_allow"],
        # Editing parameters
        "input_images": list[file],  # For multi-image fusion/editing
        "edit_mode": bool,  # Enable conversational editing
    },
    "gemini-3-pro-image-preview": {  # "Nano Banana Pro"
        "prompt": required,
        "aspect_ratio": ["1:1", "3:4", "4:3", "9:16", "16:9"],
        "number_of_images": 1-4,
        "safety_filter_level": [...],
        "person_generation": [...],
        "thinking_mode": bool,  # Enable advanced reasoning
    },
    "imagen-3.0-generate-002": {
        "prompt": required,
        "negative_prompt": optional,
        "aspect_ratio": ["1:1", "3:4", "4:3", "9:16", "16:9"],
        "number_of_images": 1-4,
        "safety_filter_level": ["block_low_and_above", "block_medium_and_above",
                                 "block_only_high", "block_none"],
        "person_generation": ["allow_all", "allow_adult", "dont_allow"],
        "language": "en",  # Prompt language hint
        "output_mime_type": ["image/png", "image/jpeg"],
        "output_compression_quality": 1-100,  # For JPEG
    },
}
```

**Key Differences:**
- **Nano Banana**: Best for speed + quality, conversational editing, multi-turn refinement
- **Nano Banana Pro**: Best for professional/complex compositions requiring reasoning
- **Imagen 3**: Production-stable, photorealistic, explicit safety controls

### 5.5 Stability AI Direct Provider (NEW)

**Supported Models:**

| Model ID | Description | Tier |
|----------|-------------|------|
| `stable-diffusion-3.5-large` | SD 3.5 Large | Premium |
| `stable-diffusion-3.5-large-turbo` | SD 3.5 Turbo | Standard |
| `stable-diffusion-3.5-medium` | SD 3.5 Medium | Standard |
| `stable-image-core` | Stable Image Core | Standard |
| `stable-image-ultra` | Stable Image Ultra | Premium |

**API Access:**
- Base URL: `https://api.stability.ai`
- Authentication: Bearer token

**Polling Strategy:**
- **No polling** - Synchronous API
- Direct REST endpoints

**Field Mapping:**
```python
STABILITY_MODEL_REGISTRY = {
    "stable-diffusion-3.5-large": {
        "prompt": required,
        "negative_prompt": optional,
        "aspect_ratio": ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
        "seed": 0-4294967294,
        "output_format": ["png", "jpeg", "webp"],
        "cfg_scale": 1.0-35.0,
        "steps": 1-50,
    },
    "stable-image-ultra": {
        "prompt": required,
        "negative_prompt": optional,
        "aspect_ratio": ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
        "seed": optional,
        "output_format": ["png", "jpeg", "webp"],
        # Automatic quality optimization
    },
}
```

### 5.6 Model Comparison Matrix (Updated Jan 2026)

| Provider/Model | Best For | Speed | Quality | Cost | Batch | Key Advantage |
|----------------|----------|-------|---------|------|-------|---------------|
| **FLUX.2 Pro** (Fal/Replicate) | Production | Fast | Excellent | $$ | 1-4 | Multi-reference (10 images) |
| **FLUX.2 Dev** (Open) | Customization | Medium | Excellent | $ | 1-4 | Combined gen+edit |
| **Flux 1.1 Pro Ultra** | High-res | Fast | Excellent | $$ | 1-4 | 4MP output, raw mode |
| **Z-Image-Turbo** (Alibaba) | Speed | Very Fast | Very Good | $ | 1-4 | Sub-second, bilingual text |
| **GPT-Image-1.5** (OpenAI) | Production | Fast | Excellent | $$$ | 1-4 | Native multimodal, editing |
| **Nano Banana** (Google) | All-rounder | Fast | Excellent | $$ | 1-4 | #1 LMArena, multi-turn edit |
| **Nano Banana Pro** (Google) | Professional | Medium | Best | $$$ | 1-4 | Advanced reasoning |
| **Recraft V3** (Fal) | Design/SVG | Fast | Excellent | $$ | 1 | Vector output, brand styles |
| **Ideogram V3** (Fal) | Text in images | Fast | Excellent | $$ | 1-4 | Best text rendering |
| **SD 3.5 Large** (Stability) | Open source | Medium | Very Good | $$ | 1 | Open weights |
| **Stable Image Ultra** | Premium | Slow | Best | $$$$ | 1 | Maximum quality |

### 5.7 Image-to-Image Capabilities Summary

| Provider | Inpainting | Outpainting | Variations | Upscaling | ControlNet | Multi-turn Edit |
|----------|------------|-------------|------------|-----------|------------|-----------------|
| **Fal** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (FLUX.2) |
| **OpenAI** | ✅ (GPT-1.5/1) | ✅ | ✅ | ❌ | ❌ | ✅ (GPT-1.5) |
| **Replicate** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (FLUX.2) |
| **Google** | ✅ (Nano Banana) | ✅ | ✅ | ❌ | ❌ | ✅ (native) |
| **Stability** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Alibaba (Z-Image)** | ✅ (Z-Image-Edit) | ❌ | ❌ | ❌ | ❌ | ❌ |

### 5.8 New Provider: Alibaba Z-Image

**Supported Models:**

| Model ID | Description | Speed | Notes |
|----------|-------------|-------|-------|
| `z-image-turbo` | Z-Image-Turbo | Very Fast | #1 open-source on Artificial Analysis, sub-second |
| `z-image-base` | Z-Image-Base | Medium | Foundation model for fine-tuning |
| `z-image-edit` | Z-Image-Edit | Fast | Instruction-following edits |

**Key Features:**
- **6B parameter** Scalable Single-Stream Diffusion Transformer (S3-DiT)
- **8 NFEs only** - distilled for efficiency
- **Sub-second inference** on H800, fits in 16GB VRAM consumer devices
- **Bilingual text rendering** (English + Chinese)
- **#8 overall on Artificial Analysis** Text-to-Image leaderboard (Dec 2025)
- **#1 open-source model** on benchmarks
- Available on HuggingFace, ModelScope, and via Fal/Replicate

**SDK Access:**
- Via HuggingFace diffusers library
- Via Fal.ai API
- Via Replicate API

**Field Mapping:**
```python
ZIMAGE_MODEL_REGISTRY = {
    "z-image-turbo": {
        "prompt": required,
        "negative_prompt": optional,
        "width": 512-2048,
        "height": 512-2048,
        "num_inference_steps": 8,  # Distilled - only 8 needed
        "guidance_scale": 1.0-20.0,
        "seed": optional,
        "output_format": ["png", "jpeg", "webp"],
    },
    "z-image-edit": {
        "prompt": required,  # Edit instruction
        "image": file,  # Input image
        "strength": 0.0-1.0,  # Edit strength
        "seed": optional,
    },
}
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Coverage Goals:**
- ✅ Client caching (sync/async strategies)
- ✅ Field mapper application
- ✅ Error mapping (400, 403, 429, 5xx)
- ✅ Request/response conversion
- ✅ Registry lookup (exact match, prefix match, fallback)
- ✅ Mock provider behavior

**Test Organization:**
```
tests/unit/
├── common/
│   ├── test_exceptions.py
│   ├── test_orchestrator.py
│   ├── test_registry.py
│   ├── test_field_mappers.py
│   └── test_utils.py
├── image/
│   ├── test_models.py
│   ├── test_api.py
│   ├── test_mock.py
│   └── providers/
│       ├── test_field_mappers.py
│       ├── test_openai.py
│       ├── test_fal.py
│       ├── test_replicate.py
│       └── test_gemini.py
└── video/
    └── ... (existing)
```

### 6.2 E2E Tests

**Coverage Goals:**
- ✅ Async generation with progress tracking
- ✅ Sync generation
- ✅ Error handling (invalid params, rate limits)
- ✅ Different model configurations
- ✅ Fallback chain execution
- ✅ Response structure validation

**Test Organization:**
```
tests/e2e/
├── image/
│   ├── test_openai.py
│   ├── test_fal.py
│   ├── test_replicate.py
│   ├── test_gemini.py
│   └── test_fallback.py
└── video/
    └── ... (existing)
```

### 6.3 Test Fixtures

**Shared Fixtures (conftest.py):**
```python
# API key fixtures
@pytest.fixture(scope="module")
def openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key

@pytest.fixture(scope="module")
def fal_api_key():
    api_key = os.getenv("FAL_KEY")
    if not api_key:
        pytest.skip("FAL_KEY not set")
    return api_key

# Config fixtures
@pytest.fixture
def openai_image_config(openai_api_key):
    return ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key=openai_api_key,
        timeout=120,
    )

# Request fixtures
@pytest.fixture
def basic_image_request():
    return ImageGenerationRequest(
        prompt="A serene mountain landscape at sunset",
        size="1024x1024",
        quality="standard",
    )
```

---

## 7. Backwards Compatibility and Migration

### 7.1 Video Gateway Migration

**Changes to Video Gateway:**
1. Import exceptions from `common.exceptions` instead of `video.exceptions`
2. Use `common.orchestrator.ExecutionOrchestrator` instead of local implementation
3. Use `common.registry.ProviderRegistry` instead of local implementation
4. Import utils from `common.utils` where applicable

**Backward Compatibility:**
- Maintain all existing imports as aliases:
  ```python
  # video/exceptions.py
  from common.exceptions import (
      TarashException,
      ValidationError,
      # ... etc
  )

  # Re-export for backward compatibility
  __all__ = ["TarashException", "ValidationError", ...]
  ```

- Public API unchanged
- All tests should pass without modification

### 7.2 Gradual Adoption

**Phased Rollout:**
1. **Phase 1**: Extract common, update video internally (no public API changes)
2. **Phase 2**: Launch image gateway (new functionality)
3. **Phase 3**: Document shared patterns for future domains (audio, etc.)

---

## 8. Future Extensibility

### 8.1 Additional Domains

The refactored architecture supports easy addition of new domains:

**Potential Future Domains:**
- `audio/` - Audio generation (music, speech, effects)
- `3d/` - 3D model generation
- `multimodal/` - Combined video+audio generation

**Pattern:**
1. Create domain directory (e.g., `audio/`)
2. Define domain models (`AudioGenerationConfig`, `AudioGenerationRequest`, etc.)
3. Implement providers using common infrastructure
4. Define domain-specific field mappers
5. Instantiate `ExecutionOrchestrator` and `ProviderRegistry` for domain

### 8.2 Cross-Domain Features

**Shared Infrastructure Benefits:**
- Unified error handling across all domains
- Consistent fallback behavior
- Shared logging and monitoring
- Common testing patterns
- Reusable provider client caching strategies

---

## 9. Risk Assessment and Mitigation

### 9.1 Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **Video regression during refactor** | High | Medium | Comprehensive test coverage, phased rollout |
| **Provider API changes** | Medium | Medium | Mock providers for testing, version pinning |
| **Performance degradation** | Medium | Low | Benchmark before/after, optimize hot paths |
| **Complex field mapping bugs** | Medium | Medium | Extensive unit tests, validation layer |
| **Polling timeout issues** | Low | Medium | Configurable timeouts, clear error messages |

### 9.2 Mitigation Strategies

**1. Comprehensive Testing:**
- 100% unit test coverage for common infrastructure
- E2E tests for all providers
- Regression test suite for video gateway

**2. Incremental Migration:**
- Refactor common first, validate with existing tests
- Launch image gateway as new feature
- Monitor production metrics

**3. Documentation:**
- Clear migration guide
- Examples for each provider
- Troubleshooting guide

**4. Monitoring:**
- Track error rates by provider
- Monitor fallback chain usage
- Alert on unusual patterns

---

## 10. Success Metrics

### 10.1 Technical Metrics

- ✅ **Test Coverage**: >90% for common/, video/, image/
- ✅ **Code Reuse**: >70% of infrastructure shared
- ✅ **Provider Parity**: All 4 image providers functional
- ✅ **Performance**: Image generation <5s overhead vs direct API
- ✅ **Error Rate**: <1% unexpected errors in production

### 10.2 Developer Experience Metrics

- ✅ **Time to Add Provider**: <4 hours for new image provider
- ✅ **Time to Add Model**: <30 minutes for new model in existing provider
- ✅ **Documentation Quality**: All features documented with examples
- ✅ **API Consistency**: Same patterns across video/image

---

## 11. Open Questions and Decisions Needed

### 11.1 Architecture Decisions

1. **Should we support batch image generation?**
   - Some providers (DALL-E 2) support n>1
   - Others only support n=1
   - **Decision**: Support in request model, validate in provider

2. **How to handle image editing (inpainting, outpainting)?**
   - Different API endpoints (OpenAI has separate endpoint)
   - **Decision**: Use `mask_image` field in request, provider-specific handling

3. **Should we support image variation generation?**
   - OpenAI DALL-E 2 supports variations from existing image
   - **Decision**: Add to future roadmap, not MVP

### 11.2 Provider-Specific Questions

1. **Fal**: Which models to prioritize?
   - **Recommendation**: Flux Pro (quality), SDXL (compatibility), Recraft (SVG)

2. **Replicate**: How to handle thousands of community models?
   - **Recommendation**: Document top 10, provide generic fallback mappers

3. **Gemini**: Should we support video generation via Gemini?
   - **Recommendation**: Yes, add to video gateway in separate task

---

## Conclusion

This design plan provides a comprehensive roadmap for implementing the Image Gateway while maximizing code reuse from the Video Gateway. The phased approach minimizes risk, and the shared infrastructure ensures consistency across domains.

**Next Steps:**
1. ✅ Review and approve this plan
2. ✅ Begin Phase 1: Extract common infrastructure
3. ✅ Implement Phase 2-6 sequentially
4. ✅ Launch image gateway with documentation

**Estimated Timeline:** 8-10 days for full implementation

**Key Benefits:**
- 70%+ code reuse from video gateway
- Consistent API across domains
- Extensible for future domains (audio, 3D)
- Comprehensive testing and documentation
