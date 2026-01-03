# Claude Code Guidelines - Tarash Gateway

This document contains guidelines for Claude Code when working on the tarash-gateway package.

## Table of Contents

1. [Overview](#overview)
2. [Provider Architecture](#provider-architecture)
3. [Sync/Async Support](#syncasync-support)
4. [Polling Logic](#polling-logic)
5. [Exception Handling](#exception-handling)
6. [Field Mapper Registry](#field-mapper-registry)
7. [Testing Practices](#testing-practices)
8. [Code Patterns](#code-patterns)

---

## Overview

Tarash Gateway is a unified video generation SDK that provides a consistent interface across multiple AI video providers (Fal, Replicate, OpenAI, Azure OpenAI, Google Veo3). The architecture emphasizes:

- **Provider abstraction**: All providers implement the same `ProviderHandler` protocol
- **Flexible field mapping**: Model-specific parameters are mapped via a registry system
- **Robust error handling**: Structured exception hierarchy with detailed context
- **Dual sync/async support**: All operations available in both modes
- **Progress tracking**: Real-time progress updates via callbacks

**Key Files:**
- [video/api.py](src/tarash/tarash_gateway/video/api.py) - Public API entry point
- [video/models.py](src/tarash/tarash_gateway/video/models.py) - Core data models and protocols
- [video/exceptions.py](src/tarash/tarash_gateway/video/exceptions.py) - Exception hierarchy
- [video/providers/](src/tarash/tarash_gateway/video/providers/) - Provider implementations
- [video/providers/field_mappers.py](src/tarash/tarash_gateway/video/providers/field_mappers.py) - Field mapping framework

---

## Provider Architecture

### ProviderHandler Protocol

All providers must implement the `ProviderHandler` protocol defined in [models.py:225-331](src/tarash/tarash_gateway/video/models.py#L225-L331):

```python
class ProviderHandler(Protocol):
    def _get_client(self, config: VideoGenerationConfig,
                   client_type: Literal["sync", "async"]) -> Any:
        """Get or create provider client (sync or async)."""
        ...

    def _validate_params(self, config, request) -> dict[str, Any]:
        """Validate and transform request parameters."""
        ...

    def _convert_request(self, config, request) -> dict[str, Any]:
        """Convert VideoGenerationRequest to provider-specific format."""
        ...

    def _convert_response(self, config, request, request_id,
                         provider_response) -> VideoGenerationResponse:
        """Convert provider response to normalized format."""
        ...

    async def generate_video_async(self, config, request,
                                  on_progress=None) -> VideoGenerationResponse:
        """Generate video asynchronously."""
        ...

    def generate_video(self, config, request,
                      on_progress=None) -> VideoGenerationResponse:
        """Generate video synchronously."""
        ...
```

### Implementation Pattern

**Key Responsibilities:**

1. **Client Management** (`_get_client`):
   - Create and cache HTTP clients
   - Handle different caching strategies for sync vs async
   - Use cache key: `f"{config.api_key}:{config.base_url or 'default'}"`

2. **Request Conversion** (`_convert_request`):
   - Get model-specific field mappers from registry
   - Apply field mappers to transform request
   - Merge with `extra_params` for manual overrides
   - Log converted request (with sensitive data redaction)

3. **Response Conversion** (`_convert_response`):
   - Normalize provider response to `VideoGenerationResponse`
   - Extract video URL, request ID, status
   - Preserve raw response for debugging

4. **Error Handling** (`_handle_error`):
   - Map provider-specific errors to tarash exceptions
   - Preserve context (provider, model, request_id)
   - Include raw error response

**Example Implementation:**

See [fal.py:236-503](src/tarash/tarash_gateway/video/providers/fal.py#L236-L503) for a complete reference implementation.

---

## Sync/Async Support

### Client Caching Strategies

Different providers use different caching strategies based on their underlying clients:

#### Strategy 1: Sync Cached, Async Fresh (Fal)

**Location:** [fal.py:236-307](src/tarash/tarash_gateway/video/providers/fal.py#L236-L307)

```python
def __init__(self):
    self._sync_client_cache: dict[str, Any] = {}
    # AsyncClient is NOT cached to avoid "Event Loop closed" errors

def _get_client(self, config, client_type):
    cache_key = f"{config.api_key}:{config.base_url or 'default'}"

    if client_type == "async":
        # Create new instance each time
        return fal_client.AsyncClient(
            key=config.api_key,
            base_url=config.base_url
        )
    else:  # sync
        if cache_key not in self._sync_client_cache:
            self._sync_client_cache[cache_key] = fal_client.SyncClient(
                key=config.api_key,
                base_url=config.base_url
            )
        return self._sync_client_cache[cache_key]
```

**Rationale:** Fal's `AsyncClient` cannot be safely reused across event loops. Creating fresh instances prevents "Event Loop closed" errors.

#### Strategy 2: Both Cached (OpenAI)

**Location:** [openai.py:107-157](src/tarash/tarash_gateway/video/providers/openai.py#L107-L157)

```python
def __init__(self):
    self._client_cache: dict[str, Any] = {}

def _get_client(self, config, client_type):
    cache_key = f"{config.api_key}:{config.base_url or 'default'}:{client_type}"

    if cache_key not in self._client_cache:
        if client_type == "async":
            self._client_cache[cache_key] = AsyncOpenAI(...)
        else:
            self._client_cache[cache_key] = OpenAI(...)

    return self._client_cache[cache_key]
```

**Rationale:** OpenAI's clients are designed to be reused across calls. Both sync and async clients can be safely cached.

### Progress Callback Pattern

**Support both sync and async callbacks:**

```python
# In async context
async for event in handler.iter_events(...):
    update = self._process_event(config, request_id, event, start_time)

    if on_progress:
        result = on_progress(update)
        if asyncio.iscoroutine(result):
            await result  # Handle async callback

# In sync context
for event in handler.iter_events(...):
    update = self._process_event(config, request_id, event, start_time)

    if on_progress:
        on_progress(update)  # Direct call for sync callback
```

**Example:** [fal.py:561-599](src/tarash/tarash_gateway/video/providers/fal.py#L561-L599)

---

## Polling Logic

### Event-Based Polling (Fal)

**Location:** [fal.py:505-599](src/tarash/tarash_gateway/video/providers/fal.py#L505-L599)

Fal provides native event streaming. Use `iter_events()` for real-time updates:

```python
async def _process_events_async(self, config, request_id, handler, on_progress=None):
    """Process events asynchronously and return final result."""
    start_time = time.time()

    async for event in handler.iter_events(
        with_logs=True,
        interval=config.poll_interval
    ):
        # Process event and create update
        update = self._process_event(config, request_id, event, start_time)

        # Invoke progress callback
        if on_progress:
            result = on_progress(update)
            if asyncio.iscoroutine(result):
                await result

    # Return final result
    return await handler.get()
```

**Key Points:**
- Use `with_logs=True` to capture log messages
- Respect `config.poll_interval` for polling frequency
- Process each event to create `VideoGenerationUpdate`
- Final result obtained via `handler.get()`

### Manual Polling Loop (Replicate)

**Location:** [replicate.py:449-587](src/tarash/tarash_gateway/video/providers/replicate.py#L449-L587)

Replicate requires manual status polling:

```python
max_attempts = config.max_poll_attempts
poll_interval = config.poll_interval

for _ in range(max_attempts):
    # Reload prediction status
    prediction = await async_client.predictions.get(prediction_id=prediction.id)

    # Send progress update
    if on_progress:
        update = parse_replicate_status(prediction)
        result = on_progress(update)
        if asyncio.iscoroutine(result):
            await result

    # Log progress
    log_info("Progress status update",
             status=prediction.status,
             request_id=request_id,
             provider="replicate")

    # Check completion
    if prediction.status == "succeeded":
        return self._convert_response(...)
    elif prediction.status in ("failed", "canceled"):
        raise GenerationFailedError(...)

    # Wait before next poll
    await asyncio.sleep(poll_interval)

# Max attempts reached
raise GenerationFailedError("Max poll attempts exceeded")
```

**Key Points:**
- Use `config.max_poll_attempts` to prevent infinite loops
- Check terminal states: `succeeded`, `failed`, `canceled`
- Use `asyncio.sleep()` for async, `time.sleep()` for sync
- Raise `GenerationFailedError` on timeout

---

## Exception Handling

### Exception Hierarchy

**Location:** [exceptions.py:1-162](src/tarash/tarash_gateway/video/exceptions.py#L1-L162)

All exceptions inherit from `TarashException`:

```python
class TarashException(Exception):
    """Base exception for all Tarash video generation errors."""

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
```

**Specific Exceptions:**

- **`ValidationError`** (Lines 38-41): Input validation failed (400-level errors)
- **`ContentModerationError`** (Lines 44-47): Content policy violation (403-level errors)
- **`HTTPError`** (Lines 50-63): HTTP errors with status code
- **`GenerationFailedError`** (Lines 66-69): Generation failed (timeouts, cancellations, server errors)

**Usage Pattern:**

```python
# Validate input
if not prompt:
    raise ValidationError(
        "Prompt is required",
        provider=config.provider,
        model=config.model
    )

# Handle content moderation
if response.status_code == 403:
    raise ContentModerationError(
        "Content policy violation",
        provider=config.provider,
        model=config.model,
        request_id=request_id,
        raw_response=response.json()
    )

# Handle generation failures
if prediction.status == "failed":
    raise GenerationFailedError(
        f"Generation failed: {prediction.error}",
        provider=config.provider,
        model=config.model,
        request_id=request_id,
        raw_response=asdict(prediction)
    )
```

### Error Handling Decorator

**Location:** [exceptions.py:72-162](src/tarash/tarash_gateway/video/exceptions.py#L72-L162)

Use `@handle_video_generation_errors` to wrap provider methods:

```python
@handle_video_generation_errors
async def generate_video_async(self, config, request, on_progress=None):
    # Implementation
    ...
```

**Behavior:**
- Lets `TarashException` and `PydanticValidationError` propagate unchanged
- Wraps unknown exceptions in `TarashException` with full traceback
- Preserves context (provider, model, raw_response)
- Logs full error details for debugging

**Example:** [fal.py:353-421](src/tarash/tarash_gateway/video/providers/fal.py#L353-L421)

### Provider-Specific Error Mapping

Each provider should implement `_handle_error` to map provider errors to tarash exceptions:

```python
def _handle_error(
    self,
    config: VideoGenerationConfig,
    request: VideoGenerationRequest,
    request_id: str,
    ex: Exception,
) -> TarashException:
    """Map provider-specific errors to TarashException."""

    if isinstance(ex, FalClientHTTPError):
        if ex.status_code == 400:
            return ValidationError(
                f"Invalid request: {ex.message}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": ex.message, "status_code": ex.status_code},
            )
        elif ex.status_code == 403:
            return ContentModerationError(
                f"Content moderation failed: {ex.message}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": ex.message, "status_code": ex.status_code},
            )
        else:
            return HTTPError(
                f"HTTP error: {ex.message}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                status_code=ex.status_code,
                raw_response={"error": ex.message, "status_code": ex.status_code},
            )

    # Fallback to generic TarashException
    return TarashException(
        f"Unknown error: {str(ex)}",
        provider=config.provider,
        model=config.model,
        request_id=request_id,
        raw_response={"error": str(ex), "error_type": type(ex).__name__},
    )
```

**Example:** [fal.py:423-503](src/tarash/tarash_gateway/video/providers/fal.py#L423-L503)

---

## Field Mapper Registry

### Overview

Field mappers translate the generic `VideoGenerationRequest` to provider-specific API formats. This allows the same request to work across different providers with different parameter names and formats.

**Key Files:**
- [field_mappers.py](src/tarash/tarash_gateway/video/providers/field_mappers.py) - Framework and utilities
- [fal.py:57-203](src/tarash/tarash_gateway/video/providers/fal.py#L57-L203) - Fal registry
- [replicate.py:42-200](src/tarash/tarash_gateway/video/providers/replicate.py#L42-L200) - Replicate registry

### FieldMapper Framework

**Location:** [field_mappers.py:20-80](src/tarash/tarash_gateway/video/providers/field_mappers.py#L20-L80)

```python
@dataclass
class FieldMapper:
    """Maps a VideoGenerationRequest field to an API field with conversion."""

    source_field: str  # Field name in VideoGenerationRequest
    converter: Callable[[VideoGenerationRequest, Any], Any]  # Conversion function
    required: bool = False  # Whether field is required

def apply_field_mappers(
    field_mappers: dict[str, FieldMapper],
    request: VideoGenerationRequest,
) -> dict[str, Any]:
    """Apply field mappers to convert VideoGenerationRequest to API format."""
    result = {}

    for api_field_name, mapper in field_mappers.items():
        # Extract source value
        source_value = getattr(request, mapper.source_field, None)

        # Validate required field
        if mapper.required and source_value is None:
            raise ValueError(f"Required field '{mapper.source_field}' is missing")

        # Apply converter
        converted_value = mapper.converter(request, source_value)

        # Validate required conversion result
        if mapper.required and converted_value is None:
            raise ValueError(f"Required field '{api_field_name}' cannot be None")

        # Exclude None and empty collections
        if converted_value is not None and converted_value != [] and converted_value != {}:
            result[api_field_name] = converted_value

    return result
```

### Pre-Built Field Mapper Utilities

**Location:** [field_mappers.py:86-284](src/tarash/tarash_gateway/video/providers/field_mappers.py#L86-L284)

#### 1. `duration_field_mapper(field_type, allowed_values, provider, model)`

Converts duration to provider-specific format with validation:

```python
# String format (e.g., "6s", "10s")
duration_field_mapper(
    field_type="str",
    allowed_values=["6s", "10s"],
    provider="fal",
    model="minimax"
)

# Integer format (e.g., 4, 8, 12)
duration_field_mapper(
    field_type="int",
    allowed_values=[4, 8, 12],
    provider="fal",
    model="sora-2"
)
```

#### 2. `single_image_field_mapper(image_type="start_frame")`

Extracts single image from `image_list` by type:

```python
# Extract start_frame image
"image_url": single_image_field_mapper(image_type="start_frame")

# Extract end_frame image
"tail_image_url": single_image_field_mapper(image_type="end_frame")
```

**Validation:** Raises error if multiple images of same type exist.

#### 3. `image_list_field_mapper()`

Converts all images in `image_list` to URLs:

```python
"images": image_list_field_mapper()  # Returns list of URLs
```

#### 4. `passthrough_field_mapper(source_field, required=False)`

Returns value unchanged:

```python
"prompt": passthrough_field_mapper("prompt", required=True)
"seed": passthrough_field_mapper("seed")
```

#### 5. `extra_params_field_mapper(extra_param_key_name)`

Extracts value from `extra_params` dict:

```python
"character_orientation": extra_params_field_mapper("character_orientation")
```

#### 6. `video_url_field_mapper()`

Converts `video` field to URL:

```python
"video_url": video_url_field_mapper()
```

### Registry Lookup with Prefix Matching

**Location:** [field_mappers.py:289-342](src/tarash/tarash_gateway/video/providers/field_mappers.py#L289-L342)

The registry supports intelligent lookup with prefix matching:

```python
def get_field_mappers_from_registry(
    model_name: str,
    registry: dict[str, dict[str, FieldMapper]],
    fallback_mappers: dict[str, FieldMapper],
) -> dict[str, FieldMapper]:
    """Get field mappers with prefix matching support."""

    # 1. Try exact match first
    if model_name in registry:
        return registry[model_name]

    # 2. Try prefix matching - find longest matching prefix
    matching_prefix = None
    for registry_key in registry:
        if model_name.startswith(registry_key):
            if matching_prefix is None or len(registry_key) > len(matching_prefix):
                matching_prefix = registry_key

    if matching_prefix:
        return registry[matching_prefix]

    # 3. Fall back to default mappers
    return fallback_mappers
```

**Example:**

```python
# Registry
FAL_MODEL_REGISTRY = {
    "fal-ai/veo3.1": VEO3_FIELD_MAPPERS,      # Longest match
    "fal-ai/veo3": VEO3_FIELD_MAPPERS,
    "fal-ai/minimax": MINIMAX_FIELD_MAPPERS,
}

# Lookups
get_field_mappers("fal-ai/veo3.1")           # Exact match
get_field_mappers("fal-ai/veo3.1/fast")      # Prefix match -> veo3.1
get_field_mappers("fal-ai/veo3/preview")     # Prefix match -> veo3
get_field_mappers("fal-ai/unknown")          # Fallback to generic
```

### Provider-Specific Registries

#### Fal Registry

**Location:** [fal.py:57-203](src/tarash/tarash_gateway/video/providers/fal.py#L57-L203)

```python
# Model-specific mappers
MINIMAX_FIELD_MAPPERS = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper("str", ["6s", "10s"], "fal", "minimax"),
    "image_url": single_image_field_mapper(),
    "prompt_optimizer": passthrough_field_mapper("enhance_prompt"),
}

VEO3_FIELD_MAPPERS = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper("str", ["4s", "6s", "8s"], "fal", "veo3"),
    "image_url": single_image_field_mapper(image_type="start_frame"),
    "first_frame_url": single_image_field_mapper(image_type="first_frame"),
    "last_frame_url": single_image_field_mapper(image_type="last_frame"),
    "video_url": video_url_field_mapper(),
    # ... more fields
}

# Registry with prefix support
FAL_MODEL_REGISTRY = {
    "fal-ai/minimax": MINIMAX_FIELD_MAPPERS,
    "fal-ai/kling-video/v2.6": KLING_VIDEO_V26_FIELD_MAPPERS,
    "fal-ai/veo3.1": VEO3_FIELD_MAPPERS,  # Registered before veo3
    "fal-ai/veo3": VEO3_FIELD_MAPPERS,
    "fal-ai/sora-2": SORA2_FIELD_MAPPERS,
}

def get_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get field mappers for Fal model with prefix matching."""
    return get_field_mappers_from_registry(
        model_name,
        FAL_MODEL_REGISTRY,
        GENERIC_FIELD_MAPPERS  # Fallback
    )
```

#### Replicate Registry

**Location:** [replicate.py:42-200](src/tarash/tarash_gateway/video/providers/replicate.py#L42-L200)

Replicate uses prefix matching extensively due to version hashes in model names:

```python
REPLICATE_MODEL_REGISTRY = {
    "kwaivgi/kling": KLING_V21_FIELD_MAPPERS,
    "minimax/": MINIMAX_FIELD_MAPPERS,      # Prefix match
    "hailuo/": MINIMAX_FIELD_MAPPERS,       # Prefix match
    "luma/": LUMA_FIELD_MAPPERS,            # Prefix match
    "wan-video/": WAN_FIELD_MAPPERS,        # Prefix match
    "google/veo-3": VEO31_FIELD_MAPPERS,
}

def get_replicate_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get field mappers for Replicate model."""
    # Normalize model name by removing version hash
    normalized = model_name.split(":")[0]

    return get_field_mappers_from_registry(
        normalized,
        REPLICATE_MODEL_REGISTRY,
        GENERIC_REPLICATE_FIELD_MAPPERS
    )
```

**Example:** `minimax/video-01:abc123` → normalized to `minimax/video-01` → matches prefix `minimax/`

### Custom Field Mappers

For complex conversions, define custom converter functions:

```python
def _veo3_reference_images_converter(
    request: VideoGenerationRequest, val: Any
) -> list[str] | None:
    """Convert image_list to reference_images, filtering out first/last frames."""
    if not request.image_list:
        return None

    # Filter out first/last frame images
    reference_images = [
        img for img in request.image_list
        if img.type not in ("first_frame", "last_frame")
    ]

    if not reference_images:
        return None

    # Convert to URLs
    return [_convert_to_url(img) for img in reference_images]

# Use in mapper
VEO31_FIELD_MAPPERS = {
    "reference_images": FieldMapper(
        source_field="image_list",
        converter=_veo3_reference_images_converter,
        required=False
    ),
}
```

**Example:** [replicate.py:102-126](src/tarash/tarash_gateway/video/providers/replicate.py#L102-L126)

---

## Testing Practices

### Unit Tests

**Location:** [tests/unit/](tests/unit/)

#### Key Principles

1. **Mock all external dependencies** (HTTP clients, API calls)
2. **Test in isolation** (clear caches between tests)
3. **Validate behavior** (cache hits, error mapping, field conversion)
4. **Use fixtures** for common setup

#### Test Organization

**conftest.py Pattern:**

```python
@pytest.fixture
def mock_sync_client():
    """Patch provider's sync client."""
    mock = MagicMock()
    with patch("tarash.tarash_gateway.video.providers.fal.fal_client.SyncClient",
               return_value=mock):
        yield mock

@pytest.fixture
def mock_async_client():
    """Patch provider's async client."""
    mock = AsyncMock()
    with patch("tarash.tarash_gateway.video.providers.fal.fal_client.AsyncClient",
               return_value=mock):
        yield mock

@pytest.fixture
def base_config():
    """Create test config."""
    return VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="test-api-key",
        timeout=600,
    )

@pytest.fixture
def base_request():
    """Create test request."""
    return VideoGenerationRequest(prompt="Test prompt")
```

#### Testing Client Caching

**Example:** [test_fal.py:81-150](tests/unit/video/providers/test_fal.py#L81-L150)

```python
def test_get_client_creates_and_caches_sync_client(handler, base_config, mock_sync_client):
    """Test that sync clients are cached."""
    handler._sync_client_cache.clear()

    client1 = handler._get_client(base_config, "sync")
    client2 = handler._get_client(base_config, "sync")

    assert client1 is client2  # Same instance (cached)

def test_get_client_creates_new_async_client_each_time(handler, base_config):
    """Test that async clients are NOT cached."""
    with patch("...fal_client.AsyncClient") as mock_constructor:
        mock_constructor.side_effect = [AsyncMock(), AsyncMock()]

        client1 = handler._get_client(base_config, "async")
        client2 = handler._get_client(base_config, "async")

        assert client1 is not client2  # Different instances

def test_get_client_different_api_keys_use_different_cache(handler, base_config):
    """Test that different API keys create separate cache entries."""
    handler._sync_client_cache.clear()

    config2 = base_config.model_copy(update={"api_key": "different-key"})

    client1 = handler._get_client(base_config, "sync")
    client2 = handler._get_client(config2, "sync")

    assert client1 is not client2  # Different cache entries
```

#### Testing Field Mappers

**Example:** [test_field_mappers.py:48-150](tests/unit/video/providers/test_field_mappers.py#L48-L150)

```python
def test_apply_field_mappers_all_fields():
    """Test field mapper application with all fields present."""
    field_mappers = {
        "api_prompt": FieldMapper(
            source_field="prompt",
            converter=lambda req, val: val.upper()
        ),
        "api_seed": FieldMapper(
            source_field="seed",
            converter=lambda req, val: val
        ),
    }

    request = VideoGenerationRequest(prompt="test", seed=42)
    result = apply_field_mappers(field_mappers, request)

    assert result == {"api_prompt": "TEST", "api_seed": 42}

def test_apply_field_mappers_required_field_missing():
    """Test that missing required field raises error."""
    field_mappers = {
        "api_prompt": FieldMapper(
            source_field="prompt",
            converter=lambda req, val: val,
            required=True
        ),
    }

    request = VideoGenerationRequest()  # No prompt

    with pytest.raises(ValueError, match="Required field 'prompt' is missing"):
        apply_field_mappers(field_mappers, request)
```

#### Testing Error Mapping

```python
def test_handle_error_validation_error(handler, base_config, base_request):
    """Test that 400 errors map to ValidationError."""
    ex = FalClientHTTPError("Bad request", status_code=400)

    result = handler._handle_error(base_config, base_request, "req-123", ex)

    assert isinstance(result, ValidationError)
    assert result.provider == "fal"
    assert result.model == "fal-ai/veo3.1"
    assert result.request_id == "req-123"
    assert result.raw_response["status_code"] == 400

def test_handle_error_content_moderation(handler, base_config, base_request):
    """Test that 403 errors map to ContentModerationError."""
    ex = FalClientHTTPError("Content policy violation", status_code=403)

    result = handler._handle_error(base_config, base_request, "req-123", ex)

    assert isinstance(result, ContentModerationError)
```

### E2E Tests

**Location:** [tests/e2e/](tests/e2e/)

#### Key Principles

1. **Test against real APIs** (requires API keys)
2. **Skip gracefully** if API keys not available
3. **Track progress updates** to validate callback system
4. **Validate response structure** and data types
5. **Use realistic parameters** (valid prompts, durations, etc.)

#### Test Setup

**API Key Fixture:**

```python
@pytest.fixture(scope="module")
def fal_api_key():
    """Get Fal API key from environment."""
    api_key = os.getenv("FAL_KEY")
    if not api_key:
        pytest.skip("FAL_KEY environment variable not set")
    return api_key
```

**conftest.py Configuration:**

The test framework automatically:
- Detects when `--e2e` flag is used
- Skips e2e tests unless flag provided
- Checks for required API keys before running
- Auto-marks tests based on directory (`/e2e/` vs `/unit/`)

See [conftest.py:36-84](tests/conftest.py#L36-L84)

#### Comprehensive E2E Test Pattern

**Example:** [test_fal.py:37-100](tests/e2e/test_fal.py#L37-L100)

```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_comprehensive_async_video_generation(fal_api_key):
    """Comprehensive async test with progress tracking."""

    # Track progress updates
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status} - {update.message}")

    # Create config with realistic timeouts
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,             # 10 minutes
        max_poll_attempts=120,   # 120 attempts
        poll_interval=5,         # 5 seconds between polls
    )

    # Create realistic request
    req = VideoGenerationRequest(
        prompt="A serene lake at sunset with mountains in the background",
        duration_seconds=4,
        aspect_ratio="16:9",
        resolution="720p",
        seed=42,
        negative_prompt="blur, low quality",
        generate_audio=True,
        auto_fix=True,
    )

    # Generate video
    print(f"\nGenerating video with model: {config.model}")
    response = await api.generate_video_async(
        config,
        req,
        on_progress=progress_callback
    )

    # Validate response structure
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"
    assert isinstance(response.raw_response, dict)

    # Validate video URL
    assert isinstance(response.video, str)
    assert response.video.startswith("http")

    # Validate progress tracking
    assert len(progress_updates) > 0, "Expected progress updates"
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses, "Expected 'completed' status in updates"

    print(f"✓ Video generated successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Progress updates: {len(progress_updates)}")
```

#### Running E2E Tests

```bash
# Run all e2e tests (requires API keys)
pytest tests/e2e/ --e2e

# Run specific provider
pytest tests/e2e/test_fal.py --e2e

# With verbose logging
pytest tests/e2e/ --e2e --log-cli-level=DEBUG

# Run in parallel (faster)
pytest tests/e2e/ --e2e -n auto
```

#### E2E Test Coverage

Ensure e2e tests cover:
- ✅ Async and sync generation
- ✅ Progress callback invocation
- ✅ Error handling (invalid prompts, timeouts)
- ✅ Different model configurations
- ✅ Image-to-video, text-to-video modes
- ✅ Response structure validation
- ✅ Raw response preservation

---

## Code Patterns

### Immutable Configuration

**Location:** [models.py:38-50](src/tarash/tarash_gateway/video/models.py#L38-L50)

All configuration objects are immutable (frozen):

```python
class VideoGenerationConfig(BaseModel):
    model: str
    provider: str
    api_key: str
    # ... more fields

    model_config = {"frozen": True}  # Immutable
```

**Usage:**

```python
# Create config
config = VideoGenerationConfig(model="fal-ai/veo3.1", ...)

# Cannot modify
config.model = "other-model"  # Raises error

# Create modified copy
config2 = config.model_copy(update={"model": "other-model"})
```

### Flexible Request Parameters

**Location:** [models.py:56-97](src/tarash/tarash_gateway/video/models.py#L56-L97)

`VideoGenerationRequest` auto-captures unknown fields into `extra_params`:

```python
@model_validator(mode="before")
@classmethod
def capture_extra_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
    """Capture unknown fields into extra_params."""
    if not isinstance(data, dict):
        return data

    extra_params = data.pop("extra_params", {})
    known_fields = set(cls.model_fields.keys())

    # Extract unknown fields
    extra = {k: v for k, v in data.items() if k not in known_fields}

    # Remove from data
    for k in extra.keys():
        data.pop(k)

    # Merge into extra_params
    extra_params.update(extra)
    data["extra_params"] = extra_params

    return data
```

**Usage:**

```python
# Both equivalent
req1 = VideoGenerationRequest(
    prompt="test",
    extra_params={"custom_field": "value"}
)

req2 = VideoGenerationRequest(
    prompt="test",
    custom_field="value"  # Auto-captured into extra_params
)
```

### Logging with Redaction

**Location:** [logging.py:1-158](src/tarash/tarash_gateway/logging.py#L1-L158)

All logging automatically redacts sensitive data:

```python
from tarash.tarash_gateway.logging import log_info, log_error

# Sensitive data is automatically redacted
log_info(
    "Client created",
    api_key="sk-secret123",        # Redacted
    config={"password": "secret"}  # Redacted
)

# Output: api_key='<redacted>', config={'password': '<redacted>'}
```

**Redacted fields:** api_key, password, token, secret, auth, authorization, credential

### Provider Registration

**Location:** [api.py:67-141](src/tarash/tarash_gateway/video/api.py#L67-L141)

Custom providers can be registered at runtime:

```python
from tarash.tarash_gateway.video import api
from tarash.tarash_gateway.video.providers.field_mappers import FieldMapper

# Register custom provider
api.register_provider("my-provider", MyProviderHandler())

# Register field mappings
api.register_provider_field_mapping("my-provider", {
    "my-model": {
        "prompt": FieldMapper(...),
        "duration": FieldMapper(...),
    }
})

# Use custom provider
config = VideoGenerationConfig(
    provider="my-provider",
    model="my-model",
    api_key="...",
)
response = api.generate_video(config, request)
```

### Singleton Handler Pattern

**Location:** [api.py:39-65](src/tarash/tarash_gateway/video/api.py#L39-L65)

Provider handlers are singletons (one instance per provider):

```python
_HANDLER_INSTANCES: dict[str, ProviderHandler] = {}

def _get_handler(provider: str) -> ProviderHandler:
    """Get or create provider handler (singleton pattern)."""
    if provider not in _HANDLER_INSTANCES:
        if provider == "fal":
            _HANDLER_INSTANCES[provider] = FalProviderHandler()
        elif provider == "replicate":
            _HANDLER_INSTANCES[provider] = ReplicateProviderHandler()
        # ... more providers

    return _HANDLER_INSTANCES[provider]
```

**Benefits:**
- Client caching works across calls
- Reduced memory usage
- Consistent state management

---

## Common Pitfalls

### 1. Event Loop Errors with Async Clients

**Problem:** Reusing async clients across different event loops causes "Event loop closed" errors.

**Solution:** Don't cache async clients if provider doesn't support it (see Fal pattern).

### 2. Missing Required Fields

**Problem:** Field mapper validation fails if required field is None after conversion.

**Solution:** Always check both source and converted values:

```python
if mapper.required and source_value is None:
    raise ValueError(f"Required field '{mapper.source_field}' is missing")

converted_value = mapper.converter(request, source_value)

if mapper.required and converted_value is None:
    raise ValueError(f"Required field '{api_field_name}' cannot be None")
```

### 3. Infinite Polling

**Problem:** Polling loop never terminates if status check is incorrect.

**Solution:** Always use `max_poll_attempts` and check terminal states:

```python
for _ in range(config.max_poll_attempts):
    # ... poll logic

    if status in ("succeeded", "completed"):
        return result
    elif status in ("failed", "canceled", "error"):
        raise GenerationFailedError(...)

    await asyncio.sleep(config.poll_interval)

# After loop
raise GenerationFailedError("Max poll attempts exceeded")
```

### 4. Not Handling Both Sync and Async Callbacks

**Problem:** Progress callback only works in one mode.

**Solution:** Always check if callback result is a coroutine:

```python
if on_progress:
    result = on_progress(update)
    if asyncio.iscoroutine(result):
        await result  # Handle async callback
```

### 5. Missing Error Context

**Problem:** Exceptions lack debugging information.

**Solution:** Always include full context when raising:

```python
raise GenerationFailedError(
    f"Generation failed: {error_message}",
    provider=config.provider,
    model=config.model,
    request_id=request_id,
    raw_response=full_response_dict  # Include for debugging
)
```

---

## Development Workflow

### Adding a New Provider

1. **Create provider file** in `video/providers/` (e.g., `my_provider.py`)

2. **Implement ProviderHandler protocol:**
   - `_get_client(config, client_type)`
   - `_validate_params(config, request)`
   - `_convert_request(config, request)`
   - `_convert_response(config, request, request_id, provider_response)`
   - `_handle_error(config, request, request_id, ex)`
   - `generate_video_async(config, request, on_progress=None)`
   - `generate_video(config, request, on_progress=None)`

3. **Define field mapper registry:**
   ```python
   MY_PROVIDER_REGISTRY = {
       "model-name": {
           "prompt": passthrough_field_mapper("prompt", required=True),
           # ... more mappers
       }
   }
   ```

4. **Register in api.py:**
   ```python
   elif provider == "my-provider":
       _HANDLER_INSTANCES[provider] = MyProviderHandler()
   ```

5. **Write unit tests:**
   - Client caching
   - Field mapping
   - Error handling
   - Request/response conversion

6. **Write e2e tests:**
   - Async generation
   - Sync generation
   - Progress tracking
   - Error scenarios

### Adding a New Model

1. **Define field mappers:**
   ```python
   NEW_MODEL_FIELD_MAPPERS = {
       "prompt": passthrough_field_mapper("prompt", required=True),
       "duration": duration_field_mapper("int", [5, 10], "provider", "model"),
       # ... more mappers
   }
   ```

2. **Add to provider registry:**
   ```python
   PROVIDER_MODEL_REGISTRY = {
       # ... existing
       "provider/new-model": NEW_MODEL_FIELD_MAPPERS,
   }
   ```

3. **Test field mapping:**
   ```python
   def test_new_model_field_mapping():
       mappers = get_field_mappers("provider/new-model")
       request = VideoGenerationRequest(prompt="test", duration_seconds=5)
       result = apply_field_mappers(mappers, request)
       assert result == {"prompt": "test", "duration": 5}
   ```

4. **Add e2e test:**
   ```python
   @pytest.mark.e2e
   async def test_new_model_generation(api_key):
       config = VideoGenerationConfig(
           model="provider/new-model",
           provider="provider",
           api_key=api_key
       )
       response = await api.generate_video_async(config, request)
       assert response.status == "completed"
   ```

---

## Summary

This package provides a robust, extensible framework for video generation across multiple providers. Key architectural decisions:

- **Protocol-based abstraction**: All providers implement the same interface
- **Field mapper registry**: Declarative parameter mapping with intelligent lookup
- **Structured exceptions**: Rich context for debugging
- **Dual sync/async**: Full support for both paradigms
- **Progress tracking**: Real-time updates via callbacks
- **Comprehensive testing**: Unit tests with mocks, e2e tests with real APIs

When working on this codebase:
- ✅ Use field mappers for parameter conversion
- ✅ Always include error context (provider, model, request_id, raw_response)
- ✅ Support both sync and async in all provider methods
- ✅ Test client caching strategy for your provider
- ✅ Write both unit and e2e tests
- ✅ Follow existing patterns for consistency

For questions or clarifications, refer to existing implementations (especially [fal.py](src/tarash/tarash_gateway/video/providers/fal.py) as the reference).
