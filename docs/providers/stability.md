# Stability AI

Stability AI provides image generation via Stable Diffusion 3.5 and Stable Image models. This provider uses the Stability AI REST API directly via `httpx` (no SDK dependency).

## Capabilities

| Feature | Supported |
|---|:---:|
| Video generation | — |
| Image generation | ✅ |
| Image-to-video | — |
| Async | ✅ |
| Progress callbacks | — |

!!! info "Image only"
    Stability AI does not support video generation. Use this provider with `ImageGenerationConfig` and `generate_image()`.

## Installation

No extra SDK required — Tarash uses `httpx` which is already a core dependency.

```bash
pip install tarash-gateway
# or
uv add tarash-gateway
```

## Configuration

```python
from tarash.tarash_gateway.models import ImageGenerationConfig

config = ImageGenerationConfig(
    provider="stability",
    model="sd3.5-large",
    api_key="sk-...",   # Required: Stability AI API key
    timeout=120,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | Stability AI API key |
| `timeout` | `int` | `600` | Request timeout in seconds |

**API base URL:** `https://api.stability.ai` (fixed, not configurable)

## Image Models

| Model ID | Quality | Speed | Notes |
|---|---|---|---|
| `sd3.5-large` | Highest | Slower | Stable Diffusion 3.5 Large |
| `sd3.5-large-turbo` | High | Fast | Turbo variant |
| `sd3.5-medium` | Good | Faster | Smaller model |
| `stable-image-ultra` | Ultra | — | Highest quality Stable Image |
| `stable-image-core` | Standard | Fast | Fast generation |
| `stable-image` | Standard | Fast | Prefix match for `stable-image-*` |

## Quick Example

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="stability",
    model="sd3.5-large",
    api_key="sk-...",
)

request = ImageGenerationRequest(
    prompt="A majestic eagle soaring over mountain peaks, ultra-detailed, photorealistic",
    aspect_ratio="16:9",
)

response = generate_image(config, request)
print(response.images[0])   # Base64 image or URL
```

### Using the turbo model for faster output

```python
config = ImageGenerationConfig(
    provider="stability",
    model="sd3.5-large-turbo",
    api_key="sk-...",
)
```

## Supported Request Parameters

| Parameter | Supported | Notes |
|---|:---:|---|
| `prompt` | ✅ | Required |
| `aspect_ratio` | ✅ | Validated against allowed values |
| `negative_prompt` | ✅ | SD 3.5 only |
| `seed` | ✅ | 0 to 4,294,967,294 |
| `cfg_scale` | ✅ | Via `extra_params` |
| `steps` | ✅ | Via `extra_params` |

### Extra params

Pass additional Stability API parameters via `extra_params`:

```python
request = ImageGenerationRequest(
    prompt="...",
    extra_params={
        "cfg_scale": 7.0,    # Guidance scale
        "steps": 30,          # Number of diffusion steps
    },
)
```

### Validated field ranges

| Field | Range / Allowed Values |
|---|---|
| `seed` | 0 – 4,294,967,294 |
| `cfg_scale` | Provider-defined |
| `aspect_ratio` | `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `21:9`, `9:21` |

## Provider-Specific Notes

**No SDK — direct REST API:** Tarash sends HTTP requests to `https://api.stability.ai` using `httpx`. There is no `stability-sdk` Python package involved.

**Model detection:** The Stability model is inferred from the `model` field using prefix matching:
- `sd3.5-large-turbo` → SD 3.5 Large Turbo endpoint
- `sd3.5-medium` → SD 3.5 Medium endpoint
- `sd3.5-large` (or any other `sd3.5-*`) → SD 3.5 Large endpoint
- `stable-image-ultra` → Stable Image Ultra endpoint
- `stable-image` (prefix) → Stable Image Ultra endpoint

**No progress callbacks:** Stability's REST API is synchronous — it returns the image directly with no polling or streaming. Progress callbacks are not invoked.

**Response format:** Images are returned as base64-encoded data or direct bytes in the response, depending on the Stability endpoint.
