# OpenAI

OpenAI provides video generation via **Sora** and image generation via **DALL-E** and **GPT Image**. Tarash uses the official `openai` Python SDK.

## Capabilities

| Feature | Supported |
|---|:---:|
| Video generation | ✅ |
| Image generation | ✅ |
| Image-to-video | ✅ |
| Async | ✅ |
| Progress callbacks | ✅ |

## Configuration

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="openai",
    model="openai/sora-2",
    api_key="sk-...",           # Required
    base_url=None,              # Optional: override endpoint
    timeout=600,                # Seconds before timeout (default: 600)
    max_poll_attempts=120,      # Max status checks (default: 120)
    poll_interval=5,            # Seconds between polls (default: 5)
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | OpenAI API key |
| `base_url` | `str \| None` | `None` | Override API endpoint |
| `timeout` | `int` | `600` | Request timeout in seconds |
| `max_poll_attempts` | `int` | `120` | Max polling iterations |
| `poll_interval` | `int` | `5` | Seconds between status checks |

## Video Models

| Model ID | Duration Options | Aspect Ratios | Notes |
|---|---|---|---|
| `openai/sora-2` | 4s, 8s, 12s | 16:9, 9:16, 1:1, 16:10, 10:16 | Standard Sora |
| `openai/sora-2-pro` | 10s, 15s, 25s | 16:9, 9:16, 1:1, 16:10, 10:16 | Higher quality, longer |

**Aspect ratio → size mapping:**

| Aspect Ratio | Size |
|---|---|
| `16:9` | 1280×720 |
| `9:16` | 720×1280 |
| `1:1` | 1024×1024 |
| `16:10` | 1792×1024 |
| `10:16` | 1024×1792 |

## Image Models

| Model ID | Max Images | Sizes |
|---|---|---|
| `gpt-image-1.5` | 1 | 1024×1024, 1024×1792, 1792×1024, auto |
| `dall-e-3` | 1 | 1024×1024, 1024×1792, 1792×1024 |
| `dall-e-2` | up to 10 | 256×256, 512×512, 1024×1024 |

## Quick Example

```python
from tarash.tarash_gateway import generate_video, generate_image
from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
    ImageGenerationConfig,
    ImageGenerationRequest,
)

# Video generation (Sora)
video_config = VideoGenerationConfig(
    provider="openai",
    model="openai/sora-2",
    api_key="sk-...",
)
video_request = VideoGenerationRequest(
    prompt="A serene mountain lake at golden hour",
    duration_seconds=8,
    aspect_ratio="16:9",
)
video_response = generate_video(video_config, video_request)
print(video_response.video)  # URL to the generated video

# Image generation (DALL-E 3)
image_config = ImageGenerationConfig(
    provider="openai",
    model="dall-e-3",
    api_key="sk-...",
)
image_request = ImageGenerationRequest(
    prompt="A futuristic cityscape at night, photorealistic",
)
image_response = generate_image(image_config, image_request)
print(image_response.images[0])
```

## Supported Request Parameters

### Video

| Parameter | Supported | Notes |
|---|:---:|---|
| `prompt` | ✅ | Required |
| `duration_seconds` | ✅ | 4, 8, 12 (sora-2) or 10, 15, 25 (sora-2-pro) |
| `aspect_ratio` | ✅ | Converted to size string |
| `image_list` | ✅ | Max 1 reference image for image-to-video |
| `negative_prompt` | — | Not supported by Sora |
| `seed` | — | Not supported |
| `generate_audio` | — | Not supported |

### Image

| Parameter | Supported | Notes |
|---|:---:|---|
| `prompt` | ✅ | Required |
| `size` | ✅ | Via `extra_params={"size": "1024x1024"}` |
| `quality` | ✅ | `hd` or `standard` (DALL-E 3) |
| `style` | ✅ | `vivid` or `natural` (DALL-E 3) |
| `n` | ✅ | Number of images (DALL-E 2: up to 10) |

## Provider-Specific Notes

**Image-to-video:** Only 1 reference image is supported. Provide it in `image_list` with any `type` value (the first image is used).

```python
from tarash.tarash_gateway.models import ImageType

request = VideoGenerationRequest(
    prompt="The car drives away into the sunset",
    image_list=[ImageType(url="https://example.com/car.jpg", type="reference")],
    duration_seconds=8,
)
```

**Video remix:** To remix an existing Sora video, pass the `video_id` via `extra_params`:

```python
request = VideoGenerationRequest(
    prompt="Same scene but with snow",
    extra_params={"video_id": "video_abc123"},
)
```

**Client caching:** Both sync and async OpenAI clients are cached per `(api_key, base_url)` pair. This is safe because OpenAI's clients support reuse across calls.

**Content download:** After generation, Tarash automatically downloads the video bytes so the response includes the content directly, not just a URL.
