# Replicate

Replicate is a platform for running open-source AI models. Tarash supports video generation via Kling, Minimax (Hailuo), Wan, and Google Veo 3.

## Supported Models

Model names on Replicate often include version hashes (e.g., `minimax/video-01:abc123`). Tarash strips the hash before registry lookup, then uses **prefix matching** so you can pass version-pinned names without changing config.

| Model ID / Prefix | Duration Options | Image-to-Video | Notes |
|---|---|:---:|---|
| `kwaivgi/kling` | 5s, 10s | ✅ | Kling v2.1. Image input **required**. |
| `minimax/` | 6s, 10s | ✅ | Matches any `minimax/*` model |
| `hailuo/` | 6s, 10s | ✅ | Matches any `hailuo/*` model |
| `wan-video/` | — | ✅ | Wan video models |
| `google/veo-3` | 4s, 6s, 8s | ✅ | Google Veo 3 via Replicate |

**Example with version hash:**

```python
config = VideoGenerationConfig(
    provider="replicate",
    model="minimax/video-01:abc123def456",  # Hash stripped, matches "minimax/" prefix
    api_key="...",
)
```

## Capabilities

| Feature | Supported |
|---|:---:|
| Video generation | ✅ |
| Image generation | — |
| Image-to-video | ✅ |
| Async | ✅ |
| Progress callbacks | ✅ |

## Configuration

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="replicate",
    model="kwaivgi/kling-v2.1",
    api_key="...",      # or omit — reads REPLICATE_API_TOKEN env var
    timeout=600,
    max_poll_attempts=120,
    poll_interval=5,
)
```

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
    ImageType,
)

config = VideoGenerationConfig(
    provider="replicate",
    model="kwaivgi/kling-v2.1",
    api_key="YOUR_REPLICATE_TOKEN",
)

# Kling requires an image input
request = VideoGenerationRequest(
    prompt="The kite soars higher into the stormy sky",
    duration_seconds=5,
    image_list=[
        ImageType(image="https://example.com/kite.jpg", type="first_frame"),
    ],
)

response = generate_video(config, request)
print(response.video)
```

### Google Veo 3 via Replicate

```python
config = VideoGenerationConfig(
    provider="replicate",
    model="google/veo-3",
    api_key="YOUR_REPLICATE_TOKEN",
)

request = VideoGenerationRequest(
    prompt="A bamboo forest in early morning mist",
    duration_seconds=8,
    aspect_ratio="16:9",
)

response = generate_video(config, request)
```

## Supported Request Parameters

| Parameter | Supported | Models | Notes |
|---|:---:|---|---|
| `prompt` | ✅ | All | Required |
| `duration_seconds` | ✅ | Kling, Minimax, Veo3 | Integer seconds |
| `image_list` (first_frame) | ✅ | Kling | Start frame |
| `image_list` (last_frame) | ✅ | Kling | End frame |
| `image_list` (reference) | ✅ | Minimax | Reference image |
| `enhance_prompt` | ✅ | Minimax | As `prompt_optimizer` |
| `aspect_ratio` | ✅ | Veo3 | Passed through |
| `seed` | — | — | |
| `negative_prompt` | — | — | |
| `generate_audio` | — | — | |

## Provider-Specific Notes

**Kling v2.1 requires image input.** The `kwaivgi/kling` model only supports image-to-video. If no image is provided in `image_list`, a `ValidationError` is raised.

**Manual polling.** Unlike Fal's event streaming, Replicate uses a manual status polling loop. Tarash checks the prediction status every `poll_interval` seconds up to `max_poll_attempts` times. Terminal statuses: `succeeded`, `failed`, `canceled`.

**Version hash handling.** Model names with `:` are split on `:` to strip version hashes before registry lookup:

```
minimax/video-01:abc123  →  lookup: minimax/video-01  →  prefix match: minimax/
```

**Generic fallback.** For models not in the registry, Tarash applies generic mappers that pass `prompt`, `seed`, `negative_prompt`, and `aspect_ratio` through unchanged, and drops everything else.
