# Runway

Runway provides video generation (Gen-3, Aleph) via the `runwayml` Python SDK. It supports text-to-video, image-to-video, and video-to-video editing.

## Capabilities

| Feature | Supported |
|---|:---:|
| Video generation | ✅ |
| Image generation | — |
| Image-to-video | ✅ |
| Video-to-video (editing) | ✅ (Aleph) |
| Async | ✅ |
| Progress callbacks | ✅ |

## Configuration

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="runway",
    model="gen3a_turbo",
    api_key="...",      # Required: Runway API key
    timeout=600,
    max_poll_attempts=120,
    poll_interval=5,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | Runway API key |
| `timeout` | `int` | `600` | Request timeout in seconds |
| `max_poll_attempts` | `int` | `120` | Max polling iterations |
| `poll_interval` | `int` | `5` | Seconds between status checks |

## Video Models

Tarash automatically selects the correct Runway endpoint (`text_to_video`, `image_to_video`, or `video_to_video`) based on the model name and the inputs provided.

| Model | Endpoint Selected | Notes |
|---|---|---|
| `gen3a_turbo` | `image_to_video` | Requires image input |
| `gen-3-alpha` | `text_to_video` or `image_to_video` | Auto-selected based on input |
| `aleph` | `video_to_video` | Requires video input; editing/extending |
| VEO-prefixed models | `text_to_video` or `image_to_video` | Auto-selected |

## Aspect Ratio Support

Runway uses pixel dimensions instead of ratio strings. Tarash converts automatically.

### Text-to-video

| `aspect_ratio` | Runway Size |
|---|---|
| `16:9` | 1280:720 |
| `9:16` | 720:1280 |
| `16:9-wide` | 1080:1920 |
| `9:16-wide` | 1920:1080 |

### Image-to-video

| `aspect_ratio` | Runway Size |
|---|---|
| `16:9` | 1280:720 |
| `9:16` | 720:1280 |
| `4:3` | 1104:832 |
| `3:4` | 832:1104 |
| `1:1` | 960:960 |
| `21:9` | 1584:672 |

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
    ImageType,
)

# Image-to-video with Gen-3 Turbo
config = VideoGenerationConfig(
    provider="runway",
    model="gen3a_turbo",
    api_key="YOUR_RUNWAY_KEY",
)

request = VideoGenerationRequest(
    prompt="The astronaut floats gently away from the ship",
    aspect_ratio="16:9",
    image_list=[
        ImageType(url="https://example.com/astronaut.jpg", type="reference"),
    ],
)

response = generate_video(config, request)
print(response.video)
```

### Text-to-video with Gen-3 Alpha

```python
config = VideoGenerationConfig(
    provider="runway",
    model="gen-3-alpha",
    api_key="YOUR_RUNWAY_KEY",
)

request = VideoGenerationRequest(
    prompt="A lone wolf running across a snow-covered tundra at dawn",
    aspect_ratio="16:9",
)

response = generate_video(config, request)
```

### Video-to-video with Aleph

```python
from tarash.tarash_gateway.models import MediaContent

request = VideoGenerationRequest(
    prompt="Make the scene look like it's at night",
    video="https://example.com/original-clip.mp4",   # or MediaContent bytes
)

config = VideoGenerationConfig(provider="runway", model="aleph", api_key="...")
response = generate_video(config, request)
```

## Supported Request Parameters

| Parameter | Supported | Notes |
|---|:---:|---|
| `prompt` | ✅ | Required |
| `aspect_ratio` | ✅ | Converted to Runway pixel dimensions |
| `image_list` | ✅ | Used as input image for image-to-video |
| `video` | ✅ | Required for Aleph (video-to-video) |
| `duration_seconds` | — | Duration is model-dependent, not user-configurable |
| `seed` | — | Not supported |
| `negative_prompt` | — | Not supported |
| `generate_audio` | — | Not supported |

### Content moderation

Pass Runway's content moderation options via `extra_params`:

```python
request = VideoGenerationRequest(
    prompt="...",
    extra_params={
        "content_moderation": {"publicFigures": "disallow"},
    },
)
```

## Provider-Specific Notes

**Endpoint auto-selection:** The endpoint is chosen at request time:
- Model name contains `"aleph"` → `video_to_video` (requires `video` input)
- Model name contains `"turbo"` (non-VEO) → `image_to_video` (requires image)
- VEO models → `image_to_video` if image provided, else `text_to_video`
- Everything else → `image_to_video` if image provided, else `text_to_video`

**Task polling:** Runway uses a task-based API. Tarash polls until the task reaches a terminal state: `SUCCEEDED`, `FAILED`, or `CANCELLED`.

**Video download:** Tarash downloads the video content after the task completes, so the response includes the video bytes (not just a URL).

**No client caching:** Runway clients are created fresh per request. Runway's SDK handles connection management internally.
