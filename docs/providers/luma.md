# Luma AI

Luma AI provides video generation via the **Dream Machine** model using the `lumaai` Python SDK. It supports text-to-video and image-to-video with keyframe control.

## Capabilities

| Feature | Supported |
|---|:---:|
| Video generation | ✅ |
| Image generation | — |
| Image-to-video | ✅ |
| First/last frame (keyframes) | ✅ |
| Async | ✅ |
| Progress callbacks | ✅ |

## Installation

```bash
pip install tarash-gateway[lumaai]
# or
uv add tarash-gateway[lumaai]
```

## Configuration

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="luma",
    model="dream-machine",      # Or any Luma model name
    api_key="...",              # Required: Luma API key (auth_token)
    timeout=600,
    max_poll_attempts=120,
    poll_interval=5,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | Luma API key |
| `timeout` | `int` | `600` | Request timeout in seconds |
| `max_poll_attempts` | `int` | `120` | Max polling iterations |
| `poll_interval` | `int` | `5` | Seconds between status checks |

## Video Models

| Model | Duration Options | Resolution Options |
|---|---|---|
| Dream Machine (any model name) | **5s or 9s only** | **720p, 1080p, 4k only** |

!!! warning "Strict resolution and duration limits"
    Luma does not support 360p or 480p. Requesting either will raise a `ValidationError`.
    Duration must be exactly 5 or 9 seconds. Any other value raises a `ValidationError`.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="luma",
    model="dream-machine",
    api_key="YOUR_LUMA_KEY",
)

request = VideoGenerationRequest(
    prompt="A slow pan across a neon-lit Tokyo street at night",
    duration_seconds=5,
    resolution="1080p",
    aspect_ratio="16:9",
)

response = generate_video(config, request)
print(response.video)
```

### Image-to-video with keyframes

Pin the first and/or last frame of the generated video:

```python
from tarash.tarash_gateway.models import ImageType

request = VideoGenerationRequest(
    prompt="The door slowly swings open to reveal sunlight",
    duration_seconds=5,
    image_list=[
        ImageType(image="https://example.com/door-closed.jpg", type="first_frame"),
        ImageType(image="https://example.com/door-open.jpg", type="last_frame"),
    ],
)
```

### Looping video

```python
request = VideoGenerationRequest(
    prompt="Waves crashing on a beach, seamless loop",
    duration_seconds=5,
    extra_params={"loop": True},
)
```

## Supported Request Parameters

| Parameter | Supported | Notes |
|---|:---:|---|
| `prompt` | ✅ | Required |
| `duration_seconds` | ✅ | **Must be 5 or 9** |
| `resolution` | ✅ | **Must be 720p, 1080p, or 4k** |
| `aspect_ratio` | ✅ | Passed to Luma API |
| `image_list` (first_frame) | ✅ | Mapped to Luma `frame0` keyframe |
| `image_list` (last_frame) | ✅ | Mapped to Luma `frame1` keyframe |
| `seed` | — | Not supported |
| `negative_prompt` | — | Not supported |
| `generate_audio` | — | Not supported |

### Extra params

| Key | Type | Description |
|---|---|---|
| `loop` | `bool` | Generate a seamlessly looping video |

## Provider-Specific Notes

**Keyframe mapping:** Tarash maps `first_frame` → Luma `frame0` and `last_frame` → Luma `frame1`. Luma's keyframe API accepts image URLs directly.

**No client caching:** A fresh `LumaAI` / `AsyncLumaAI` client is created per request. Luma's SDK does not recommend client reuse.

**Manual polling:** Luma does not support event streaming. Tarash polls the generation status manually at `poll_interval` second intervals.

**Content moderation:** Luma raises a `ContentModerationError` (maps from Luma's `PermissionDeniedError`) if the request violates content policies.
