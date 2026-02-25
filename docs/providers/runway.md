# Runway

Runway provides video generation (Gen-3, Aleph) via the `runwayml` Python SDK. It supports text-to-video, image-to-video, and video-to-video editing.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

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
print(response.video)
```

---

## Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | Text description of the video |
| `aspect_ratio` | — | ✅ | Converted to Runway pixel dimensions |
| `image_list` | — | ✅ | Used as input image for image-to-video |
| `video` | — | ✅ | Required for Aleph (video-to-video) |
| `duration_seconds` | — | — | Duration is model-dependent, not user-configurable |
| `seed` | — | — | Not supported |
| `negative_prompt` | — | — | Not supported |
| `generate_audio` | — | — | Not supported |

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

---

## Supported Models

Tarash automatically selects the correct Runway endpoint (`text_to_video`, `image_to_video`, or `video_to_video`) based on the model name and the inputs provided.

| Model | Notes |
|---|---|
| `gen3a_turbo` | Requires image input |
| `gen-3-alpha` | Auto-selected based on input (text or image) |
| `aleph` | Requires video input; editing/extending |
| `VEO-prefixed models` | Auto-selected based on input |

**Aspect ratio support:** Runway uses pixel dimensions instead of ratio strings. Tarash converts automatically.

| `aspect_ratio` | Runway Size | Modes |
|---|---|---|
| `16:9` | `1280:720` | Text, Image |
| `9:16` | `720:1280` | Text, Image |
| `16:9-wide` | `1080:1920` | Text only |
| `9:16-wide` | `1920:1080` | Text only |
| `4:3` | `1104:832` | Image only |
| `3:4` | `832:1104` | Image only |
| `1:1` | `960:960` | Image only |
| `21:9` | `1584:672` | Image only |

---

## Image-to-Video

Use `gen3a_turbo` with a reference image to generate video from a starting frame.

```python
from tarash.tarash_gateway.models import ImageType

config = VideoGenerationConfig(
    provider="runway",
    model="gen3a_turbo",
    api_key="YOUR_RUNWAY_KEY",
)

request = VideoGenerationRequest(
    prompt="The astronaut floats gently away from the ship",
    aspect_ratio="16:9",
    image_list=[
        ImageType(image="https://example.com/astronaut.jpg", type="reference"),
    ],
)

response = generate_video(config, request)
print(response.video)
```

---

## Video-to-Video

Use the `aleph` model to edit or transform an existing video clip.

```python
from tarash.tarash_gateway.models import MediaContent

config = VideoGenerationConfig(provider="runway", model="aleph", api_key="YOUR_RUNWAY_KEY")

request = VideoGenerationRequest(
    prompt="Make the scene look like it's at night",
    video="https://example.com/original-clip.mp4",   # or MediaContent bytes
)

response = generate_video(config, request)
print(response.video)
```

---

## Provider-Specific Notes

**Endpoint auto-selection:** The endpoint is chosen at request time:
- Model name contains `"aleph"` → `video_to_video` (requires `video` input)
- Model name contains `"turbo"` (non-VEO) → `image_to_video` (requires image)
- VEO models → `image_to_video` if image provided, else `text_to_video`
- Everything else → `image_to_video` if image provided, else `text_to_video`

**Task polling:** Runway uses a task-based API. Tarash polls until the task reaches a terminal state: `SUCCEEDED`, `FAILED`, or `CANCELLED`.

**Video download:** Tarash downloads the video content after the task completes, so the response includes the video bytes (not just a URL).

**No client caching:** Runway clients are created fresh per request. Runway's SDK handles connection management internally.
