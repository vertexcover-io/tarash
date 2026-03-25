# xAI

xAI provides video and image generation via Grok Imagine models through its native gRPC-based SDK (`xai-sdk`). This provider connects directly to the xAI API.

> **Looking for xAI models on Fal.ai?** See [Grok Imagine Image (via Fal.ai)](fal/grok-imagine.md) for the Fal-hosted variant, which uses a different SDK and API key.

---

## Installation

```bash
pip install tarash-gateway[xai]
```

---

## Video Generation

### Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="xai",
    model="grok-imagine-video",
    api_key="YOUR_XAI_KEY",
    max_poll_attempts=60,
    poll_interval=5,
)

request = VideoGenerationRequest(
    prompt="A drone shot flying over a misty forest at sunrise",
    duration_seconds=5,
    aspect_ratio="16:9",
    resolution="720p",
)

response = generate_video(config, request)
print(response.video)
```

### Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | Yes | Yes | Text description of the video |
| `duration_seconds` | -- | Yes | Integer, 1-15 seconds |
| `resolution` | -- | Yes | `"480p"` or `"720p"` only |
| `aspect_ratio` | -- | Yes | Passthrough to xAI API |
| `image_list` | -- | Yes | Single image for image-to-video (1 image only) |
| `video` | -- | Yes | Video URL for video-to-video generation |
| `seed` | -- | -- | Not supported |
| `negative_prompt` | -- | -- | Not supported |
| `generate_audio` | -- | -- | Not supported |

### Image-to-Video

Provide a single reference image via `image_list` for image-to-video generation:

```python
from tarash.tarash_gateway.models import ImageType

request = VideoGenerationRequest(
    prompt="The landscape comes alive with gentle wind and drifting clouds",
    image_list=[
        ImageType(image="https://example.com/landscape.jpg", type="reference"),
    ],
    resolution="720p",
)

response = generate_video(config, request)
print(response.video)
```

### Video-to-Video

Pass a video URL to transform an existing clip:

```python
request = VideoGenerationRequest(
    prompt="Add a cinematic color grade with warm tones",
    video="https://example.com/original.mp4",
)

response = generate_video(config, request)
print(response.video)
```

---

## Image Generation

### Quick Example

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="xai",
    model="grok-imagine-image",
    api_key="YOUR_XAI_KEY",
)

request = ImageGenerationRequest(
    prompt="A photorealistic portrait of a cat wearing a tiny astronaut helmet",
    aspect_ratio="1:1",
    extra_params={"resolution": "2k"},
)

response = generate_image(config, request)
print(response.images[0])
```

### Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | Yes | Yes | Text description of the image |
| `aspect_ratio` | -- | Yes | Passthrough to xAI API |
| `extra_params.resolution` | -- | Yes | `"1k"` or `"2k"` |
| `image_list` (1 image) | -- | Yes | Single reference image (maps to `image_url`) |
| `image_list` (2+ images) | -- | Yes | Multiple reference images (maps to `image_urls`) |
| `n` | -- | -- | Not supported (single image per request) |
| `seed` | -- | -- | Not supported |
| `negative_prompt` | -- | -- | Not supported |

### Image Editing with Reference Images

Pass one or more reference images via `image_list`:

```python
# Single reference image
request = ImageGenerationRequest(
    prompt="Transform this photo into a watercolor painting",
    image_list=[
        {"image": "https://example.com/photo.jpg", "type": "reference"},
    ],
)

# Multiple reference images
request = ImageGenerationRequest(
    prompt="Combine these scenes into a single panoramic landscape",
    image_list=[
        {"image": "https://example.com/left.jpg", "type": "reference"},
        {"image": "https://example.com/right.jpg", "type": "reference"},
    ],
)
```

---

## Supported Models

| Model | Type | Notes |
|---|---|---|
| `grok-imagine-video` | Video | Polling-based video generation (text/image/video-to-video) |
| `grok-imagine-image` | Image | Single-call image generation |
| `grok-2-image` | Image | Grok 2 image generation |

---

## Provider-Specific Notes

- **SDK:** Requires `xai-sdk`. Install with `pip install tarash-gateway[xai]`.
- **Authentication:** `api_key` is passed directly to the xAI `Client`/`AsyncClient`. There is no automatic environment variable reading -- always provide the key explicitly.
- **Video polling:** Video generation is asynchronous on the xAI side. Tarash polls until the task reaches `DONE` or `EXPIRED`. Configure polling behavior with `max_poll_attempts` and `poll_interval` on `VideoGenerationConfig`.
- **Image generation:** Image generation is a single synchronous API call (no polling required).
- **Content moderation:** xAI enforces content policies. If a request is rejected, Tarash raises a `ContentModerationError` with details in `raw_response`.
- **gRPC errors:** The xAI SDK uses gRPC. Network and auth errors are mapped to Tarash exceptions (`TimeoutError`, `HTTPConnectionError`, `HTTPError`, `ValidationError`).
