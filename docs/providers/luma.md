# Luma AI

Luma AI provides video generation (Dream Machine) and image generation (Photon) via the `lumaai` Python SDK. It supports text-to-video with keyframe control and image generation with reference/style images.

## Installation

```bash
pip install tarash-gateway[luma]
```

---

## Quick Example (Video)

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="luma",
    model="luma-dream-machine",
    api_key="YOUR_LUMA_KEY",
)

request = VideoGenerationRequest(
    prompt="A golden retriever running through a field of wildflowers at sunset",
    aspect_ratio="16:9",
    resolution="1080p",
    duration_seconds=5,
)

response = generate_video(config, request)
print(response.video)
```

## Quick Example (Image)

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="luma",
    model="photon-1",
    api_key="YOUR_LUMA_KEY",
)

request = ImageGenerationRequest(
    prompt="A futuristic cityscape with flying cars and neon lights",
    aspect_ratio="16:9",
)

response = generate_image(config, request)
print(response.images[0])
```

---

## Video Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | Yes | Yes | Text description of the video |
| `aspect_ratio` | — | Yes | Passed through to Luma API |
| `resolution` | — | Yes | `720p`, `1080p`, `4k` only (360p, 480p rejected) |
| `duration_seconds` | — | Yes | `5` or `9` seconds only |
| `image_list` | — | Yes | `first_frame` and `last_frame` types used as keyframes |
| `seed` | — | No | Not supported |
| `negative_prompt` | — | No | Not supported |
| `generate_audio` | — | No | Not supported |

### Extra params (video)

| Key | Type | Notes |
|---|---|---|
| `loop` | `bool` | Whether the generated video should loop seamlessly |

```python
request = VideoGenerationRequest(
    prompt="Waves crashing on a rocky shore",
    duration_seconds=5,
    extra_params={"loop": True},
)
```

---

## Image Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | Yes | Yes | Text description of the image |
| `aspect_ratio` | — | Yes | Passed through to Luma API |
| `image_list` | — | Yes | `reference` and `style` types supported (see below) |

### Extra params (image)

| Key | Type | Default | Notes |
|---|---|---|---|
| `image_ref_weight` | `float` | `1.0` | Weight for reference images |
| `style_ref_weight` | `float` | `0.8` | Weight for style reference images |
| `character_ref` | `dict` | — | Character reference object (passed through) |
| `modify_image_ref` | `dict` | — | Modify-image reference object (passed through) |

---

## Keyframe Video (Image-to-Video)

Use `first_frame` and `last_frame` image types to control the start and end of the generated video.

```python
from tarash.tarash_gateway.models import ImageType

config = VideoGenerationConfig(
    provider="luma",
    model="luma-dream-machine",
    api_key="YOUR_LUMA_KEY",
)

request = VideoGenerationRequest(
    prompt="The camera slowly zooms out revealing the full landscape",
    duration_seconds=5,
    image_list=[
        ImageType(image="https://example.com/start.jpg", type="first_frame"),
        ImageType(image="https://example.com/end.jpg", type="last_frame"),
    ],
)

response = generate_video(config, request)
print(response.video)
```

---

## Image with References

Use `reference` and `style` image types to guide image generation.

```python
from tarash.tarash_gateway.models import ImageType

config = ImageGenerationConfig(
    provider="luma",
    model="photon-1",
    api_key="YOUR_LUMA_KEY",
)

request = ImageGenerationRequest(
    prompt="A portrait in the same style",
    image_list=[
        ImageType(image="https://example.com/ref.jpg", type="reference"),
        ImageType(image="https://example.com/style.jpg", type="style"),
    ],
    extra_params={
        "image_ref_weight": 0.9,
        "style_ref_weight": 0.7,
    },
)

response = generate_image(config, request)
print(response.images[0])
```

---

## Provider-Specific Notes

- **Authentication:** The API key is passed as `auth_token` to the Luma SDK (`LumaAI(auth_token=api_key)`). Always provide `api_key` explicitly in the config.
- **Status polling:** Tarash manually polls the generation status until it reaches a terminal state (`completed` or `failed`). Configure polling behavior with `max_poll_attempts` and `poll_interval` on the config.
- **Status mapping:** Luma statuses are mapped as follows: `queued` -> `queued`, `dreaming` -> `processing`, `completed` -> `completed`, `failed` -> `failed`.
- **Video format:** Completed video responses return an MP4 URL (`video/mp4`).
- **Image format:** Completed image responses return a JPEG URL (`image/jpeg`).
- **Resolution validation:** Only `720p`, `1080p`, and `4k` are accepted for video. Requesting `360p` or `480p` raises a `ValidationError`.
- **Duration validation:** Only `5` and `9` seconds are accepted for video. Other values raise a `ValidationError`.
