# Google

Google provides **Veo 3** video generation and **Imagen 3** / **Gemini** image generation via the `google-genai` SDK. Supports both the Gemini Developer API (API key) and Google Cloud Vertex AI (service account / ADC).

---

## Image Generation

### Quick Example

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="google",
    model="imagen-3.0-generate-002",
    api_key="AIza...",
)

request = ImageGenerationRequest(
    prompt="A serene Japanese garden with cherry blossoms, photorealistic",
    aspect_ratio="16:9",
)

response = generate_image(config, request)
print(response.images[0])
```

### Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | Text description of the image |
| `aspect_ratio` | — | ✅ | Supported by Imagen 3 and Gemini image models |
| `n` | — | ✅ | Number of images to generate (see image count rules) |
| `seed` | — | — | Not supported |
| `negative_prompt` | — | — | Not supported |

### Supported Models

| Model ID | Notes |
|---|---|
| `imagen-3.0-generate-001` | Imagen 3 |
| `imagen-3.0-generate-002` | Imagen 3 latest |
| `gemini-2.5-flash-image` | Nano Banana |
| `gemini-3-pro-image` | Gemini pro image |

Imagen models use the dedicated `generate_images` API. Gemini image models (`gemini-*`) use the `generate_content` API. Tarash detects automatically based on model name prefix.

---

## Video Generation

### Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="google",
    model="veo-3.0-generate-preview",
    api_key="AIza...",
)

request = VideoGenerationRequest(
    prompt="A drone shot over a coastal cliff at sunset",
    aspect_ratio="16:9",
    resolution="1080p",
)

response = generate_video(config, request)
print(response.video)
```

### Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | Text description of the video |
| `aspect_ratio` | — | ✅ | Passed to Veo3 |
| `resolution` | — | ✅ | `720p`, `1080p`, `4k` |
| `image_list` (reference/asset/style) | — | ✅ | Up to 3 reference images |
| `image_list` (first_frame) | — | ✅ | Interpolation mode start |
| `image_list` (last_frame) | — | ✅ | Interpolation mode end |
| `negative_prompt` | — | ✅ | |
| `seed` | — | ✅ | |
| `enhance_prompt` | — | ✅ | |
| `generate_audio` | — | ✅ | |

### Supported Models

| Model ID | Notes |
|---|---|
| `veo-3.0-generate-preview` | Veo 3 — standard |
| `veo-3.1-generate-preview` | Veo 3.1 — longer sequences |

**Supported resolutions:** 720p, 1080p, 4k

### Image-to-Video

Veo3 supports using reference images as start and/or end frames for interpolation. Pass images via `image_list` with `type="first_frame"` and/or `type="last_frame"`.

```python
from tarash.tarash_gateway.models import ImageType

request = VideoGenerationRequest(
    prompt="The sun rises above the mountain",
    image_list=[
        ImageType(image="https://example.com/dawn.jpg", type="first_frame"),
        ImageType(image="https://example.com/sunrise.jpg", type="last_frame"),
    ],
)
```

---

## Provider-Specific Notes

**Authentication:**

- **Gemini Developer API:** Set `api_key` to your `AIza...` key. No `provider_config` needed.
- **Vertex AI:** Set `api_key=None` and supply a `provider_config` dict with at minimum `gcp_project`. Optional keys: `location` (default `us-central1`) and `credentials_path` (path to a service account JSON; omit to use Application Default Credentials).

```python
# Vertex AI config
config = VideoGenerationConfig(
    provider="google",
    model="veo-3.0-generate-preview",
    api_key=None,
    provider_config={
        "gcp_project": "my-gcp-project",
        "location": "us-central1",               # Optional, default us-central1
        "credentials_path": "/path/to/key.json", # Optional, uses ADC if omitted
    },
    timeout=600,
)
```

**Image count rules (Veo3):**

| Scenario | Allowed |
|---|---|
| Reference images only (asset/style) | Up to 3 |
| First frame only | 1 |
| First + last frame (interpolation) | 1 + 1 |
| Reference images + first_frame | ❌ Not allowed |
| Any + last_frame without first_frame | ❌ Requires first_frame |

**No SDK caching:** Google clients are created fresh per request. The `google-genai` SDK manages its own connection pooling internally.

**Person generation policy:** Veo3 has strict content policies for generating people. If your request triggers a policy violation, a `ContentModerationError` is raised with details in `raw_response`.
