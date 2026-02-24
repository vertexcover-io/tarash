# Google

Google provides **Veo 3** video generation and **Imagen 3** / **Gemini** image generation via the `google-genai` SDK. Supports both the Gemini Developer API (API key) and Google Cloud Vertex AI (service account / ADC).

## Supported Models

### Video

| Model ID | Notes |
|---|---|
| `veo-3.0-generate-preview` | Veo 3 — standard |
| `veo-3.5-generate-preview` | Veo 3.5 — longer sequences |

**Supported resolutions:** 720p, 1080p, 4k

### Image

| Model ID | Notes |
|---|---|
| `imagen-3.0-generate-001` | Imagen 3 |
| `imagen-3.0-generate-002` | Imagen 3 latest |
| `gemini-2.5-flash-image` | Nano Banana |
| `gemini-3-pro-image` | Gemini pro image |

Imagen models use the dedicated `generate_images` API. Gemini image models (`gemini-*`) use the `generate_content` API. Tarash detects automatically based on model name prefix.

## Capabilities

| Feature | Supported |
|---|:---:|
| Video generation | ✅ |
| Image generation | ✅ |
| Image-to-video | ✅ |
| First/last frame (interpolation) | ✅ |
| Async | ✅ |
| Progress callbacks | ✅ |

## Configuration

### Gemini Developer API (API key)

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="google",
    model="veo-3.0-generate-preview",
    api_key="AIza...",   # or omit — reads GOOGLE_API_KEY env var
    timeout=600,
    max_poll_attempts=120,
    poll_interval=5,
)
```

### Vertex AI (Google Cloud)

```python
config = VideoGenerationConfig(
    provider="google",
    model="veo-3.0-generate-preview",
    api_key=None,   # No API key for Vertex AI
    provider_config={
        "gcp_project": "my-gcp-project",
        "location": "us-central1",               # Optional, default us-central1
        "credentials_path": "/path/to/key.json", # Optional, uses ADC if omitted
    },
    timeout=600,
)
```

**`provider_config` keys for Vertex AI:**

| Key | Required | Description |
|---|:---:|---|
| `gcp_project` | ✅ | Google Cloud project ID |
| `location` | — | Region (default: `us-central1`) |
| `credentials_path` | — | Path to service account JSON. Omit to use Application Default Credentials. |

**Authentication detection:** If `api_key` is set, the Gemini Developer API is used. If `api_key` is `None`, Vertex AI mode is activated and `provider_config.gcp_project` is required (or the `GOOGLE_CLOUD_PROJECT` environment variable).

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

# Text-to-video
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

### Image-to-video (first/last frame interpolation)

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

## Supported Request Parameters

| Parameter | Supported | Notes |
|---|:---:|---|
| `prompt` | ✅ | Required |
| `aspect_ratio` | ✅ | Passed to Veo3 |
| `resolution` | ✅ | `720p`, `1080p`, `4k` |
| `image_list` (reference/asset/style) | ✅ | Up to 3 reference images |
| `image_list` (first_frame) | ✅ | Interpolation mode start |
| `image_list` (last_frame) | ✅ | Interpolation mode end |
| `negative_prompt` | ✅ | |
| `seed` | ✅ | |
| `enhance_prompt` | ✅ | |
| `generate_audio` | ✅ | |

## Provider-Specific Notes

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
