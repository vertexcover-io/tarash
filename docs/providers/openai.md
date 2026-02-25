# OpenAI

OpenAI provides image generation via **DALL-E** and **GPT Image**, and video generation via **Sora**. Tarash uses the official `openai` Python SDK.

---

## Image Generation

### Quick Example

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="openai",
    model="dall-e-3",
    api_key="sk-...",
)
request = ImageGenerationRequest(
    prompt="A futuristic cityscape at night, photorealistic",
    size="1792x1024",
    quality="hd",
    style="vivid",
    n=1,
)
response = generate_image(config, request)
print(response.images[0])
```

### Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | Text description of the image |
| `size` | — | ✅ | e.g. `1024x1024`, `1792x1024` |
| `quality` | — | ✅ | `hd` or `standard` (DALL-E 3) |
| `style` | — | ✅ | `vivid` or `natural` (DALL-E 3) |
| `n` | — | ✅ | Number of images (DALL-E 2: up to 10) |
| `seed` | — | — | Not supported |
| `negative_prompt` | — | — | Not supported |

### Supported Models

| Model ID | Max Images | Sizes |
|---|---|---|
| `gpt-image-1.5` | 1 | `1024×1024`, `1024×1792`, `1792×1024`, `auto` |
| `dall-e-3` | 1 | `1024×1024`, `1024×1792`, `1792×1024` |
| `dall-e-2` | up to 10 | `256×256`, `512×512`, `1024×1024` |

---

## Video Generation

### Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="openai",
    model="openai/sora-2",
    api_key="sk-...",
)
request = VideoGenerationRequest(
    prompt="A serene mountain lake at golden hour",
    duration_seconds=8,
    aspect_ratio="16:9",
)
response = generate_video(config, request)
print(response.video)
```

### Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | Text description of the video |
| `duration_seconds` | — | ✅ | 4, 8, 12 (sora-2) or 10, 15, 25 (sora-2-pro) |
| `aspect_ratio` | — | ✅ | Converted to size string |
| `image_list` | — | ✅ | Max 1 reference image for image-to-video |
| `negative_prompt` | — | — | Not supported by Sora |
| `seed` | — | — | Not supported |
| `generate_audio` | — | — | Not supported |

### Supported Models

| Model ID | Duration Options | Aspect Ratios | Notes |
|---|---|---|---|
| `openai/sora-2` | 4s, 8s, 12s | 16:9, 9:16, 1:1, 16:10, 10:16 | Standard Sora |
| `openai/sora-2-pro` | 10s, 15s, 25s | 16:9, 9:16, 1:1, 16:10, 10:16 | Higher quality, longer |

### Image-to-Video

Sora supports using a reference image as the first frame of the generated video. Pass a single image via `image_list` — only the first image is used regardless of its `type` value.

```python
from tarash.tarash_gateway.models import ImageType, VideoGenerationRequest

request = VideoGenerationRequest(
    prompt="The car drives away into the sunset",
    image_list=[ImageType(image="https://example.com/car.jpg", type="reference")],
    duration_seconds=8,
    aspect_ratio="16:9",
)
```

### Video Remix

To remix an existing Sora video, pass the original `video_id` via `extra_params`. Sora will use the existing video as a base and apply your prompt as a variation.

```python
request = VideoGenerationRequest(
    prompt="Same scene but with snow",
    extra_params={"video_id": "video_abc123"},
)
```

---

## Azure OpenAI

Azure OpenAI runs the same Sora and image models through Microsoft Azure. Use `provider="azure_openai"` and provide two additional required fields: `base_url` (your Azure resource endpoint) and `api_version`.

### Quick Example

```python
config = VideoGenerationConfig(
    provider="azure_openai",
    model="my-sora-deployment",   # your Azure deployment name
    api_key="YOUR_AZURE_KEY",
    base_url="https://my-resource.openai.azure.com/",
    api_version="2024-05-01-preview",
)
```

Parameters, image/video generation, Image-to-Video, and Video Remix all work identically to the standard OpenAI provider above.

### Azure-Specific Notes

**Deployment names:** Azure requires a custom deployment name instead of a model ID (e.g. `my-sora-deployment` rather than `openai/sora-2`).

**`api_version` from URL:** If `base_url` already contains `?api-version=...`, it is extracted automatically.

**Authentication:** Use `api_key` for an Azure resource key, or set it to a bearer token obtained via `azure-identity` for Azure AD authentication.

---

## Provider-Specific Notes

**Client caching:** Both sync and async OpenAI clients are cached per `(api_key, base_url)` pair. This is safe because OpenAI's clients support reuse across calls.

**Content download:** After generation, Tarash automatically downloads the video bytes so the response includes the content directly, not just a URL.
