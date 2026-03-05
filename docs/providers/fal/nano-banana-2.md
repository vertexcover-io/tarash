# Nano Banana 2 (via Fal.ai)

Nano Banana 2 (Google Gemini 3.1 Flash Image) text-to-image generation and image editing models hosted on Fal.ai.

## Quick Example

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/nano-banana-2",
    api_key="YOUR_FAL_KEY",
)

request = ImageGenerationRequest(
    prompt="A photorealistic mountain landscape at golden hour",
    aspect_ratio="16:9",
    seed=42,
)

response = generate_image(config, request)
print(response.images[0])
```

---

## Supported Models

| Model | Input | Notes |
|---|---|---|
| `fal-ai/nano-banana-2` | Text only | Prompt-based generation |
| `fal-ai/nano-banana-2/edit` | Text + images | Image editing with up to 14 reference images |

---

## Parameters

| Parameter | Field | Required | Notes |
|---|---|:---:|---|
| Prompt | `prompt` | ✅ | Required for all variants (3–50,000 chars) |
| Images | `image_list=[{"image": url, "type": "reference"}, ...]` | — | Required for edit variant |
| Number of images | `n` | — | 1–4 output images |
| Seed | `seed` | — | Reproducible generation |
| Aspect ratio | `aspect_ratio` | — | `"auto"`, `"21:9"`, `"16:9"`, `"3:2"`, `"4:3"`, `"5:4"`, `"1:1"`, `"4:5"`, `"3:4"`, `"2:3"`, `"9:16"` |
| Output format | `extra_params={"output_format": "png"}` | — | `"jpeg"`, `"png"`, `"webp"` |
| Resolution | `extra_params={"resolution": "1K"}` | — | `"0.5K"`, `"1K"`, `"2K"`, `"4K"` |
| Safety tolerance | `extra_params={"safety_tolerance": "4"}` | — | `"1"` (strictest) to `"6"` (least strict) |
| Web search | `extra_params={"enable_web_search": true}` | — | Use latest web information |
| Limit generations | `extra_params={"limit_generations": true}` | — | Restrict to 1 generation per round |

---

## Image Editing

Provide reference images for editing. The model edits them based on the prompt.

```python
request = ImageGenerationRequest(
    prompt="Add a rainbow arching over the mountains",
    image_list=[{"image": "https://example.com/photo.jpg", "type": "reference"}],
)

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/nano-banana-2/edit",
    api_key="YOUR_FAL_KEY",
)

response = generate_image(config, request)
```

---

## Provider-Specific Notes

**Unified mapper:** Both Nano Banana 2 variants share a single set of field mappers registered at the `fal-ai/nano-banana-2` prefix.

**Resolution options:** The `resolution` parameter controls output quality from `"0.5K"` (fastest) to `"4K"` (highest quality). Default is `"1K"`.
