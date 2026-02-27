# Grok Imagine Image (via Fal.ai)

xAI's Grok Imagine model for text-to-image generation and image editing.

## Quick Example

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="fal",
    model="xai/grok-imagine-image",
    api_key="YOUR_FAL_KEY",
)

request = ImageGenerationRequest(
    prompt="A serene mountain lake at sunrise with mist rising from the water",
    aspect_ratio="16:9",
    n=1,
)

response = generate_image(config, request)
print(response.images[0])
```

---

## Supported Models

| Model | Variant | Notes |
|---|---|---|
| `xai/grok-imagine-image` | Text-to-image | Generate images from text prompts |
| `xai/grok-imagine-image/edit` | Image editing | Edit images using reference images (up to 3) |

---

## Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | Max 8000 characters |
| `aspect_ratio` | — | ✅ | Text-to-image only (e.g. `16:9`, `1:1`) |
| `n` | — | ✅ | Number of images (1–4, default: 1) |
| `image_list` (reference) | — | ✅ | Image editing only; up to 3 reference images |
| `extra_params.output_format` | — | ✅ | `jpeg`, `png`, or `webp` (default: `jpeg`) |

---

## Image Editing Example

```python
config = ImageGenerationConfig(
    provider="fal",
    model="xai/grok-imagine-image/edit",
    api_key="YOUR_FAL_KEY",
)

request = ImageGenerationRequest(
    prompt="Make the background a snowy winter landscape",
    image_list=[
        {"type": "reference", "image": "https://example.com/photo.jpg"},
    ],
    extra_params={"output_format": "png"},
)

response = generate_image(config, request)
print(response.images[0])
```
