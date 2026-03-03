# Grok Imagine Image (via Fal.ai)

xAI's Grok Imagine model for text-to-image generation and image editing via Fal.ai.

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
    prompt="A photorealistic mountain landscape at golden hour",
    n=1,
    aspect_ratio="16:9",
)

response = generate_image(config, request)
print(response.images)  # → list of image URLs
```

---

## Supported Models

| Model | Description | Image Input | Notes |
|---|---|:---:|---|
| `xai/grok-imagine-image` | Text-to-image generation | — | Supports aspect_ratio |
| `xai/grok-imagine-image/edit` | Image editing | ✅ | Up to 3 reference images |

---

## Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | Text description of desired image |
| `n` | — | ✅ | Number of images (1–4), maps to `num_images` |
| `aspect_ratio` | — | ✅ | Text-to-image only (e.g. `"16:9"`, `"1:1"`) |
| `image_list` (reference) | — | ✅ | Image editing: up to 3 reference images |
| `extra_params.output_format` | — | ✅ | `"jpeg"`, `"png"`, or `"webp"` |

---

## Image Editing Example

```python
config = ImageGenerationConfig(
    provider="fal",
    model="xai/grok-imagine-image/edit",
    api_key="YOUR_FAL_KEY",
)

request = ImageGenerationRequest(
    prompt="Transform the scene into a watercolor painting style",
    image_list=[
        {"image": "https://example.com/photo.jpg", "type": "reference"},
    ],
    extra_params={"output_format": "png"},
)

response = generate_image(config, request)
print(response.images[0])
```
