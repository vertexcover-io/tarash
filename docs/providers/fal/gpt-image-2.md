# GPT Image 2 (via Fal.ai)

OpenAI's GPT Image 2 model for image editing via Fal.ai.

## Quick Example

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="fal",
    model="openai/gpt-image-2/edit",
    api_key="YOUR_FAL_KEY",
)

request = ImageGenerationRequest(
    prompt="Add a dramatic rainbow arching over the elephant",
    image_list=[
        {"image": "https://example.com/photo.jpg", "type": "reference"},
    ],
    n=1,
    quality="high",
)

response = generate_image(config, request)
print(response.images)  # → list of image URLs
```

---

## Supported Models

| Model | Description | Image Input | Notes |
|---|---|:---:|---|
| `openai/gpt-image-2/edit` | Image editing with reference images | ✅ | Up to 16 reference images; optional mask for targeted edits |

---

## Parameters

| Parameter | Required | Notes |
|---|:---:|---|
| `prompt` | ✅ | Text description of the desired edit |
| `image_list` (reference) | ✅ | One or more reference images to edit |
| `n` | — | Number of images to generate (1–4), maps to `num_images` |
| `quality` | — | `"low"`, `"medium"`, or `"high"` (default: `"high"`) |
| `size` | — | `"auto"`, `"square_hd"`, `"portrait_4_3"`, etc. (default: `"auto"`) |
| `extra_params.output_format` | — | `"jpeg"`, `"png"`, or `"webp"` (default: `"png"`) |
| `extra_params.mask_url` | — | URL of a mask image — white areas indicate the region to edit |

---

## Image Editing with Mask

```python
config = ImageGenerationConfig(
    provider="fal",
    model="openai/gpt-image-2/edit",
    api_key="YOUR_FAL_KEY",
)

request = ImageGenerationRequest(
    prompt="Replace the sky with a stormy night",
    image_list=[
        {"image": "https://example.com/landscape.jpg", "type": "reference"},
    ],
    extra_params={
        "mask_url": "https://example.com/sky-mask.png",
        "output_format": "png",
    },
)

response = generate_image(config, request)
print(response.images[0])
```
