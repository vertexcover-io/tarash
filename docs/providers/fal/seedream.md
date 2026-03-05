# Seedream (via Fal.ai)

ByteDance Seedream v5 Lite image generation and editing models hosted on Fal.ai.

## Quick Example

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/bytedance/seedream/v5/lite/text-to-image",
    api_key="YOUR_FAL_KEY",
)

request = ImageGenerationRequest(
    prompt="A serene Japanese garden with cherry blossoms, photorealistic",
    size="auto_2K",
)

response = generate_image(config, request)
print(response.images[0])
```

---

## Supported Models

| Model | Input | Notes |
|---|---|---|
| `fal-ai/bytedance/seedream/v5/lite/text-to-image` | Text only | Prompt-based generation |
| `fal-ai/bytedance/seedream/v5/lite/edit` | Text + 1–10 images | Intelligent image editing |

---

## Parameters

| Parameter | Field | Required | Notes |
|---|---|:---:|---|
| Prompt | `prompt` | ✅ | Required for all variants |
| Input images | `image_list=[{"image": url, "type": "reference"}, ...]` | — | Required for edit variant (up to 10 images) |
| Image size | `size` | — | `"auto_2K"` (default), `"auto_3K"`, `"square_hd"`, `"square"`, `"portrait_4_3"`, `"portrait_16_9"`, `"landscape_4_3"`, `"landscape_16_9"` |
| Number of images | `n` | — | 1–6 separate generations |
| Max images per generation | `extra_params={"max_images": 3}` | — | 1–6 images per generation |
| Safety checker | `extra_params={"enable_safety_checker": true}` | — | Enabled by default |

---

## Image Editing

Provide up to 10 reference images. The model edits them based on the prompt.

```python
request = ImageGenerationRequest(
    prompt="Add a rainbow arching over the scene",
    image_list=[{"image": "https://example.com/photo.jpg", "type": "reference"}],
)

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/bytedance/seedream/v5/lite/edit",
    api_key="YOUR_FAL_KEY",
)

response = generate_image(config, request)
```

---

## Provider-Specific Notes

**Unified mapper:** Both Seedream v5 Lite variants share a single set of field mappers. The `image_urls` field is automatically populated from `image_list` when reference images are provided.

**Image size:** Unlike most image models that use pixel dimensions, Seedream uses preset size names like `"auto_2K"` (2560x1440) and `"auto_3K"` (3072x3072).
