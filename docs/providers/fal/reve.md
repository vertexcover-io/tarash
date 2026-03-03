# Reve (via Fal.ai)

Reve image generation, editing, and remixing models hosted on Fal.ai.

## Quick Example

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/reve/text-to-image",
    api_key="YOUR_FAL_KEY",
)

request = ImageGenerationRequest(
    prompt="A serene mountain lake at dawn, photorealistic",
    aspect_ratio="16:9",
)

response = generate_image(config, request)
print(response.images[0])
```

---

## Supported Models

| Model | Input | Notes |
|---|---|---|
| `fal-ai/reve/text-to-image` | Text only | Prompt-based generation |
| `fal-ai/reve/edit` | Text + 1 image | Single image editing |
| `fal-ai/reve/fast/edit` | Text + 1 image | Fast editing variant |
| `fal-ai/reve/remix` | Text + 1–6 images | Multi-image scene reimagining |
| `fal-ai/reve/fast/remix` | Text + 1–6 images | Fast remix variant |

---

## Parameters

| Parameter | Field | Required | Notes |
|---|---|:---:|---|
| Prompt | `prompt` | ✅ | Required for all variants |
| Single image | `image_list=[{"image": url, "type": "reference"}]` | — | Required for edit variants |
| Multiple images | `image_list=[{"image": url, "type": "reference"}, ...]` | — | Required for remix variants (1–6 images) |
| Aspect ratio | `aspect_ratio` | — | For remix and text-to-image |
| Number of images | `n` | — | 1–4 output images |
| Output format | `extra_params={"output_format": "png"}` | — | `"png"`, `"jpeg"`, `"webp"` |

---

## Image Editing

Provide exactly one reference image. The model edits it based on the prompt.

```python
request = ImageGenerationRequest(
    prompt="Change the sky to a dramatic sunset",
    image_list=[{"image": "https://example.com/photo.jpg", "type": "reference"}],
)

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/reve/edit",
    api_key="YOUR_FAL_KEY",
)

response = generate_image(config, request)
```

## Image Remixing

Provide 1–6 reference images. The model reimagines the scene combining elements from all images.

```python
request = ImageGenerationRequest(
    prompt="Combine these into a futuristic cityscape",
    image_list=[
        {"image": "https://example.com/building.jpg", "type": "reference"},
        {"image": "https://example.com/skyline.jpg", "type": "reference"},
    ],
    aspect_ratio="16:9",
)

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/reve/remix",
    api_key="YOUR_FAL_KEY",
)

response = generate_image(config, request)
```

---

## Provider-Specific Notes

**Unified mapper:** All Reve variants share a single set of field mappers. The mapper automatically routes single images to `image_url` and multiple images to `image_urls` based on the input count.

**Fast variants:** The `/fast/edit` and `/fast/remix` endpoints trade some quality for speed. They use the same parameters as their non-fast counterparts.
