# Image Generation

Tarash Gateway provides a unified interface for generating images across multiple providers. Pass a
[`ImageGenerationConfig`](../api-reference/models.md#imagegenerationconfig) and an
[`ImageGenerationRequest`](../api-reference/models.md#imagegenerationrequest) to
[`generate_image()`](../api-reference/gateway.md) (or `generate_image_async()` for async
workflows); the response is always a normalized `ImageGenerationResponse` regardless of which
provider handled the request.

## Text to Image

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/flux/dev",
    api_key="YOUR_FAL_KEY",
)
request = ImageGenerationRequest(
    prompt="A serene mountain lake at sunrise, photorealistic",
)
response = generate_image(config, request)
print(response.images[0])
```

[Fal.ai image models →](../providers/fal/index.md)

---

## Image to Image / Edit

Providers that support image-to-image workflows accept an input image via the `image_list` field on the request. Each entry is a dict with an `image` key (URL, base64 string, or raw bytes) and a `type` key describing its role (e.g. `"reference"`, `"asset"`).

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/flux-2-pro",
    api_key="YOUR_FAL_KEY",
)
request = ImageGenerationRequest(
    prompt="Make it look like a watercolor painting",
    image_list=[
        {"image": "https://example.com/input.jpg", "type": "reference"},
    ],
)
response = generate_image(config, request)
print(response.images[0])
```

[Fal.ai image models →](../providers/fal/index.md)
