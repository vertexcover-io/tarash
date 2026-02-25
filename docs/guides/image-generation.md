# Image Generation

Tarash Gateway provides a unified interface for generating images across multiple providers. Pass a provider-specific `ImageGenerationConfig` and an `ImageGenerationRequest` to `generate_image()` (or `generate_image_async()` for async workflows); the response is always a normalized `ImageGenerationResponse` regardless of which provider handled the request.

## Text to Image

Each example below uses the minimal config required to call the provider. Set `api_key` to your key for the relevant provider.

### Fal.ai

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/flux-pro",
    api_key="YOUR_API_KEY",
)
request = ImageGenerationRequest(
    prompt="A serene mountain lake at sunrise, photorealistic",
)
response = generate_image(config, request)
print(response.images[0])
```

### OpenAI

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="openai",
    model="dall-e-3",
    api_key="YOUR_API_KEY",
)
request = ImageGenerationRequest(
    prompt="A serene mountain lake at sunrise, photorealistic",
)
response = generate_image(config, request)
print(response.images[0])
```

### Google

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="google",
    model="imagen-3.0-generate-002",
    api_key="YOUR_API_KEY",
)
request = ImageGenerationRequest(
    prompt="A serene mountain lake at sunrise, photorealistic",
)
response = generate_image(config, request)
print(response.images[0])
```

### Azure OpenAI

Azure OpenAI requires `base_url` (your Azure endpoint) and `api_version`.

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="azure-openai",
    model="dall-e-3",
    api_key="YOUR_API_KEY",
    base_url="https://YOUR_RESOURCE.openai.azure.com/",
    api_version="2024-05-01-preview",
)
request = ImageGenerationRequest(
    prompt="A serene mountain lake at sunrise, photorealistic",
)
response = generate_image(config, request)
print(response.images[0])
```

### Stability AI

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="stability",
    model="stable-image-core",
    api_key="YOUR_API_KEY",
)
request = ImageGenerationRequest(
    prompt="A serene mountain lake at sunrise, photorealistic",
)
response = generate_image(config, request)
print(response.images[0])
```

---

## Image to Image / Edit

Providers that support image-to-image workflows accept an input image via the `image_list` field on the request. Each entry is a dict with an `image` key (URL, base64 string, or raw bytes) and a `type` key describing its role (e.g. `"reference"`, `"asset"`).

### OpenAI

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="openai",
    model="gpt-image-1.5",
    api_key="YOUR_API_KEY",
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

### Stability AI

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="stability",
    model="stable-image-core",
    api_key="YOUR_API_KEY",
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

### Fal.ai (Flux Kontext)

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/flux-pro/kontext",
    api_key="YOUR_API_KEY",
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
