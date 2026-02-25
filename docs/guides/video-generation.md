# Video Generation

Tarash Gateway provides a unified interface for text-to-video, image-to-video, and video-to-video
generation across Fal.ai, OpenAI, Google, Runway, Replicate, and Azure OpenAI — using a
single `generate_video` call regardless of provider.

## Text to Video

Pass a prompt and config to generate a video from text.

### Fal.ai

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1",
    api_key="YOUR_FAL_KEY",
)
request = VideoGenerationRequest(
    prompt="A golden retriever running on a beach",
    duration_seconds=5,
    aspect_ratio="16:9",
)
response = generate_video(config, request)
print(response.video)
```

### OpenAI

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="openai",
    model="sora",
    api_key="YOUR_OPENAI_KEY",
)
request = VideoGenerationRequest(
    prompt="A golden retriever running on a beach",
    duration_seconds=5,
    aspect_ratio="16:9",
)
response = generate_video(config, request)
print(response.video)
```

### Google

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="google",
    model="veo-3.0-generate-001",
    api_key="YOUR_GOOGLE_KEY",
)
request = VideoGenerationRequest(
    prompt="A golden retriever running on a beach",
    duration_seconds=5,
    aspect_ratio="16:9",
)
response = generate_video(config, request)
print(response.video)
```

### Runway

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="runway",
    model="gen3a_turbo",
    api_key="YOUR_RUNWAY_KEY",
)
request = VideoGenerationRequest(
    prompt="A golden retriever running on a beach",
    duration_seconds=5,
    aspect_ratio="16:9",
)
response = generate_video(config, request)
print(response.video)
```

### Replicate

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="replicate",
    model="google/veo-3",
    api_key="YOUR_REPLICATE_TOKEN",
)
request = VideoGenerationRequest(
    prompt="A golden retriever running on a beach",
    duration_seconds=5,
    aspect_ratio="16:9",
)
response = generate_video(config, request)
print(response.video)
```

### Azure OpenAI

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="azure_openai",
    model="sora",  # deployment name in Azure
    api_key="YOUR_AZURE_API_KEY",
    base_url="https://YOUR_RESOURCE.openai.azure.com/",
    api_version="2025-01-01-preview",
)
request = VideoGenerationRequest(
    prompt="A golden retriever running on a beach",
    duration_seconds=5,
    aspect_ratio="16:9",
)
response = generate_video(config, request)
print(response.video)
```

---

## Image to Video

Pass one or more images via `image_list` to animate a still into a video.

### Fal.ai

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest, ImageType

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/kling-video/v2.6",
    api_key="YOUR_FAL_KEY",
)
request = VideoGenerationRequest(
    prompt="The dog shakes water off its fur",
    duration_seconds=5,
    aspect_ratio="16:9",
    image_list=[ImageType(image="https://example.com/dog.jpg", type="first_frame")],
)
response = generate_video(config, request)
print(response.video)
```

### OpenAI

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest, ImageType

config = VideoGenerationConfig(
    provider="openai",
    model="sora",
    api_key="YOUR_OPENAI_KEY",
)
request = VideoGenerationRequest(
    prompt="The dog shakes water off its fur",
    duration_seconds=5,
    aspect_ratio="16:9",
    image_list=[ImageType(image="https://example.com/dog.jpg", type="first_frame")],
)
response = generate_video(config, request)
print(response.video)
```

### Google

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest, ImageType

config = VideoGenerationConfig(
    provider="google",
    model="veo-3.0-generate-001",
    api_key="YOUR_GOOGLE_KEY",
)
request = VideoGenerationRequest(
    prompt="The dog shakes water off its fur",
    duration_seconds=5,
    aspect_ratio="16:9",
    image_list=[ImageType(image="https://example.com/dog.jpg", type="first_frame")],
)
response = generate_video(config, request)
print(response.video)
```

### Runway

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest, ImageType

config = VideoGenerationConfig(
    provider="runway",
    model="gen3a_turbo",
    api_key="YOUR_RUNWAY_KEY",
)
request = VideoGenerationRequest(
    prompt="The dog shakes water off its fur",
    duration_seconds=5,
    aspect_ratio="16:9",
    image_list=[ImageType(image="https://example.com/dog.jpg", type="first_frame")],
)
response = generate_video(config, request)
print(response.video)
```

### Replicate

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest, ImageType

config = VideoGenerationConfig(
    provider="replicate",
    model="kwaivgi/kling",
    api_key="YOUR_REPLICATE_TOKEN",
)
request = VideoGenerationRequest(
    prompt="The dog shakes water off its fur",
    duration_seconds=5,
    aspect_ratio="16:9",
    image_list=[ImageType(image="https://example.com/dog.jpg", type="first_frame")],
)
response = generate_video(config, request)
print(response.video)
```

---

## Video to Video

Only Runway's Aleph model supports video-to-video currently. Pass the source video via the `video`
field on the request.

### Runway

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="runway",
    model="aleph",
    api_key="YOUR_RUNWAY_KEY",
)
request = VideoGenerationRequest(
    prompt="Make the scene look like it was filmed at golden hour",
    duration_seconds=5,
    aspect_ratio="16:9",
    video="https://example.com/source.mp4",
)
response = generate_video(config, request)
print(response.video)
```
