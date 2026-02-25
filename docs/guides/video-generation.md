# Video Generation

Tarash Gateway provides a unified interface for text-to-video, image-to-video, and video-to-video
generation across Fal.ai, OpenAI, Google, Runway, Replicate, and Azure OpenAI — using a
single `generate_video` call regardless of provider.

## Text to Video

Pass a prompt and config to generate a video from text.

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

---

## Image to Video

Pass one or more images via `image_list` to animate a still into a video.

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

---

## Video to Video

Pass the source video via the `video` field to edit or transform an existing clip.

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
    video="https://example.com/source.mp4",
)
response = generate_video(config, request)
print(response.video)
```
