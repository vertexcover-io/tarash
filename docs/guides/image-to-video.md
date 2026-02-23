# Image to Video

Animate a still image into a video by passing it as `image_list` on the request.

## Single image

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(provider="fal", api_key="YOUR_KEY")
request = VideoGenerationRequest(
    prompt="The subject slowly turns to face the camera",
    image_list=["https://example.com/photo.jpg"],  # URL or base64
    duration_seconds=4,
    aspect_ratio="16:9",
)
response = generate_video(config, request)
```

## Local file

Pass a file path or bytes directly — Tarash handles the upload:

```python
request = VideoGenerationRequest(
    prompt="Camera slowly pans left",
    image_list=[open("frame.jpg", "rb").read()],
)
```

## Multiple images

Some providers (e.g. Runway) accept a start and end frame:

```python
request = VideoGenerationRequest(
    prompt="Smooth transition between two scenes",
    image_list=[
        "https://example.com/start.jpg",
        "https://example.com/end.jpg",
    ],
)
```

!!! note
    Not all providers support multiple input images. Check each provider's documentation
    for support details.
