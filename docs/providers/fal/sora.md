# Sora (via Fal.ai)

OpenAI's Sora-2 model hosted on Fal.ai's infrastructure.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/sora-2",
    api_key="YOUR_FAL_KEY",
)

# Text-to-video
request = VideoGenerationRequest(
    prompt="A futuristic cityscape at night with flying cars",
    duration_seconds=8,
    aspect_ratio="16:9",
)

response = generate_video(config, request)
print(response.video)
```

---

## Supported Models

| Model prefix | Duration | Image-to-Video | Notes |
|---|---|:---:|---|
| `fal-ai/sora-2` | `4s`, `8s`, `12s` | ✅ | Remix via `/video-to-video/remix` |

---

## Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | |
| `duration_seconds` | — | ✅ | `4`, `8`, or `12` |
| `aspect_ratio` | — | ✅ | Passed through directly |
| `image_list` (reference) | — | ✅ | Image-to-video |
| `video` | — | ✅ | Video edit/remix input |

---

## Video Remix

To remix an existing video, use the `/video-to-video/remix` endpoint and pass the source `video_id` via `extra_params`.

```python
# Sora-2 remix (requires fal-ai/sora-2/video-to-video/remix endpoint)
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/sora-2/video-to-video/remix",
)

request = VideoGenerationRequest(
    prompt="Same scene in winter",
    extra_params={"video_id": "video_abc123"},
)
```

---