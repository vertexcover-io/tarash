# Wan (via Fal.ai)

Wan v2.6 and v2.5 preview models hosted on Fal.ai.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="wan/v2.6/text-to-video",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    prompt="A bamboo forest swaying in the wind",
    duration_seconds=5,
)

response = generate_video(config, request)
print(response.video)
```

---

## Supported Models

| Model prefix | Duration | Image-to-Video | Notes |
|---|---|:---:|---|
| `wan/v2.6/` | configurable | ✅ | Text, image, reference-to-video |
| `fal-ai/wan-25-preview/` | configurable | ✅ | Wan v2.5 preview |
| `fal-ai/wan/v2.2-14b/animate/` | — | ✅ | Video+image motion control |

---

## Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | |
| `duration_seconds` | — | ✅ | Configurable per sub-variant |
| `image_list` (reference) | — | ✅ | Image-to-video |

---