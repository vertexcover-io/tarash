# Seedance (via Fal.ai)

ByteDance Seedance v1 and v1.5 models hosted on Fal.ai.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/bytedance/seedance/v1/lite/text-to-video",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    prompt="A panda eating bamboo in a lush forest",
    duration_seconds=5,
)

response = generate_video(config, request)
print(response.video)
```

---

## Supported Models

| Model prefix | Duration | Image-to-Video | Notes |
|---|---|:---:|---|
| `fal-ai/bytedance/seedance` | `2s`–`12s` | ✅ | Seedance v1/v1.5; reference-to-video |

---

## Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | |
| `duration_seconds` | — | ✅ | `2`–`12` seconds |
| `image_list` (reference) | — | ✅ | Image-to-video / reference-to-video |

---