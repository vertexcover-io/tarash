# Seedance (via Fal.ai)

ByteDance Seedance v1, v1.5, and v2.0 models hosted on Fal.ai.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="bytedance/seedance-2.0/text-to-video",
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
| `bytedance/seedance-2.0` | `4s`–`15s` | ✅ | Seedance v2.0; text-to-video and image-to-video |

---

## Parameters

### Seedance v2.0 (`bytedance/seedance-2.0/*`)

| Parameter | Required | Notes |
|---|:---:|---|
| `prompt` | ✅ | |
| `duration_seconds` | — | `4`–`15` seconds |
| `resolution` | — | `480p`, `720p`, `1080p` |
| `aspect_ratio` | — | `16:9`, `9:16`, `1:1`, `21:9`, `4:3`, `3:4` |
| `generate_audio` | — | Synchronized audio generation |
| `seed` | — | |
| `image_list` (reference) | — | Starting frame for image-to-video |
| `image_list` (last_frame) | — | Ending frame for image-to-video |
| `extra_params.end_user_id` | — | User identifier |

### Seedance v1/v1.5 (`fal-ai/bytedance/seedance/*`)

| Parameter | Required | Notes |
|---|:---:|---|
| `prompt` | ✅ | |
| `duration_seconds` | — | `2`–`12` seconds |
| `image_list` (reference) | — | Image-to-video / reference-to-video |

---