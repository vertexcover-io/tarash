# Pixverse (via Fal.ai)

Pixverse v5 and v5.5 models with transition, effects, and swap support.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/pixverse/v5",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    prompt="A vibrant coral reef teeming with colorful fish",
    duration_seconds=5,
)

response = generate_video(config, request)
print(response.video)
```

---

## Supported Models

| Model prefix | Duration | Image-to-Video | Notes |
|---|---|:---:|---|
| `fal-ai/pixverse/v5` | `5s`, `8s`, `10s` | ✅ | Transition, effects, swap |
| `fal-ai/pixverse/v5.5` | `5s`, `8s`, `10s` | ✅ | Same API as v5 |
| `fal-ai/pixverse/swap` | — | ✅ | Swap variant |

---

## Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | |
| `duration_seconds` | — | ✅ | `5`, `8`, or `10` |
| `image_list` (reference) | — | ✅ | Image-to-video |

---