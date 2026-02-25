# Kling (via Fal.ai)

Kuaishou's Kling v2.6 and o1 models hosted on Fal.ai.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/kling-video/v2.6",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    prompt="A cat playing piano in a jazz club",
    duration_seconds=5,
    aspect_ratio="16:9",
)

response = generate_video(config, request)
print(response.video)
```

---

## Supported Models

| Model prefix | Duration | Image-to-Video | Notes |
|---|---|:---:|---|
| `fal-ai/kling-video/v2.6` | `5s`, `10s` | ✅ | Motion control, `cfg_scale`, last-frame pinning |
| `fal-ai/kling-video/o1` | `5s`, `10s` | ✅ | Reference-to-video, video edit, start/end frame |

---

## Parameters

| Parameter | Required | Supported | Models | Notes |
|---|:---:|:---:|---|---|
| `prompt` | ✅ | ✅ | All | |
| `duration_seconds` | — | ✅ | All | `5` or `10` |
| `image_list` (reference) | — | ✅ | kling-o1 | Image-to-video |
| `image_list` (first_frame) | — | ✅ | kling-o1 | First frame pinning |
| `image_list` (last_frame) | — | ✅ | kling-v2.6, kling-o1 | Last frame pinning |
| `negative_prompt` | — | ✅ | kling-v2.6 | |
| `generate_audio` | — | ✅ | kling-v2.6 | |

---

## Motion Control

```python
# Kling v2.6 motion control
request = VideoGenerationRequest(
    prompt="A dancer performing on stage",
    duration_seconds=5,
    extra_params={
        "cfg_scale": 0.5,
        "character_orientation": "front",
        "keep_original_sound": True,
    },
)

# Kling o1 reference-to-video elements
request = VideoGenerationRequest(
    prompt="The character walks through a park",
    extra_params={
        "elements": [{"frontal_image_url": "https://example.com/character.jpg"}]
    },
)
```

---