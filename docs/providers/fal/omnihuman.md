# OmniHuman (via Fal.ai)

ByteDance OmniHuman generates vivid, high-quality videos from a human figure image paired with an audio file. The character's emotions and movements maintain a strong correlation with the audio, producing synchronized lip movements and expressive body motion.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/bytedance/omnihuman/v1.5",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    prompt="A person speaking naturally",
    image_list=[{"type": "reference", "image": "https://example.com/person.jpg"}],
    resolution="720p",
    extra_params={
        "audio_url": "https://example.com/speech.mp3",
        "turbo_mode": True,
    },
)

response = generate_video(config, request)
print(response.video)
```

---

## Supported Models

| Model ID | Cost | Notes |
|---|---|---|
| `fal-ai/bytedance/omnihuman/v1.5` | $0.16/sec | Latest version, improved quality |
| `fal-ai/bytedance/omnihuman` | $0.16/sec | Original version |

---

## Parameters

| Parameter | Field | Required | Notes |
|---|---|:---:|---|
| `image_url` | `image_list` (type: `reference`) | ✅ | Human figure image URL |
| `audio_url` | `extra_params` | ✅ | Audio file URL (max 30s for 1080p, 60s for 720p) |
| `prompt` | `prompt` | — | Optional text guidance for video generation |
| `resolution` | `resolution` | — | `720p` or `1080p` (default: `1080p`) |
| `turbo_mode` | `extra_params` | — | Faster generation with slight quality trade-off |

---
