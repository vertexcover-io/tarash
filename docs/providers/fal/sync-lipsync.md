# Sync Lipsync (via Fal.ai)

Sync Labs lipsync models generate realistic lip-sync animations from video and audio inputs while maintaining natural facial features.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/sync-lipsync/v2/pro",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    prompt="lipsync",  # Required by interface but unused by model
    video="https://example.com/input-video.mp4",
    extra_params={
        "audio_url": "https://example.com/input-audio.mp3",
        "sync_mode": "cut_off",
    },
)

response = generate_video(config, request)
print(response.video)
```

---

## Supported Models

| Model prefix | Cost | Notes |
|---|---|---|
| `fal-ai/sync-lipsync` | $0.70/min | v1.9.0-beta |
| `fal-ai/sync-lipsync/v2` | $3.00/min | Enhanced facial tracking |
| `fal-ai/sync-lipsync/v2/pro` | $5.00/min | Optimized for speed and throughput |

---

## Parameters

| Parameter | Field | Required | Notes |
|---|---|:---:|---|
| `video` | `video` | ✅ | Input video URL (MP4, MOV, WebM, M4V, GIF) |
| `audio_url` | `extra_params` | ✅ | Input audio URL (MP3, OGG, WAV, M4A, AAC) |
| `sync_mode` | `extra_params` | — | Duration mismatch handling: `cut_off` (default), `loop`, `bounce`, `silence`, `remap` |
| `model` | `extra_params` | — | Sub-model selection on v1/v2 base endpoints |

---
