# Pixverse (via Fal.ai)

Pixverse v5 and v5.5 models with transition, effects, swap, and lipsync support.

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
    prompt="A coral reef with fish, underwater",
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
| `fal-ai/pixverse/lipsync` | — | — | Lipsync (video + audio/TTS) |

---

## Parameters

| Parameter | Required | Notes |
|---|:---:|---|
| `prompt` | ✅ | |
| `duration_seconds` | — | `5`, `8`, or `10` |
| `image_list` (reference) | — | Image-to-video |

---

## Lipsync

The `fal-ai/pixverse/lipsync` model syncs lip movements in a video to audio. You can provide audio directly via `audio_url`, or use built-in TTS by supplying `text` and `voice_id`.

### With Audio URL

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/pixverse/lipsync",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    video_url="https://example.com/speaker.mp4",
    audio_url="https://example.com/speech.mp3",
)

response = generate_video(config, request)
print(response.video)
```

### With TTS (text + voice)

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/pixverse/lipsync",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    video_url="https://example.com/speaker.mp4",
    text="Hello, welcome to our demo!",
    voice_id="Emily",
)

response = generate_video(config, request)
print(response.video)
```

### Lipsync Parameters

| Parameter | Required | Notes |
|---|:---:|---|
| `video_url` | ✅ | Source video with a speaking face |
| `audio_url` | — | Audio to sync (mutually exclusive with `text`+`voice_id`) |
| `text` | — | Text for built-in TTS (requires `voice_id`) |
| `voice_id` | — | TTS voice: `Emily`, `James`, `Isabella`, `Liam`, `Chloe`, `Adrian`, `Harper`, `Ava`, `Sophia`, `Julia`, `Mason`, `Jack`, `Oliver`, `Ethan`, `Auto` |

---