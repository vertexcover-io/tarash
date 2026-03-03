# Replicate

Replicate is a platform for running open-source AI models. Tarash supports video generation via Kling, Kling Lip Sync, Luma Dream Machine, Minimax (Hailuo), Wan, and Google Veo 3.

---

## Installation

```bash
pip install tarash-gateway[replicate]
```

---

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
    ImageType,
)

config = VideoGenerationConfig(
    provider="replicate",
    model="kwaivgi/kling-v2.1",
    api_key="YOUR_REPLICATE_TOKEN",
)

# Kling requires an image input
request = VideoGenerationRequest(
    prompt="The kite soars higher into the stormy sky",
    duration_seconds=5,
    image_list=[
        ImageType(image="https://example.com/kite.jpg", type="first_frame"),
    ],
)

response = generate_video(config, request)
print(response.video)
```

### Google Veo 3 via Replicate

```python
config = VideoGenerationConfig(
    provider="replicate",
    model="google/veo-3",
    api_key="YOUR_REPLICATE_TOKEN",
)

request = VideoGenerationRequest(
    prompt="A bamboo forest in early morning mist",
    duration_seconds=8,
    aspect_ratio="16:9",
)

response = generate_video(config, request)
```

### Kling Lip Sync via Replicate

```python
config = VideoGenerationConfig(
    provider="replicate",
    model="kwaivgi/kling-lip-sync",
    api_key="YOUR_REPLICATE_TOKEN",
)

# Audio-driven lip sync
request = VideoGenerationRequest(
    prompt="lipsync",
    video="https://example.com/talking-head.mp4",
    extra_params={
        "audio_file": "https://example.com/speech.mp3",
    },
)

response = generate_video(config, request)
```

**Text-to-speech lip sync** (no audio file needed):

```python
request = VideoGenerationRequest(
    prompt="lipsync",
    video="https://example.com/talking-head.mp4",
    extra_params={
        "text": "Hello, this is a lip sync demo!",
        "voice_id": "en_AOT",
        "voice_speed": 1.0,
    },
)
```

---

## Parameters

| Parameter | Required | Supported | Models | Notes |
|---|:---:|:---:|---|---|
| `prompt` | ✅ | ✅ | All | Text description of the video |
| `duration_seconds` | — | ✅ | Kling, Minimax, Veo3 | Integer seconds |
| `image_list` (first_frame) | — | ✅ | Kling, Luma | Start frame |
| `image_list` (last_frame) | — | ✅ | Luma | End frame |
| `image_list` (reference) | — | ✅ | Minimax | Reference image |
| `enhance_prompt` | — | ✅ | Minimax | As `prompt_optimizer` |
| `aspect_ratio` | — | ✅ | Luma, Veo3 | Passed through |
| `video` | — | ✅ | Kling Lip Sync | Input video URL for lip sync |
| `extra_params.audio_file` | — | ✅ | Kling Lip Sync | Audio file URL (.mp3/.wav/.m4a/.aac) |
| `extra_params.text` | — | ✅ | Kling Lip Sync | Text for TTS (if no audio) |
| `extra_params.voice_id` | — | ✅ | Kling Lip Sync | Voice ID for TTS (default: `en_AOT`) |
| `extra_params.voice_speed` | — | ✅ | Kling Lip Sync | TTS speech rate (0.8–2.0) |
| `extra_params.video_id` | — | ✅ | Kling Lip Sync | Kling video ID (alt to `video`) |
| `seed` | — | — | — | |
| `negative_prompt` | — | — | — | |
| `generate_audio` | — | — | — | |

---

## Supported Models

Model names on Replicate often include version hashes (e.g., `minimax/video-01:abc123`). Tarash strips the hash before registry lookup, then uses **prefix matching** so you can pass version-pinned names without changing config.

| Model ID / Prefix | Duration Options | Image-to-Video | Notes |
|---|---|:---:|---|
| `kwaivgi/kling-lip-sync` | — | — | Kling Lip Sync. Video + audio or text+TTS input. |
| `kwaivgi/kling` | 5s, 10s | ✅ | Kling v2.1. Image input **required**. |
| `luma/` | — | ✅ | Matches any `luma/*` model (Dream Machine) |
| `minimax/` | 6s, 10s | ✅ | Matches any `minimax/*` model |
| `hailuo/` | 6s, 10s | ✅ | Matches any `hailuo/*` model |
| `wan-video/` | — | ✅ | Wan video models |
| `google/veo-3` | 4s, 6s, 8s | ✅ | Google Veo 3 via Replicate |

**Example with version hash:**

```python
config = VideoGenerationConfig(
    provider="replicate",
    model="minimax/video-01:abc123def456",  # Hash stripped, matches "minimax/" prefix
    api_key="...",
)
```

---

## Provider-Specific Notes

**Kling Lip Sync supports two input modes.** Provide either `audio_file` (audio-driven) or `text` + `voice_id` (TTS-driven). Video input can be a URL via the `video` field, or a Kling-generated video via `extra_params.video_id`. Video should be 2–10 seconds, 720p–1080p, under 100MB.

**Kling v2.1 requires image input.** The `kwaivgi/kling` model only supports image-to-video. If no image is provided in `image_list`, a `ValidationError` is raised.

**Manual polling.** Unlike Fal's event streaming, Replicate uses a manual status polling loop. Tarash checks the prediction status every `poll_interval` seconds up to `max_poll_attempts` times. Terminal statuses: `succeeded`, `failed`, `canceled`.

**Version hash handling.** Model names with `:` are split on `:` to strip version hashes before registry lookup:

```
minimax/video-01:abc123  →  lookup: minimax/video-01  →  prefix match: minimax/
```

**Generic fallback.** For models not in the registry, Tarash applies generic mappers that pass `prompt`, `seed`, `negative_prompt`, and `aspect_ratio` through unchanged, and drops everything else.
