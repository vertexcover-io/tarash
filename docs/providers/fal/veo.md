# Veo (via Fal.ai)

Google's Veo3 and Veo3.1 models hosted on Fal.ai.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    prompt="A hummingbird hovering over a flower, slow motion",
    duration_seconds=6,
    aspect_ratio="16:9",
    generate_audio=True,
    seed=42,
)

response = generate_video(config, request)
print(response.video)
```

---

## Supported Models

| Model prefix | Duration | Image-to-Video | Notes |
|---|---|:---:|---|
| `fal-ai/veo3` | `4s`, `6s`, `8s` | ✅ | Audio; first/last frame; extend-video |
| `fal-ai/veo3.1` | `4s`, `6s`, `7s`, `8s` | ✅ | Fast variant via `/fast`; extend via `/fast/extend-video` |

---

## Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | |
| `duration_seconds` | — | ✅ | `4`, `6`, `8` for veo3; `4`, `6`, `7`, `8` for veo3.1 |
| `aspect_ratio` | — | ✅ | Passed through directly |
| `resolution` | — | ✅ | e.g. `"720p"`, `"1080p"` |
| `image_list` (reference) | — | ✅ | Image-to-video |
| `image_list` (first_frame) | — | ✅ | First frame pinning |
| `image_list` (last_frame) | — | ✅ | Last frame pinning |
| `video` | — | ✅ | Video extend |
| `seed` | — | ✅ | Reproducibility |
| `negative_prompt` | — | ✅ | |
| `generate_audio` | — | ✅ | |

---

## Video Extend

```python
# Extend an existing video with veo3.1 fast
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast/extend-video",
)

request = VideoGenerationRequest(
    prompt="Continue the scene in winter",
    duration_seconds=7,
    video="https://example.com/source.mp4",
)
```

---

## Provider-Specific Notes

**Veo3.1 vs veo3:** Use `fal-ai/veo3.1` for the latest model. Both use the same field mappers, so switching is transparent. `fal-ai/veo3.1/fast` also matches the `fal-ai/veo3.1` prefix.

**Extend-video constraints:** The `fal-ai/veo3.1/fast/extend-video` endpoint has stricter limits:
- Duration: 7s only
- Resolution: 720p only
- Requires both `prompt` and a `video` input

---