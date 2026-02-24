# Fal.ai

!!! tip "Adding a new Fal model? There's a skill for that. <span style='background:#1565C0;color:#fff;padding:1px 7px;border-radius:4px;font-size:0.72em;font-weight:700;letter-spacing:0.04em;vertical-align:middle'>BETA</span>"
    Fal hosts hundreds of models. To add one that isn't in the registry yet, run this in Claude Code:

    ```
    /add-fal-model fal-ai/your-model-id
    ```

    It fetches the model schema from fal.ai, generates field mappers, registers the model, and writes unit + e2e tests — all in one shot.

Fal.ai is a serverless AI inference platform that hosts multiple video and image generation models including Veo3, Sora, Minimax (Hailuo), Kling, and Flux.

## Capabilities

| Feature | Supported |
|---|:---:|
| Video generation | ✅ |
| Image generation | ✅ |
| Image-to-video | ✅ |
| First/last frame | ✅ |
| Video extend/remix | ✅ |
| Async | ✅ |
| Progress callbacks | ✅ |

## Configuration

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    api_key="...",          # Required: Fal API key
    base_url=None,          # Optional: override Fal endpoint
    timeout=600,
    max_poll_attempts=120,
    poll_interval=5,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | Fal API key |
| `base_url` | `str \| None` | `None` | Override Fal endpoint |
| `timeout` | `int` | `600` | Request timeout in seconds |
| `max_poll_attempts` | `int` | `120` | Max polling iterations |
| `poll_interval` | `int` | `5` | Seconds between status checks |

## Video Models

Model lookup uses **prefix matching**: `fal-ai/veo3.1/fast` matches the `fal-ai/veo3.1` registry entry,
so any sub-variant automatically inherits the right field mappers.

| Model prefix | Duration | Image-to-Video | Notes |
|---|---|:---:|---|
| `fal-ai/veo3` | 4s, 6s, 8s | ✅ | Audio; first/last frame; extend-video |
| `fal-ai/veo3.1` | 4s, 6s, 7s, 8s | ✅ | Latest Veo; fast variant via `/fast`; extend-video via `/fast/extend-video` |
| `fal-ai/minimax` | 6s, 10s | ✅ | Hailuo series; prompt optimizer support |
| `fal-ai/kling-video/v2.6` | 5s, 10s | ✅ | Motion control, cfg_scale, last-frame pinning |
| `fal-ai/kling-video/o1` | 5s, 10s | ✅ | Reference-to-video, video edit, start/end frame |
| `fal-ai/sora-2` | 4s, 8s, 12s | ✅ | Sora via Fal; remix via `/video-to-video/remix` |
| `wan/v2.6/` | configurable | ✅ | Wan v2.6; text, image, reference-to-video |
| `fal-ai/wan-25-preview/` | configurable | ✅ | Wan v2.5 preview |
| `fal-ai/wan/v2.2-14b/animate/` | — | ✅ | Wan animate: video+image motion control |
| `fal-ai/bytedance/seedance` | 2s–12s | ✅ | ByteDance Seedance v1/v1.5; reference-to-video |
| `fal-ai/pixverse/v5` | 5s, 8s, 10s | ✅ | Pixverse v5; transition, effects, swap |
| `fal-ai/pixverse/v5.5` | 5s, 8s, 10s | ✅ | Pixverse v5.5; same API as v5 |
| `fal-ai/pixverse/swap` | — | ✅ | Pixverse swap variant |
| Any other `fal-ai/*` | — | ✅ | Generic field mappers (prompt, seed, aspect_ratio) |

Any Fal model not in this table gets **generic mappers** (prompt passthrough + common fields). For full support with model-specific parameters, [add the model](https://github.com/vertexcover-io/tarash/issues) or use `/add-fal-model`.

## Image Models

Fal hosts many Flux-family image models. Use any Fal image model path as the `model` field in `ImageGenerationConfig`.

| Model ID (prefix) | Notes |
|---|---|
| `fal-ai/flux-pro` | Flux Pro |
| `fal-ai/flux/dev` | Flux Dev |
| `fal-ai/flux-realism` | Photorealism LoRA |
| `fal-ai/flux-pro/kontext` | Context-aware editing |

For unlisted Fal image models, Tarash applies generic field mappers (prompt passthrough + seed).

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
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

### Async with progress tracking

```python
import asyncio
from tarash.tarash_gateway import generate_video_async
from tarash.tarash_gateway.models import VideoGenerationUpdate

async def main():
    async def on_progress(update: VideoGenerationUpdate):
        print(f"[{update.status}] {update.message}")

    response = await generate_video_async(config, request, on_progress=on_progress)
    print(response.video)

asyncio.run(main())
```

## Supported Request Parameters

### Video

| Parameter | Supported | Models | Notes |
|---|:---:|---|---|
| `prompt` | ✅ | All | Required |
| `duration_seconds` | ✅ | All | See model table for allowed values |
| `aspect_ratio` | ✅ | veo3, kling-o1, sora-2 | Passed through directly |
| `resolution` | ✅ | veo3 | e.g. `"720p"`, `"1080p"` |
| `image_list` (reference) | ✅ | veo3, minimax, kling | Image-to-video |
| `image_list` (first_frame) | ✅ | veo3, kling-o1 | First frame pinning |
| `image_list` (last_frame) | ✅ | veo3, kling-v2.6, kling-o1 | Last frame pinning |
| `video` | ✅ | veo3, kling-o1, sora-2 | Video extend/edit |
| `seed` | ✅ | veo3 | Reproducibility |
| `negative_prompt` | ✅ | veo3, kling-v2.6 | |
| `generate_audio` | ✅ | veo3, kling-v2.6 | |
| `enhance_prompt` | ✅ | minimax | As `prompt_optimizer` |

### Extra params (model-specific)

Pass additional Fal API parameters via `extra_params`:

```python
# Kling v2.6 motion control
request = VideoGenerationRequest(
    prompt="...",
    extra_params={
        "cfg_scale": 0.5,
        "character_orientation": "front",
        "keep_original_sound": True,
    },
)

# Sora-2 remix (requires fal-ai/sora-2/video-to-video/remix endpoint)
request = VideoGenerationRequest(
    prompt="Same scene in winter",
    extra_params={"video_id": "video_abc123"},
)

# Kling O1 reference-to-video elements
request = VideoGenerationRequest(
    prompt="...",
    extra_params={
        "elements": [{"frontal_image_url": "https://..."}]
    },
)
```

## Provider-Specific Notes

**Async client caching:** Fal's async client is **not cached** — a fresh instance is created per call. This prevents "Event Loop closed" errors when running in environments that recreate event loops between calls. Sync clients are cached per `(api_key, base_url)`.

**Event-based polling:** Fal uses native event streaming (`iter_events()`), which gives real-time progress updates without constant HTTP polling.

**Extend-video constraints:** The `fal-ai/veo3.1/fast/extend-video` endpoint has stricter limits:
- Duration: 7s only
- Resolution: 720p only
- Requires both `prompt` and a `video` input

**Veo3.1 vs veo3:** Use `fal-ai/veo3.1` for the latest model. Both use the same field mappers, so switching is transparent. `fal-ai/veo3.1/fast` also matches the `fal-ai/veo3.1` prefix.

