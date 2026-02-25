# Fal.ai

!!! tip "Model not listed? Add it in seconds."
    **Option 1 — GitHub issue** (no local setup needed):
    [Open an "Add Fal model" issue](https://github.com/vertexcover-io/tarash/issues/new?template=add-fal-model.yml) — a bot picks it up, runs the skill, and opens a PR automatically.

    **Option 2 — Claude Code skill** (in your terminal):
    ```
    /add-fal-model fal-ai/your-model-id
    ```

    Both paths fetch the model schema from fal.ai, generate field mappers, register the model, and write unit + e2e tests automatically.

Fal.ai is a serverless AI inference platform that hosts multiple video and image generation models including Veo3, Sora, Minimax (Hailuo), Kling, and Flux.

---

## Image Generation

### Quick Example

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="fal",
    model="fal-ai/flux-pro",
    api_key="YOUR_FAL_KEY",
)

request = ImageGenerationRequest(
    prompt="A futuristic city skyline at golden hour, photorealistic",
    aspect_ratio="16:9",
)

response = generate_image(config, request)
print(response.images[0])
```

### Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | Text description of the image |
| `aspect_ratio` | — | ✅ | Passed through to Fal |
| `seed` | — | ✅ | Reproducibility |
| `negative_prompt` | — | — | Not supported |
| `n` | — | — | Not supported |

### Supported Models

Fal hosts many Flux-family image models. Use any Fal image model path as the `model` field in `ImageGenerationConfig`.

| Model ID (prefix) | Notes |
|---|---|
| `fal-ai/flux-pro` | Flux Pro |
| `fal-ai/flux/dev` | Flux Dev |
| `fal-ai/flux-realism` | Photorealism LoRA |
| `fal-ai/flux-pro/kontext` | Context-aware editing |

For unlisted Fal image models, Tarash Gateway applies generic field mappers (prompt passthrough + seed).

---

## Video Generation

### Quick Example

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

### Parameters

| Parameter | Required | Supported | Models | Notes |
|---|:---:|:---:|---|---|
| `prompt` | ✅ | ✅ | All | Text description of the video |
| `duration_seconds` | — | ✅ | All | See model table for allowed values |
| `aspect_ratio` | — | ✅ | veo3, kling-o1, sora-2 | Passed through directly |
| `resolution` | — | ✅ | veo3 | e.g. `"720p"`, `"1080p"` |
| `image_list` (reference) | — | ✅ | veo3, minimax, kling | Image-to-video |
| `image_list` (first_frame) | — | ✅ | veo3, kling-o1 | First frame pinning |
| `image_list` (last_frame) | — | ✅ | veo3, kling-v2.6, kling-o1 | Last frame pinning |
| `video` | — | ✅ | veo3, kling-o1, sora-2 | Video extend/edit |
| `seed` | — | ✅ | veo3 | Reproducibility |
| `negative_prompt` | — | ✅ | veo3, kling-v2.6 | |
| `generate_audio` | — | ✅ | veo3, kling-v2.6 | |
| `enhance_prompt` | — | ✅ | minimax | As `prompt_optimizer` |

### Supported Models

Model lookup uses **prefix matching**: `fal-ai/veo3.1/fast` matches the `fal-ai/veo3.1` registry entry, so any sub-variant automatically inherits the right field mappers.

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

Any Fal model not in this table gets **generic mappers** (prompt passthrough + common fields). For full support with model-specific parameters, use `/add-fal-model` in Claude Code.

### Async with Progress Tracking

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

### Video Remix

To remix an existing Sora-2 video, pass the original `video_id` via `extra_params`. Fal routes this to the `fal-ai/sora-2/video-to-video/remix` endpoint.

```python
request = VideoGenerationRequest(
    prompt="Same scene in winter",
    extra_params={"video_id": "video_abc123"},
)
```

### Video Extend

Extend an existing video using the `fal-ai/veo3.1/fast/extend-video` endpoint. Pass the source video via the `video` field and a continuation prompt.

```python
request = VideoGenerationRequest(
    prompt="The bird continues flying over the mountains",
    video="https://example.com/original-clip.mp4",
)

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast/extend-video",
    api_key="YOUR_FAL_KEY",
)
```

!!! note "Extend-video constraints"
    - Duration: 7s only
    - Resolution: 720p only
    - Requires both `prompt` and a `video` input

---

## Provider-Specific Notes

**Async client caching:** Fal's async client is **not cached** — a fresh instance is created per call. This prevents "Event Loop closed" errors when running in environments that recreate event loops between calls. Sync clients are cached per `(api_key, base_url)`.

**Event-based polling:** Fal uses native event streaming (`iter_events()`), which gives real-time progress updates without constant HTTP polling.
