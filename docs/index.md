# Tarash Gateway

**One API. Every AI video and image provider.**

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3.1/fast")
request = VideoGenerationRequest(prompt="A cat playing piano, cinematic lighting")
response = generate_video(config, request)
print(response.video)  # → URL to generated video
```

Switch providers by changing two words:

```python
config = VideoGenerationConfig(provider="runway", model="gen4_turbo")
```

```bash
pip install tarash-gateway[fal]     # single provider
pip install tarash-gateway[all]     # every provider
```

[:fontawesome-solid-rocket: Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[:fontawesome-brands-github: GitHub](https://github.com/vertexcover-io/tarash){ .md-button }

---

## Why Tarash Gateway

Sora, Veo3, Runway, Kling, Imagen — every provider ships a different API, different parameters, and different polling logic. Tarash Gateway handles the translation. You write one integration and it runs on all of them.

---

## Features

### Fallback chains

If a provider fails or rate-limits, Tarash Gateway automatically tries the next one in your chain:

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast",
    fallback_configs=[
        VideoGenerationConfig(provider="replicate", model="google/veo-3.1"),
        VideoGenerationConfig(provider="openai", model="openai/sora-2"),
    ],
)
response = generate_video(config, request)
# Fal → Replicate → OpenAI — first success wins
```

[Fallback & Routing guide →](guides/fallback-and-routing.md)

### Mock provider

Test your full pipeline locally without hitting any API or spending credits:

```python
from tarash.tarash_gateway.mock import MockConfig

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast",
    mock=MockConfig(enabled=True),
)
response = generate_video(config, request)
print(response.is_mock)   # True — no real API call was made
```

[Mock Provider guide →](guides/mock.md)

### Progress callbacks

Track generation in real time with sync or async callbacks:

```python
def on_progress(update):
    print(f"[{update.status}] {update.progress_percent}%")

response = generate_video(config, request, on_progress=on_progress)
```

### Raw response access

Every response keeps the original provider JSON — useful for debugging or reading provider-specific fields not in the standard interface:

```python
response = generate_video(config, request)
print(response.raw_response)       # original provider JSON, unmodified
print(response.provider_metadata)  # extra provider-specific fields
```

### Sync + async

Every call has both a sync and async variant:

```python
response = generate_video(config, request)            # sync
response = await generate_video_async(config, request)  # async
```


---

## Providers

### Video Generation

<div class="provider-table" markdown="1">

| Model | Variants | Provider(s) |
|---|---|---|
| **Veo 3** | `fal-ai/veo3`<br>`fal-ai/veo3.1/fast`<br>`fal-ai/veo3.1/fast/image-to-video`<br>`fal-ai/veo3.1/fast/first-last-frame-to-video`<br>`fal-ai/veo3.1/fast/extend-video`<br>`veo-3.0-generate-001` (Google)<br>`veo-3.0-generate-preview` (Google)<br>`google/veo-3` (Replicate)<br>`google/veo-3.1` (Replicate) | [Fal.ai](providers/fal/index.md) · [Google](providers/google.md) · [Replicate](providers/replicate.md) |
| **Kling** | `fal-ai/kling-video/v2.6`<br>`fal-ai/kling-video/v2.6/standard/motion-control`<br>`fal-ai/kling-video/v3/pro/text-to-video`<br>`fal-ai/kling-video/v3/pro/image-to-video`<br>`fal-ai/kling-video/v3/standard/text-to-video`<br>`fal-ai/kling-video/v3/standard/image-to-video`<br>`fal-ai/kling-video/o1/image-to-video`<br>`fal-ai/kling-video/o1/standard/reference-to-video`<br>`fal-ai/kling-video/o1/standard/video-to-video/edit`<br>`fal-ai/kling-video/o1/standard/video-to-video/reference`<br>`fal-ai/kling-video/o3/pro/text-to-video`<br>`fal-ai/kling-video/o3/pro/image-to-video`<br>`fal-ai/kling-video/o3/standard/reference-to-video`<br>`kwaivgi/kling-v2.1` (Replicate) | [Fal.ai](providers/fal/index.md) · [Replicate](providers/replicate.md) |
| **Sora** | `sora`<br>`fal-ai/sora-2/text-to-video`<br>`fal-ai/sora-2/image-to-video` | [OpenAI](providers/openai.md) · [Fal.ai](providers/fal/index.md) |
| **Minimax** | `fal-ai/minimax/video-01`<br>`fal-ai/minimax/hailuo-02-fast/image-to-video`<br>`minimax/video-01` (Replicate) | [Fal.ai](providers/fal/index.md) · [Replicate](providers/replicate.md) |
| **WAN** | `fal-ai/wan-25-preview/text-to-video`<br>`fal-ai/wan-25-preview/image-to-video`<br>`fal-ai/wan/v2.2-14b/animate/move`<br>`fal-ai/wan/v2.2-a14b/image-to-video`<br>`fal-ai/wan/v2.2-a14b/image-to-video/lora`<br>`fal-ai/wan/v2.2-a14b/text-to-video/lora`<br>`fal-ai/wan/v2.2-a14b/video-to-video`<br>| [Fal.ai](providers/fal/index.md) · [Replicate](providers/replicate.md) |
| **Seedance** | `fal-ai/bytedance/seedance/v1.5/pro/text-to-video`<br>`fal-ai/bytedance/seedance/v1/pro/image-to-video`<br>`fal-ai/bytedance/seedance/v1/lite/reference-to-video` | [Fal.ai](providers/fal/index.md) |
| **Pixverse** | `fal-ai/pixverse/v5.5/text-to-video`<br>`fal-ai/pixverse/v5.5/image-to-video`<br>`fal-ai/pixverse/v5/text-to-video` | [Fal.ai](providers/fal/index.md) |
| **Runway** | `gen4_turbo`<br>`gen4_aleph`<br>`veo3.1`<br>`veo3.1_fast` | [Runway](providers/runway.md) |

</div>

### Image Generation

<div class="provider-table" markdown="1">

| Model | Variants | Provider(s) |
|---|---|---|
| **DALL-E** | `dall-e-3`<br>`dall-e-2` | [OpenAI](providers/openai.md) |
| **GPT Image** | `gpt-image-1.5` | [OpenAI](providers/openai.md) |
| **Imagen 3** | `imagen-3.0-generate-001`<br>`imagen-3.0-generate-002`<br>`imagen-3.0-fast-generate-001` | [Google](providers/google.md) |
| **Gemini Image** | `gemini-2.5-flash-image-preview`<br>`gemini-3-pro-image-preview` | [Google](providers/google.md) |
| **Flux** | `fal-ai/flux/dev`<br>`fal-ai/flux/schnell`<br>`fal-ai/flux-2`<br>`fal-ai/flux-2/pro`<br>`fal-ai/flux-2/dev`<br>`fal-ai/flux-2/flex`<br>`fal-ai/flux-pro/v1.1-ultra`<br>`fal-ai/flux-pro/v1.1-raw` | [Fal.ai](providers/fal/index.md) |
| **Stable Diffusion** | `sd3.5-large`<br>`sd3.5-medium`<br>`sd3.5-large-turbo` | [Stability AI](providers/stability.md) |
| **Stable Image** | `stable-image-ultra`<br>`stable-image-core` | [Stability AI](providers/stability.md) |
| **Recraft** | `fal-ai/recraft-v3`<br>`fal-ai/recraft` | [Fal.ai](providers/fal/index.md) |
| **Ideogram** | `fal-ai/ideogram` | [Fal.ai](providers/fal/index.md) |
| **Z-Image Turbo** | `fal-ai/z-image/turbo` | [Fal.ai](providers/fal/index.md) |

</div>