# Providers Overview

Tarash Gateway provides a unified interface to generate videos and images across multiple AI providers. Switch providers by changing a single field in your config — no other code changes needed.

## Supported Providers

| Provider | Video | Image | Image-to-Video | Async | Install Extra |
|---|:---:|:---:|:---:|:---:|---|
| [OpenAI](openai.md) | ✅ | ✅ | ✅ | ✅ | `openai` |
| [Azure OpenAI](azure-openai.md) | ✅ | ✅ | ✅ | ✅ | `openai` |
| [Fal.ai](fal.md) | ✅ | ✅ | ✅ | ✅ | `fal` |
| [Google](google.md) | ✅ | ✅ | ✅ | ✅ | `veo3` |
| [Runway](runway.md) | ✅ | — | ✅ | ✅ | `runway` |
| [Replicate](replicate.md) | ✅ | — | ✅ | ✅ | `replicate` |
| [Stability AI](stability.md) | — | ✅ | — | ✅ | — |

## Switching Providers

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

request = VideoGenerationRequest(
    prompt="A cat playing piano, cinematic lighting",
    duration_seconds=5,
    aspect_ratio="16:9",
)

# Use Fal.ai
config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3", api_key="...")
response = generate_video(config, request)

# Switch to Runway — same request object, one line change
config = VideoGenerationConfig(provider="runway", model="gen3a_turbo", api_key="...")
response = generate_video(config, request)
```

## Model Quick Reference

### Video Models

| Model ID | Provider | Duration Options | Notes |
|---|---|---|---|
| `openai/sora-2` | OpenAI | 4s, 8s, 12s | Text & image-to-video |
| `openai/sora-2-pro` | OpenAI | 10s, 15s, 25s | Higher quality |
| `fal-ai/veo3` | Fal.ai | 4s, 6s, 8s | With audio |
| `fal-ai/veo3.1` | Fal.ai | 4s, 6s, 8s | Latest Veo |
| `fal-ai/minimax` | Fal.ai | 6s, 10s | Hailuo models |
| `fal-ai/kling-video/v2.6` | Fal.ai | 5s, 10s | Image-to-video |
| `fal-ai/kling-video/o1` | Fal.ai | 5s, 10s | Reference/edit |
| `fal-ai/sora-2` | Fal.ai | 4s, 8s, 12s | Sora via Fal |
| `veo-3.0-generate-preview` | Google | — | Native Veo3 |
| `gen3a_turbo` | Runway | — | Image-to-video only |
| `gen-3-alpha` | Runway | — | Text & image |
| `aleph` | Runway | — | Video-to-video |
| `kwaivgi/kling` | Replicate | 5s, 10s | Image-to-video |
| `minimax/` | Replicate | 6s, 10s | Prefix match |
| `google/veo-3` | Replicate | 4s, 6s, 8s | Via Replicate |

### Image Models

| Model ID | Provider | Notes |
|---|---|---|
| `gpt-image-1.5` | OpenAI | Latest image model |
| `dall-e-3` | OpenAI | High quality |
| `dall-e-2` | OpenAI | Fast, cheaper |
| `fal-ai/flux-pro` | Fal.ai | Flux family |
| `imagen-3.0-generate-001` | Google | Imagen 3 |
| `gemini-2.5-flash-image` | Google | Gemini image |
| `sd3.5-large` | Stability AI | SD 3.5 |
| `sd3.5-large-turbo` | Stability AI | Faster variant |
| `sd3.5-medium` | Stability AI | Balanced |
| `stable-image-ultra` | Stability AI | Highest quality |
| `stable-image-core` | Stability AI | Fast |

## Fallback Chains

Configure automatic fallback to a backup provider if the primary fails:

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    api_key="FAL_KEY",
    fallback_configs=[
        VideoGenerationConfig(
            provider="replicate",
            model="google/veo-3",
            api_key="REPLICATE_KEY",
        ),
    ],
)
```

See each provider page for detailed configuration options, model lists, and provider-specific notes.
