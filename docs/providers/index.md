# Providers

Tarash Gateway provides a unified interface to generate videos and images across multiple AI providers.
Change `provider` and `model` in your config — nothing else in your code changes.

## Supported providers

### Video & Image

| Provider | Video | Image | Image-to-Video | Install extra |
|---|:---:|:---:|:---:|---|
| [OpenAI](openai.md) | ✅ | ✅ | ✅ | `openai` |
| [Azure OpenAI](openai.md#azure-openai) | ✅ | ✅ | ✅ | `openai` |
| [Fal.ai](fal.md) | ✅ | ✅ | ✅ | `fal` |
| [Google](google.md) | ✅ | ✅ | ✅ | `veo3` |
| [Runway](runway.md) | ✅ | — | ✅ | `runway` |
| [Replicate](replicate.md) | ✅ | — | ✅ | `replicate` |
| [xAI](xai.md) | ✅ | ✅ | ✅ | `xai` |
| [Luma AI](luma.md) | ✅ | ✅ | ✅ | `luma` |
| [Stability AI](stability.md) | — | ✅ | — | — |

### Audio (TTS / STS)

| Provider | TTS | STS | Install extra |
|---|:---:|:---:|---|
| [Fal.ai](fal.md) | ✅ | — | `fal` |
| [ElevenLabs](elevenlabs.md) | ✅ | ✅ | `elevenlabs` |
| [Cartesia](cartesia.md) | ✅ | ✅ | `cartesia` |
| [Sarvam AI](sarvam.md) | ✅ | — | `sarvam` |
| [Hume AI](hume.md) | ✅ | — | `hume` |

---

## Switching providers

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

request = VideoGenerationRequest(
    prompt="A cat playing piano, cinematic lighting",
    duration_seconds=4,
    aspect_ratio="16:9",
)

# Use Fal.ai
config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3.1/fast", api_key="YOUR_FAL_KEY")
response = generate_video(config, request)

# Switch to Runway — same request, one line change
config = VideoGenerationConfig(provider="runway", model="gen4_turbo", api_key="YOUR_RUNWAY_KEY")
response = generate_video(config, request)
```

For the full list of model IDs per provider, click any provider name in the table above.

For automatic failover between providers, see the [Fallback & Routing guide](../guides/fallback-and-routing.md).
