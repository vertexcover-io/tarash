# Providers

Tarash Gateway provides a unified interface to generate videos and images across multiple AI providers.
Change `provider` and `model` in your config — nothing else in your code changes.

## Supported providers

| Provider | Video | Image | Image-to-Video | Async | Install extra |
|---|:---:|:---:|:---:|:---:|---|
| [OpenAI](openai.md) | ✅ | ✅ | ✅ | ✅ | `openai` |
| [Azure OpenAI](azure-openai.md) | ✅ | ✅ | ✅ | ✅ | `openai` |
| [Fal.ai](fal.md) | ✅ | ✅ | ✅ | ✅ | `fal` |
| [Google](google.md) | ✅ | ✅ | ✅ | ✅ | `veo3` |
| [Runway](runway.md) | ✅ | — | ✅ | ✅ | `runway` |
| [Replicate](replicate.md) | ✅ | — | ✅ | ✅ | `replicate` |
| [Stability AI](stability.md) | — | ✅ | — | ✅ | — |

## Switching providers

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

request = VideoGenerationRequest(
    prompt="A cat playing piano, cinematic lighting",
    duration_seconds=5,
    aspect_ratio="16:9",
)

# Use Fal.ai
config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3")
response = generate_video(config, request)

# Switch to Runway — same request, one line change
config = VideoGenerationConfig(provider="runway", model="gen3a_turbo")
response = generate_video(config, request)
```

## Fallback chains

Configure automatic fallback to a backup provider if the primary fails or rate-limits:

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    fallback_configs=[
        VideoGenerationConfig(
            provider="replicate",
            model="google/veo-3",
        ),
    ],
)
```

See the [Fallback & Routing guide](../guides/fallback-and-routing.md) for full details on retry behavior, execution metadata, and chaining multiple providers.

## Adding a new Fal model

Fal hosts hundreds of models. To add one that isn't registered:

```
/add-fal-model fal-ai/your-model-id
```

Or [open a GitHub issue](https://github.com/vertexcover-io/tarash/issues).
