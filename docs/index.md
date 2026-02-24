# Tarash Gateway

**One API. Every AI video and image provider.**

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3")
request = VideoGenerationRequest(prompt="A cat playing piano, cinematic lighting")
response = generate_video(config, request)
print(response.video)  # → URL to generated video
```

Switch providers by changing two words:

```python
config = VideoGenerationConfig(provider="runway", model="gen3a_turbo")
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
    model="fal-ai/veo3",
    fallback_configs=[
        VideoGenerationConfig(provider="replicate", model="google/veo-3"),
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
    model="fal-ai/veo3",
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

| Provider | Video | Image | Install extra |
|---|:---:|:---:|---|
| [Fal.ai](providers/fal.md) | ✅ | ✅ | `fal` |
| [OpenAI](providers/openai.md) | ✅ | ✅ | `openai` |
| [Azure OpenAI](providers/azure-openai.md) | ✅ | ✅ | `openai` |
| [Google](providers/google.md) | ✅ | ✅ | `veo3` |
| [Runway](providers/runway.md) | ✅ | — | `runway` |
| [Replicate](providers/replicate.md) | ✅ | — | `replicate` |
| [Stability AI](providers/stability.md) | — | ✅ | — |
