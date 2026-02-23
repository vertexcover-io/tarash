# Tarash

**AI video generation. Any provider. One API.**

Pick a provider and generate:

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3", api_key="...")
request = VideoGenerationRequest(
    prompt="A cat playing piano, cinematic lighting",
    duration_seconds=6,
    aspect_ratio="16:9",
)
response = generate_video(config, request)
```

Switch providers — nothing else changes:

```python
config = VideoGenerationConfig(provider="runway", model="gen3a_turbo", api_key="...")
```

---

Sora, Veo3, Runway, Kling, Imagen — every provider ships a different API with
different parameters, auth patterns, and polling logic. Tarash handles the translation.
You write one integration and it runs on all of them.

---

## Providers

| Provider | Video | Image |
|---|:---:|:---:|
| Fal.ai | ✅ | ✅ |
| OpenAI | ✅ | ✅ |
| Azure OpenAI | ✅ | ✅ |
| Google | ✅ | ✅ |
| Runway | ✅ | — |
| Replicate | ✅ | — |
| Stability AI | — | ✅ |

---

## What's included

**Fallback chains** — if a provider fails or rate-limits, automatically continue with the next one.

```python
config = VideoGenerationConfig(
    provider="fal", model="fal-ai/veo3", api_key="...",
    fallback_configs=[
        VideoGenerationConfig(provider="replicate", model="google/veo-3", api_key="..."),
    ],
)
```

**Async support** — every method has a sync and async variant with the same signature.

```python
response = await generate_video_async(config, request, on_progress=callback)
```

**Progress callbacks** — real-time status during long generations.

**Mock provider** — run your full integration locally without hitting an API or spending credits.

**Pydantic v2** — every request and response is a typed model. No dict guessing.

---

## Installation

```bash
pip install tarash-gateway[fal]       # single provider
pip install tarash-gateway[all]       # every provider
```

[:fontawesome-solid-rocket: Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[:fontawesome-brands-github: GitHub](https://github.com/vertexcover-io/tarash){ .md-button }
