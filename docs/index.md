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

---

## Why Tarash Gateway

Sora, Veo3, Runway, Kling, Imagen — every provider ships a different API with different parameters, auth patterns, and polling logic. Tarash Gateway handles the translation. You write one integration and it runs on all of them.

---

## Features

<div class="grid cards" markdown>

-   **Fallback chains**

    If a provider fails or rate-limits, automatically continue with the next one.
    [Learn more →](guides/fallback-and-routing.md)

    ```python
    config = VideoGenerationConfig(
        provider="fal", model="fal-ai/veo3",
        fallback_configs=[
            VideoGenerationConfig(provider="replicate", model="google/veo-3"),
        ],
    )
    ```

-   **Async + progress callbacks**

    Every method has a sync and async variant. Get real-time status updates during generation.

    ```python
    async def on_progress(update):
        print(f"{update.status} — {update.progress_percent}%")

    response = await generate_video_async(config, request, on_progress=on_progress)
    ```

-   **Mock provider**

    Run your full integration locally without hitting any API or spending credits.
    [Learn more →](guides/mock.md)

    ```python
    from tarash.tarash_gateway.mock import MockConfig

    config = VideoGenerationConfig(
        provider="fal", model="fal-ai/veo3",
        mock=MockConfig(enabled=True),
    )
    ```

-   **Raw response access**

    Every response preserves the original provider payload for debugging and
    accessing provider-specific fields.

    ```python
    response.raw_response       # full provider JSON
    response.provider_metadata  # extra provider fields
    ```

-   **Pydantic v2**

    Every request and response is a typed model. No dict guessing. Full IDE support.

-   **Custom providers**

    Plug in your own provider or extend Fal with any model.
    [Learn more →](guides/custom-providers.md)

</div>

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

---

## Adding a new Fal model

Fal hosts hundreds of models. To add support for one that isn't registered yet,
open a GitHub issue — or run this in Claude Code:

```
/add-fal-model fal-ai/your-model-id
```

The skill fetches the model schema, generates field mappers, and writes tests automatically.

---

## Installation

```bash
pip install tarash-gateway[fal]       # single provider
pip install tarash-gateway[all]       # every provider
```

[:fontawesome-solid-rocket: Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[:fontawesome-brands-github: GitHub](https://github.com/vertexcover-io/tarash){ .md-button }
