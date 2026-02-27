# Fal.ai

!!! tip "Model not listed? Add it in seconds."
    **Option 1 — GitHub issue** (no local setup needed):
    [Open an "Add Fal model" issue](https://github.com/vertexcover-io/tarash/issues/new?template=add-fal-model.yml) — a bot picks it up, runs the skill, and opens a PR automatically.

    **Option 2 — Claude Code skill** (in your terminal):
    First, install the skill into your project:

    ```bash
    mkdir -p .claude/skills/add-fal-model
    curl -o .claude/skills/add-fal-model/SKILL.md \
      https://raw.githubusercontent.com/vertexcover-io/tarash/master/.claude/skills/add-fal-model/SKILL.md
    ```

    Or [view the skill file on GitHub](https://github.com/vertexcover-io/tarash/blob/master/.claude/skills/add-fal-model/SKILL.md) to download it manually. Then run:

    ```
    /add-fal-model fal-ai/your-model-id
    ```

    Both paths fetch the model schema from fal.ai, generate field mappers, register the model, and write unit + e2e tests automatically.

Fal.ai is a serverless AI inference platform hosting multiple video and image generation models including Veo3, Sora, Minimax (Hailuo), Kling, and Flux.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast",
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

## Configuration

| Field | Type | Required | Default | Description |
|---|---|:---:|---|---|
| `provider` | `str` | ✅ | — | Must be `"fal"` |
| `model` | `str` | ✅ | — | Model ID, e.g. `fal-ai/veo3.1` |
| `api_key` | `str | None` | ✅ | — | Fal API key |
| `timeout` | `int` | — | `600` | Request timeout in seconds |
| `max_poll_attempts` | `int` | — | `120` | Max polling iterations |
| `poll_interval` | `int` | — | `5` | Seconds between polls |

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast",
    api_key="...",
    timeout=600,
    max_poll_attempts=120,
    poll_interval=5,
)
```

---

## Supported Models

| Family | Page | Video Models | Image |
|---|---|---|:---:|
| [Veo](veo.md) | Veo | `fal-ai/veo3`, `fal-ai/veo3.1` | — |
| [Kling](kling.md) | Kling | `fal-ai/kling-video/v2.6`, `fal-ai/kling-video/o1` | — |
| [Minimax](minimax.md) | Minimax | `fal-ai/minimax` | — |
| [Wan](wan.md) | Wan | `wan/v2.6/`, `fal-ai/wan-25-preview/` | — |
| [Sora](sora.md) | Sora | `fal-ai/sora-2` | — |
| [Seedance](seedance.md) | Seedance | `fal-ai/bytedance/seedance` | — |
| [Pixverse](pixverse.md) | Pixverse | `fal-ai/pixverse/v5`, `fal-ai/pixverse/v5.5` | — |
| [Grok Imagine](grok-imagine-image.md) | Grok Imagine | — | ✅ |
| Generic | — | Any other `fal-ai/*` | ✅ |

Model lookup uses **prefix matching**: `fal-ai/veo3.1/fast` matches the `fal-ai/veo3.1` registry entry, so any sub-variant automatically inherits the right field mappers.

Any Fal model not in this table gets **generic mappers** (prompt passthrough + common fields). For full support with model-specific parameters, use `/add-fal-model` in Claude Code.


---

## Provider Notes

**Event-based polling:** Fal uses native event streaming (`iter_events()`), which gives real-time progress updates without constant HTTP polling.

**Extend-video constraints:** The `fal-ai/veo3.1/fast/extend-video` endpoint has stricter limits:
- Duration: 7s only
- Resolution: 720p only
- Requires both `prompt` and a `video` input
