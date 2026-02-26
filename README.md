# Tarash

**A monorepo of Python SDKs for AI-powered media generation.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Goal

The AI media generation landscape is fragmented. Every provider — Fal.ai, Runway, Replicate, OpenAI, Google — ships its own SDK, its own parameter names, its own response shapes, and its own error formats. Building a product that supports more than one provider means writing and maintaining N separate integrations.

Tarash solves this. It is a suite of SDKs that abstract AI media generation behind unified interfaces — similar to what LiteLLM does for text generation. You write one integration. Swap providers by changing a config value. Get consistent responses, errors, and progress events regardless of who is generating on the backend.

---

## Packages

### `tarash-gateway` — Video & Image Generation

**Status: Implemented**

Unified SDK for AI video and image generation. One API surface across all providers, with automatic parameter mapping, orchestrated fallback chains, real-time progress tracking, and production-ready error handling.

**Video generation providers:**

| Provider | SDK Extra | Models |
|----------|-----------|--------|
| Fal.ai | `fal` | Veo3, Veo3.1, Minimax, Kling v2.6, Wan, ByteDance Seedance, Sora-2 |
| OpenAI | `openai` | Sora-2 Turbo |
| Azure OpenAI | `openai` | Sora-2 |
| Replicate | `replicate` | Kling, Luma, Minimax, Wan, Veo3 |
| Runway | `runway` | Gen-3 Alpha |
| Google (Vertex AI) | `google` | Veo3 |
| Luma | `lumaai` | Dream Machine |

**Image generation providers:**

| Provider | SDK Extra | Models |
|----------|-----------|--------|
| Stability AI | `stability` | Stable Diffusion 3.5 Large, Stable Image Ultra |
| Google | `google` | Imagen 3, Nano Banana |

**Capabilities:**
- Text-to-video, image-to-video, video-to-video, extend, remix
- Text-to-image with multiple style and resolution options
- Async/sync support — full `async/await` or blocking calls
- Progress callbacks — real-time updates during long-running generation
- Provider fallback — automatic failover across a prioritized provider chain
- Type safety — Pydantic v2 models with full IDE autocomplete
- Mock generation — deterministic fake responses for testing without API calls
- Structured logging — automatic credential redaction, request tracing
- Field mapper registry — declarative parameter translation per model, with prefix matching

```python
from tarash_gateway import VideoGenerationConfig, VideoGenerationRequest, generate_video

config = VideoGenerationConfig(
    model="fal-ai/veo3.1/fast",
    provider="fal",
    api_key="your-api-key",
)

request = VideoGenerationRequest(
    prompt="Sunset over mountains, cinematic",
    aspect_ratio="16:9",
    duration_seconds=5,
)

response = generate_video(config, request)
print(response.video)  # URL to generated video
```

Switching providers requires only changing `model` and `provider` in the config. The request and the rest of your code stay the same.

---

### `tarash-captions` — Caption Generation

**Status: Implemented (OpenAI provider)**

SDK for AI-powered video caption generation. Follows the same architecture as `tarash-gateway` — provider-agnostic config, unified request/response models, sync and async generation.

**Implemented providers:**
- **OpenAI** — `gpt-4o` (default), configurable model

**Planned providers:**
- Additional LLM providers following the same `CaptionProviderHandler` protocol

```python
from tarash.captions import CaptionConfig, CaptionRequest, generate_caption

config = CaptionConfig(provider="openai", api_key="your-api-key", model="gpt-4o")
request = CaptionRequest(prompt="A chef preparing pasta in a busy kitchen", language="en")

response = generate_caption(config, request)
print(response.text)
```

---

### `tarash-scene-detector` — Scene Detection

**Status: Early stage**

Scene detection for video files using [PySceneDetect](https://www.scenedetect.com/) with OpenCV. Identifies cut points and scene boundaries in video content — useful as a preprocessing step for video generation workflows.

---

## Architecture

All packages share the same design principles:

- **Protocol-based providers** — every provider implements the same handler interface; adding a new provider is a drop-in
- **Immutable config** — frozen Pydantic models, safe to share across threads and async tasks
- **Rich exceptions** — every error includes `provider`, `model`, `request_id`, and the raw provider response for debugging
- **Dual sync/async** — all public APIs have both `generate_*` and `generate_*_async` variants
- **Singleton handlers** — provider handler instances are cached; client connections are reused across calls

---

## Development

**Requirements:** Python 3.12+, [`uv`](https://docs.astral.sh/uv/)

```bash
git clone <repository-url>
cd tarash
uv sync
```

Run tests for a specific package:

```bash
# Unit tests (no API keys needed)
uv run pytest packages/tarash-gateway/tests/unit/

# End-to-end tests (requires API keys)
uv run pytest packages/tarash-gateway/tests/e2e/ --e2e
```

### Adding a Workspace Package

Create a new package under `packages/`. Each package must have its own `pyproject.toml`.

To depend on another workspace package:

```toml
dependencies = ["tarash-gateway"]

[tool.uv.sources]
tarash-gateway = { workspace = true }
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
