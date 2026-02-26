# Tarash Gateway

**Unified Python SDK for AI video and image generation.**

One interface. Multiple providers. Production-ready.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is Tarash Gateway?

Tarash Gateway is a unified Python SDK that abstracts AI video and image generation across multiple providers — similar to what LiteLLM does for text generation.

Each AI video provider (OpenAI Sora, Runway, Fal.ai, etc.) has different APIs, parameter names, and response formats. Supporting multiple providers means writing and maintaining separate integration code for each. Tarash Gateway provides a single, consistent API that works across all providers with automatic parameter mapping, unified error handling, and production-ready features like fallback chains and progress tracking.

---

## Features

- **Multi-provider support** — OpenAI Sora, Runway, Fal.ai, Replicate, Google Veo3, Azure OpenAI
- **Unified interface** — one API with automatic parameter mapping across providers
- **Async/sync support** — full async/await or blocking calls
- **Progress tracking** — real-time callbacks for long-running generation tasks
- **Type safety** — Pydantic models with IDE autocomplete and validation
- **Mock generation** — test without API calls using configurable mock responses
- **Structured logging** — automatic credential redaction and request tracing
- **Advanced capabilities** — image-to-video, video-to-video, extend, remix, and more
- **Provider fallback** — automatic failover to backup providers on errors
- **Rich exceptions** — detailed error context with provider, model, and request info

---

## Supported Providers

| Provider | Models | Text-to-Video | Image-to-Video | Video-to-Video | Extend/Remix |
|----------|--------|:---:|:---:|:---:|:---:|
| **OpenAI** | Sora-2 Turbo | ✓ | ✓ | ✓ | ✓ |
| **Runway** | Gen-3 Alpha | ✓ | ✓ | — | — |
| **Fal.ai** | Veo3, Minimax, Kling, Sora-2 | ✓ | ✓ | ✓ | ✓ |
| **Google** | Veo3 (Vertex AI) | ✓ | ✓ | — | — |
| **Replicate** | Minimax, Luma, Kling, Veo3 | ✓ | ✓ | ✓ | — |
| **Azure OpenAI** | Sora-2 | ✓ | ✓ | ✓ | ✓ |

---

## Installation

```bash
pip install tarash-gateway

# Provider-specific extras
pip install tarash-gateway[openai]
pip install tarash-gateway[fal]
pip install tarash-gateway[all]
```

**Requirements:** Python 3.12+

---

## Quick Start

```python
from tarash_gateway import VideoGenerationConfig, VideoGenerationRequest, generate_video

config = VideoGenerationConfig(
    model="openai/sora-2-turbo",
    provider="openai",
    api_key="your-api-key"
)

request = VideoGenerationRequest(
    prompt="Sunset over mountains, cinematic",
    aspect_ratio="16:9",
    duration_seconds=5
)

response = generate_video(config, request)
print(f"Video: {response.video}")
```

Switching providers requires only changing `model` and `provider` in the config — the request and the rest of your code stay the same.

See [CLAUDE.md](CLAUDE.md) for detailed usage guides covering async generation, progress tracking, image-to-video, provider fallback, and mock testing.

---

## Architecture

```
tarash_gateway/
├── api.py                  # Public API
├── models.py               # Request/response models
├── exceptions.py           # Exception hierarchy
├── orchestrator.py         # Fallback and retry orchestration
├── mock.py                 # Mock provider for testing
├── logging.py              # Structured logging
└── providers/              # Provider implementations + field mapper registry
```

**Key design principles:**
- All providers implement the same `ProviderHandler` protocol
- Declarative parameter translation via a field mapper registry
- Immutable, thread-safe configuration with frozen Pydantic models
- Every operation available in both sync and async modes

---

## Development

**Requirements:** Python 3.12+, [`uv`](https://github.com/astral-sh/uv)

```bash
uv sync
uv run pytest tests/unit/      # No API keys needed
uv run pytest tests/e2e/ --e2e # Requires API keys
```

See [CLAUDE.md](CLAUDE.md) for coding standards, provider implementation guides, and contribution guidelines.

---

## License

MIT — see [LICENSE](LICENSE) for details.
