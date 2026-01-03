<div align="center">

# 🎬 Tarash Gateway

### **Unified Python SDK for AI Video Generation**

*One interface. Multiple providers. Production-ready.*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-332%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-90%2B%25-brightgreen.svg)](htmlcov/)

[**Quick Start**](#quick-start) • [**Examples**](#quick-start) • [**Providers**](#supported-providers) • [**Documentation**](#documentation)

</div>

---

## What is Tarash Gateway?

Tarash Gateway is a unified Python SDK that abstracts AI video generation across multiple providers—similar to what LiteLLM does for text generation.

**The Problem**: Each AI video provider (OpenAI Sora, Runway, Fal.ai, etc.) has different APIs, parameter names, and response formats. Switching providers or multi-provider support requires rewriting integration code.

**The Solution**: Tarash Gateway provides a single, consistent API that works across all providers with intelligent parameter mapping, unified error handling, and production-ready features.

```python
from tarash_gateway.video import VideoGenerationConfig, VideoGenerationRequest, generate_video

# Configure provider
config = VideoGenerationConfig(
    model="openai/sora-2-turbo",
    provider="openai",
    api_key="your-api-key"
)

# Create request
request = VideoGenerationRequest(
    prompt="A cat playing piano in a jazz club",
    aspect_ratio="16:9",
    duration_seconds=5
)

# Generate video - works with any provider, just change model/provider in config
response = generate_video(config, request)

print(f"Video URL: {response.video}")
```

---

## ✨ Features

- 🔌 **Multi-Provider Support** - OpenAI Sora, Runway, Fal.ai, Replicate, Google Veo3, Azure OpenAI
- 🎯 **Unified Interface** - One API, automatic parameter mapping across providers
- ⚡ **Async/Sync Support** - Full async/await with `generate_video_async()` or blocking calls
- 📊 **Progress Tracking** - Real-time callbacks for long-running generation tasks
- 🛡️ **Type Safety** - Pydantic models with IDE autocomplete and validation
- 🧪 **Mock Generation** - Test without API calls using configurable mock responses
- 📝 **Structured Logging** - Automatic credential redaction and request tracing
- 🎨 **Advanced Features** - Image-to-video, video-to-video, extend, remix, and more
- 🔄 **Provider Fallback** - Automatic failover to backup providers on errors
- 🎭 **Rich Exceptions** - Detailed error context with provider/model/request info

---

## 🌐 Supported Providers

| Provider | Models | Text-to-Video | Image-to-Video | Video-to-Video | Extend/Remix |
|----------|--------|---------------|----------------|----------------|--------------|
| **OpenAI** | Sora-2 Turbo | ✅ | ✅ | ✅ | ✅ |
| **Runway** | Gen-3 Alpha | ✅ | ✅ | ❌ | ❌ |
| **Fal.ai** | Veo3, Minimax, Kling, Sora-2 | ✅ | ✅ | ✅ | ✅ |
| **Google** | Veo3 (Vertex AI) | ✅ | ✅ | ❌ | ❌ |
| **Replicate** | Minimax, Luma, Kling, Veo3 | ✅ | ✅ | ✅ | ❌ |
| **Azure OpenAI** | Sora-2 | ✅ | ✅ | ✅ | ✅ |

*See [CLAUDE.md](CLAUDE.md) for complete model list and capabilities.*

---

## 🚀 Installation

**Base installation** (includes core functionality):
```bash
pip install tarash-gateway
```

**Provider-specific extras** (install only what you need):
```bash
# For OpenAI Sora
pip install tarash-gateway[openai]

# For Runway
pip install tarash-gateway[runway]

# For Fal.ai
pip install tarash-gateway[fal]

# For all providers
pip install tarash-gateway[all]
```

**Requirements**: Python 3.12+

---

## ⚡ Quick Start

### Basic Usage

```python
from tarash_gateway.video import VideoGenerationConfig, VideoGenerationRequest, generate_video

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

### Async Usage

```python
import asyncio
from tarash_gateway.video import VideoGenerationConfig, VideoGenerationRequest, generate_video_async

async def main():
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

    response = await generate_video_async(config, request)
    print(f"Video: {response.video}")

asyncio.run(main())
```

### Progress Tracking

```python
from tarash_gateway.video import VideoGenerationConfig, VideoGenerationRequest, generate_video

def on_progress(update):
    print(f"{update.status}: {update.progress_percent}% - {update.message}")

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

response = generate_video(config, request, on_progress=on_progress)
```

### Image-to-Video

```python
from tarash_gateway.video import VideoGenerationConfig, VideoGenerationRequest, generate_video

config = VideoGenerationConfig(
    model="openai/sora-2-turbo",
    provider="openai",
    api_key="your-api-key"
)

request = VideoGenerationRequest(
    prompt="Camera zooms in dramatically",
    image_list=[{
        "image": "https://example.com/image.jpg",
        "type": "first_frame"
    }],
    duration_seconds=5
)

response = generate_video(config, request)
```

### Mock Generation for Testing

```python
from tarash_gateway.video import VideoGenerationConfig, generate_video
from tarash_gateway.video.mock import MockConfig

# Enable mock mode in config
config = VideoGenerationConfig(
    model="mock-model",
    provider="mock",
    api_key="fake-key",
    mock=MockConfig(enabled=True)
)

response = generate_video(
    config=config,
    request=VideoGenerationRequest(
        prompt="Test video",
        aspect_ratio="16:9",
        duration_seconds=4
    )
)

assert response.is_mock == True
assert response.status == "completed"
assert response.video is not None
```

### Provider Fallback

```python
from tarash_gateway.video import VideoGenerationConfig, generate_video

# Primary config with fallback chain
config = VideoGenerationConfig(
    model="fal/veo-3",
    provider="fal",
    api_key="fal_key",
    fallback_configs=[
        VideoGenerationConfig(
            model="openai/sora-2-turbo",
            provider="openai",
            api_key="openai_key"
        ),
        VideoGenerationConfig(
            model="runway/gen-3-alpha",
            provider="runway",
            api_key="runway_key"
        )
    ]
)

# Automatically tries fallbacks if primary fails
response = generate_video(
    prompt="Sunset over mountains",
    config=config,
    aspect_ratio="16:9"
)

# Check execution metadata
print(f"Attempts: {response.execution_metadata.total_attempts}")
print(f"Fallback triggered: {response.execution_metadata.fallback_triggered}")
```

---

## 🎨 Advanced Usage

- **Provider-Specific Parameters** - Access provider-native features through `extra_params`
- **Video-to-Video Workflows** - Extend, remix, and transform existing videos
- **Custom Field Mapping** - Add support for new models via field mapper registry
- **Mock Generation for Testing** - Configure deterministic test responses with weighted responses
- **Execution Metadata** - Track attempts, fallback chains, and timing for debugging
- **Logging & Debugging** - Structured logs with automatic credential redaction

*See [CLAUDE.md](CLAUDE.md) for detailed guides.*

---

## 🤔 Why Tarash Gateway?

| **Without Tarash Gateway** | **With Tarash Gateway** |
|----------------------------|-------------------------|
| Different SDK per provider | Single unified interface |
| Manual parameter mapping | Automatic field translation |
| Provider-specific error handling | Unified exception hierarchy |
| Custom retry logic per provider | Built-in fallback chains |
| Multiple authentication patterns | Consistent config system |
| Different async implementations | Unified async/sync support |
| Manual progress tracking | Built-in callbacks |
| Test against live APIs only | Mock generation included |
| No execution visibility | Rich metadata with attempt tracking |

---

## 🏗️ Architecture

Tarash Gateway is designed around **provider abstraction** with intelligent parameter mapping:

```
tarash_gateway/
├── video/
│   ├── api.py              # Public API (generate_video, generate_video_async)
│   ├── models.py           # Request/Response models
│   ├── exceptions.py       # Exception hierarchy
│   ├── registry.py         # Provider handler registry
│   ├── mock.py             # Mock provider for testing
│   └── providers/
│       ├── fal.py          # Fal.ai provider implementation
│       ├── openai.py       # OpenAI Sora provider
│       ├── veo3.py         # Google Veo3 provider
│       ├── replicate.py    # Replicate provider
│       ├── runway.py       # Runway provider
│       └── field_mappers.py # Parameter translation framework
└── logging.py              # Structured logging with credential redaction
```

**Key Design Principles**:
- **Provider Abstraction**: All providers implement `ProviderHandler` protocol
- **Field Mapper Registry**: Declarative parameter translation with model variant support
- **Immutable Configuration**: Thread-safe frozen Pydantic models
- **Progressive Enhancement**: Use SDK types when available, graceful fallback

---

## 🛠️ Development

**Requirements**: Python 3.12+, [`uv`](https://github.com/astral-sh/uv) package manager

```bash
# Clone repository
git clone https://github.com/yourusername/tarash.git
cd tarash/packages/tarash-gateway

# Install dependencies (from package root)
uv sync

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=tarash_gateway --cov-report=html

# Run only unit tests (fast, no API keys needed)
uv run pytest tests/unit/

# Run E2E tests (requires API keys)
uv run pytest tests/e2e/
```

**Project Structure**:
- **Unit Tests**: Mock-based tests in `tests/unit/`
- **E2E Tests**: Real API tests in `tests/e2e/` (requires API keys)
- **Guidelines**: See [CLAUDE.md](CLAUDE.md) for coding standards

---

## 🧪 Testing

Tarash Gateway includes **332 tests** covering unit and end-to-end scenarios.

**Run all tests**:
```bash
uv run pytest
```

**Mock generation for testing**:
```python
from tarash_gateway.video import VideoGenerationConfig, VideoGenerationRequest, generate_video
from tarash_gateway.video.mock import MockConfig

config = VideoGenerationConfig(
    model="mock-model",
    provider="mock",
    api_key="fake-key",
    mock=MockConfig(enabled=True)
)

response = generate_video(
    config=config,
    request=VideoGenerationRequest(
        prompt="Test video",
        aspect_ratio="16:9",
        duration_seconds=4
    )
)

assert response.is_mock == True
assert response.status == "completed"
```

**Test Coverage**: 90%+ across core modules

---

## 🤝 Contributing

We welcome contributions! See [CLAUDE.md](CLAUDE.md) for guidelines.

**Quick checklist**:
- ✅ Use `uv run` for all Python commands
- ✅ Add tests for new features
- ✅ Follow type hints (prefer specific types over `Any`)
- ✅ Use function-based tests (not class-based)
- ✅ Commit messages with GitHub emoji (`:sparkles:`, `:bug:`, etc.)

**Adding a new provider?** See provider implementations in `src/tarash/tarash_gateway/video/providers/` for examples.

---

## 🗺️ Roadmap

- [x] **Provider fallback** - Automatic failover to backup providers ✅
- [ ] **Additional video providers** - Stability AI, Pika Labs, Genmo
- [ ] **Audio generation** - Music/sound/speech generation providers
- [ ] **Image generation** - DALL-E, Midjourney, Stable Diffusion support
- [ ] **Streaming responses** - Real-time frame/progress streaming
- [ ] **Cost tracking** - Built-in usage/cost monitoring per provider
- [ ] **Smart routing** - Automatic provider selection based on cost/speed
- [ ] **Batch operations** - Process multiple videos efficiently

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with:
- [Pydantic](https://pydantic.dev) - Data validation
- [httpx](https://www.python-httpx.org/) - Async HTTP
- [pytest](https://pytest.org) - Testing framework

Inspired by [LiteLLM](https://github.com/BerriAI/litellm) for text generation.
