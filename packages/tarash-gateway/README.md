<h1 align="center">Tarash Gateway</h1>

<p align="center">
  <strong>Unified Python SDK for AI video, image, and audio generation</strong><br>
  One interface. Multiple providers. Production-ready.
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://tarash.vertexcover.io"><b>Documentation</b></a> &nbsp;·&nbsp;
  <a href="https://tarash.vertexcover.io/getting-started/quickstart/">Quick Start</a> &nbsp;·&nbsp;
  <a href="https://tarash.vertexcover.io/providers/">Providers</a>
</p>

---

## What is Tarash Gateway?

Tarash Gateway abstracts AI media generation across multiple providers behind a single, consistent API. Write one integration and swap providers by changing config — your request code, response handling, and error logic stay identical.

> **Full documentation at [tarash.vertexcover.io](https://tarash.vertexcover.io)**

---

## Installation

```bash
pip install tarash-gateway[fal]
```

Install only the provider extras you need:

```bash
pip install tarash-gateway[openai]       # OpenAI / Azure
pip install tarash-gateway[runway]       # Runway
pip install tarash-gateway[elevenlabs]   # ElevenLabs TTS
pip install tarash-gateway[fal,runway]   # Multiple providers
pip install tarash-gateway[all]          # Everything
```

**Requires Python 3.12+** — see the [installation guide](https://tarash.vertexcover.io/getting-started/installation/) for details.

---

## Quick Start

### Video Generation

```python
from tarash_gateway import VideoGenerationConfig, VideoGenerationRequest, generate_video

config = VideoGenerationConfig(
    model="fal-ai/veo3.1/fast",
    provider="fal",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    prompt="Sunset over mountains, cinematic",
    aspect_ratio="16:9",
    duration_seconds=5,
)

response = generate_video(config, request)
print(response.video)  # URL to generated video
```

Switch providers — your request code stays the same:

```python
config = VideoGenerationConfig(
    model="gen4_turbo", provider="runway", api_key="YOUR_RUNWAY_KEY",
)
response = generate_video(config, request)  # Same request, same response shape
```

<details>
<summary><strong>Image Generation</strong></summary>

```python
from tarash_gateway import ImageGenerationConfig, ImageGenerationRequest, generate_image

config = ImageGenerationConfig(
    model="fal-ai/flux-2/pro",
    provider="fal",
    api_key="YOUR_FAL_KEY",
)

request = ImageGenerationRequest(
    prompt="A futuristic cityscape at dawn, digital art",
    aspect_ratio="16:9",
)

response = generate_image(config, request)
print(response.image)
```

</details>

<details>
<summary><strong>Audio — TTS</strong></summary>

```python
from tarash_gateway import AudioGenerationConfig, TTSRequest, generate_tts

config = AudioGenerationConfig(
    model="eleven_multilingual_v2",
    provider="elevenlabs",
    api_key="YOUR_ELEVENLABS_KEY",
)

request = TTSRequest(
    text="Hello from Tarash!",
    voice_id="your-voice-id",
    output_format="mp3_44100_128",
)

response = generate_tts(config, request)
# response.audio — base64-encoded audio
# response.content_type — e.g. "audio/mpeg"
```

</details>

<details>
<summary><strong>Async & Provider Fallback</strong></summary>

Every function has an async variant:

```python
from tarash_gateway import generate_video_async

response = await generate_video_async(config, request)
```

Set up automatic fallback across providers:

```python
config = VideoGenerationConfig(
    model="fal-ai/veo3.1/fast",
    provider="fal",
    api_key="YOUR_FAL_KEY",
    fallback_configs=[
        VideoGenerationConfig(
            model="gen4_turbo",
            provider="runway",
            api_key="YOUR_RUNWAY_KEY",
        ),
    ],
)

# Falls back to Runway automatically if Fal fails
response = generate_video(config, request)
```

</details>

> See the [guides](https://tarash.vertexcover.io/guides/video-generation/) for more examples.

---

## Providers

| Provider | Video | Image | Audio |
|----------|:-----:|:-----:|:-----:|
| **[Fal.ai](https://tarash.vertexcover.io/providers/fal/)** | ✓ | ✓ | ✓ |
| **[OpenAI](https://tarash.vertexcover.io/providers/openai/)** | ✓ | ✓ | — |
| **[Google](https://tarash.vertexcover.io/providers/google/)** | ✓ | ✓ | — |
| **[Runway](https://tarash.vertexcover.io/providers/runway/)** | ✓ | — | — |
| **[Replicate](https://tarash.vertexcover.io/providers/replicate/)** | ✓ | — | — |
| **[XAI](https://tarash.vertexcover.io/providers/)** | ✓ | — | — |
| **[Stability AI](https://tarash.vertexcover.io/providers/stability/)** | — | ✓ | — |
| **[ElevenLabs](https://tarash.vertexcover.io/providers/)** | — | — | ✓ |
| **[Cartesia](https://tarash.vertexcover.io/providers/)** | — | — | ✓ |
| **[Sarvam](https://tarash.vertexcover.io/providers/)** | — | — | ✓ |
| **[Hume](https://tarash.vertexcover.io/providers/)** | — | — | ✓ |

> **Full model list and provider details at [tarash.vertexcover.io/providers](https://tarash.vertexcover.io/providers/)**

---

## Features

| Feature | Description |
|---------|-------------|
| **Sync + Async** | Every function has `generate_*` and `generate_*_async` variants |
| **Provider Fallback** | Automatic failover across a prioritized provider chain |
| **Progress Callbacks** | Real-time updates during long-running generation |
| **Type-Safe** | Pydantic v2 models with full IDE autocomplete |
| **Rich Errors** | Every exception includes `provider`, `model`, `request_id`, and raw response |
| **Mock Provider** | Deterministic fake responses for testing without API calls |
| **Structured Logging** | Automatic credential redaction and request tracing |

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

**Requirements:** Python 3.12+, [`uv`](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/vertexcover-io/tarash.git
cd tarash
uv sync
```

```bash
# Unit tests (no API keys needed)
uv run pytest packages/tarash-gateway/tests/unit/

# End-to-end tests (requires API keys)
uv run pytest packages/tarash-gateway/tests/e2e/ --e2e
```

---

## License

MIT — see [LICENSE](../../LICENSE) for details.
