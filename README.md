<h1 align="center">Tarash</h1>

<p align="center">
  <strong>AI media toolkit</strong><br>
  Generate video, image, and audio with a unified interface — plus tools to process and enhance them
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://tarash.vertexcover.io"><b>Documentation</b></a> &nbsp;·&nbsp;
  <a href="https://tarash.vertexcover.io/getting-started/quickstart/">Quick Start</a> &nbsp;·&nbsp;
  <a href="https://tarash.vertexcover.io/providers/">Providers</a> &nbsp;·&nbsp;
  <a href="https://blog.vertexcover.io/unified-ai-video-gateway">Blog</a>
</p>

---

## What is Tarash?

Every AI media provider ships its own SDK with its own parameter names, response shapes, and error formats. Tarash is a Python toolkit that unifies all of this — generation across providers, plus processing tools to enhance the output.

**Tarash Gateway** is the first package: a unified SDK for video, image, and audio generation across 10+ providers and 100+ models. Write one integration, swap providers by changing config.

---

## Quick Example

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

Switch to any other provider — same request, same response:

```python
config = VideoGenerationConfig(
    model="gen4_turbo", provider="runway", api_key="YOUR_RUNWAY_KEY",
)
response = generate_video(config, request)
```

> See [tarash-gateway](packages/tarash-gateway/) for image, audio, async, and fallback examples.

---

## Packages

| Package | Description | |
|---------|-------------|-|
| **[tarash-gateway](packages/tarash-gateway/)** | Unified SDK for AI video, image, and audio generation | Stable |
| *tarash-tools* | Media processing utilities (silence removal, scene detection, and more) | Coming soon |

---

## Supported Providers

| Provider | Video | Image | Audio |
|----------|:-----:|:-----:|:-----:|
| **Fal.ai** | ✓ | ✓ | ✓ |
| **OpenAI** | ✓ | ✓ | — |
| **Google** | ✓ | ✓ | — |
| **Runway** | ✓ | — | — |
| **Replicate** | ✓ | — | — |
| **Luma** | ✓ | ✓ | — |
| **XAI** | ✓ | — | — |
| **Stability AI** | — | ✓ | — |
| **ElevenLabs** | — | — | ✓ |
| **Cartesia** | — | — | ✓ |
| **Sarvam** | — | — | ✓ |
| **Hume** | — | — | ✓ |

> **Full model list at [tarash.vertexcover.io/providers](https://tarash.vertexcover.io/providers/)**

---

## Highlights

- **One interface for video, image, and audio** — stop rewriting integrations for every provider
- **Swap providers by changing config** — your request code, response handling, and error logic stay identical
- **Automatic fallback chains** — if a provider goes down, the next one picks up seamlessly
- **Sync and async** — every function has both `generate_*` and `generate_*_async` variants
- **Production-ready** — type-safe Pydantic v2 models, structured logging, and rich error context

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

> **Requires Python 3.12+** — see the [installation guide](https://tarash.vertexcover.io/getting-started/installation/) for details.

---

## Contributing

Tarash is open source and contributions are welcome.

- **Questions or ideas?** Open a [Discussion](https://github.com/vertexcover-io/tarash/discussions)
- **Found a bug?** File an [Issue](https://github.com/vertexcover-io/tarash/issues)
- **Want to add a provider?** See the [custom providers guide](https://tarash.vertexcover.io/guides/custom-providers/)

<details>
<summary><strong>Development setup</strong></summary>

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

</details>

---

## License

MIT — see [LICENSE](LICENSE) for details.
