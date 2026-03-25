# Installation

## Requirements

- Python 3.12+

---

## Install

Install with all provider dependencies:

```bash
pip install tarash-gateway[all]
```

Or install a single provider:

```bash
pip install tarash-gateway[fal]
pip install tarash-gateway[openai]
pip install tarash-gateway[runway]
```

---

## Available extras

| Extra | Providers included |
|---|---|
| `fal` | Fal.ai |
| `openai` | OpenAI, Azure OpenAI |
| `runway` | Runway |
| `veo3` | Google Veo (Gemini API) |
| `google` | Google Vertex AI |
| `replicate` | Replicate |
| `xai` | xAI (Grok) |
| `elevenlabs` | ElevenLabs TTS |
| `cartesia` | Cartesia TTS |
| `sarvam` | Sarvam AI TTS |
| `hume` | Hume AI TTS |
| `all` | All of the above |

> **Note:** Stability AI uses `httpx`, which is a core dependency of tarash-gateway. No extra pip install is needed — just `pip install tarash-gateway`.

---

## Verify installation

```python
import tarash.tarash_gateway
print("tarash-gateway installed successfully")
```

---

## Next steps

- [Quickstart](quickstart.md) — generate your first video in 10 lines
- [Authentication](authentication.md) — set up your API keys
