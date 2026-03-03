# Qwen 3 TTS (via Fal.ai)

Qwen 3 text-to-speech model hosted on Fal.ai, with voice customization and fine-grained sampling control.

## Quick Example

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, TTSRequest

config = AudioGenerationConfig(
    provider="fal",
    model="fal-ai/qwen-3-tts",
    api_key="YOUR_FAL_KEY",
)

request = TTSRequest(
    text="Hello! This is a test of the Qwen 3 TTS model.",
    voice_id="Chelsie",
    language_code="en",
)

response = generate_tts(config, request)
print(f"Audio size: {len(response.audio)} chars (base64)")
```

---

## Parameters

| Parameter | TTSRequest field | Required | Notes |
|---|---|:---:|---|
| Text | `text` | ✅ | Text to synthesize |
| Voice | `voice_id` | — | Voice selection |
| Style prompt | `prompt` | — | Voice style guidance |
| Language | `language_code` | — | Language / locale |

### Sampling Parameters

Pass via `extra_params` for fine-grained control over generation:

| Parameter | Type | Notes |
|---|---|---|
| `temperature` | `float` | Sampling temperature |
| `top_k` | `int` | Top-k sampling filter |
| `top_p` | `float` | Nucleus (top-p) sampling |
| `repetition_penalty` | `float` | Penalty for repeated tokens |
| `max_new_tokens` | `int` | Maximum token count |

### Sub-Talker Parameters

Control the sub-talker (additional speaker modeling) via `extra_params`:

| Parameter | Type | Notes |
|---|---|---|
| `subtalker_dosample` | `bool` | Whether to use sampling |
| `subtalker_top_k` | `int` | Sub-talker top-k |
| `subtalker_top_p` | `float` | Sub-talker nucleus sampling |
| `subtalker_temperature` | `float` | Sub-talker temperature |

```python
request = TTSRequest(
    text="Hello world",
    voice_id="Chelsie",
    extra_params={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 2048,
    },
)
```
