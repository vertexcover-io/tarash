# Sarvam AI

Sarvam AI provides high-quality text-to-speech (TTS) for **Indian languages**, supporting 11 language codes across major regional languages. Tarash uses the `sarvamai` Python SDK.

!!! info "TTS Only"
    Sarvam AI supports **text-to-speech only**. Speech-to-speech (STS) is not available through this provider.

---

## Quick Example

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, AudioOutputFormat, TTSRequest

config = AudioGenerationConfig(
    provider="sarvam",
    model="bulbul:v3",
    api_key="YOUR_SARVAM_KEY",
)

request = TTSRequest(
    text="नमस्ते, आज का मौसम बहुत अच्छा है।",
    language_code="hi-IN",
    output_format=AudioOutputFormat(format="wav", sample_rate=24000),
)

response = generate_tts(config, request)
print(f"Request ID: {response.request_id}")
print(f"Content type: {response.content_type}")
```

---

## Parameters

| Parameter | TTSRequest Field | Required | Notes |
|---|---|:---:|---|
| Text | `text` | ✅ | The text to synthesize into speech |
| Language | `language_code` | ✅ | **Must be provided.** Sarvam requires a target language code (e.g. `hi-IN`). Omitting this raises a `ValidationError`. |
| Voice | `voice_id` | — | Maps to Sarvam's `speaker` parameter |
| Output format | `output_format.format` | — | Audio codec (e.g. `wav`, `mp3`) |
| Sample rate | `output_format.sample_rate` | — | Defaults to `24000` Hz if not specified |

### Extra Parameters

Provider-specific options can be passed through `extra_params` and `voice_settings`:

```python
request = TTSRequest(
    text="नमस्ते दुनिया",
    language_code="hi-IN",
    voice_settings={"pace": 1.2},
    extra_params={"enable_preprocessing": True},
)
```

---

## Supported Languages

| Language | Code |
|---|---|
| Hindi | `hi-IN` |
| English (India) | `en-IN` |
| Tamil | `ta-IN` |
| Telugu | `te-IN` |
| Kannada | `kn-IN` |
| Malayalam | `ml-IN` |
| Marathi | `mr-IN` |
| Bengali | `bn-IN` |
| Gujarati | `gu-IN` |
| Odia | `od-IN` |
| Punjabi | `pa-IN` |

---

## Supported Models

| Model ID | Notes |
|---|---|
| `bulbul:v3` | Latest version, recommended |
| `bulbul:v2` | Previous version |

---

## Installation

Install tarash-gateway with the Sarvam extra:

```bash
pip install tarash-gateway[sarvam]
```

This installs the `sarvamai` SDK (`>=0.1.25`).

---

## Provider-Specific Notes

**`language_code` is mandatory:** Unlike most other TTS providers, Sarvam requires `language_code` on every request. If omitted, a `ValidationError` is raised before the API call is made. Use the BCP-47 style codes listed above (e.g. `hi-IN`, `en-IN`).

**Authentication:** The `api_key` is passed to the Sarvam SDK as `api_subscription_key`. You must always provide `api_key` explicitly in `AudioGenerationConfig`.

**Response format:** Sarvam returns audio as a base64-encoded string. The `TTSResponse.audio` field contains this base64 data, and `content_type` is set based on the requested output format.

**No streaming:** Sarvam returns the complete audio in a single response. The `on_progress` callback is accepted but unused.
