# ElevenLabs

ElevenLabs provides high-quality text-to-speech (TTS) and speech-to-speech (STS) audio generation. This provider uses the `elevenlabs` Python SDK.

!!! info "Audio only"
    ElevenLabs supports TTS and STS audio generation. Use this provider with `AudioGenerationConfig` and `generate_tts()` / `generate_sts()`.

## Quick Example — Text-to-Speech (TTS)

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    AudioOutputFormat,
    TTSRequest,
)

config = AudioGenerationConfig(
    provider="elevenlabs",
    model="eleven_multilingual_v2",
    api_key="YOUR_ELEVENLABS_KEY",
)

request = TTSRequest(
    text="Hello, welcome to Tarash Gateway!",
    voice_id="21m00Tcm4TlvDq8ikWAM",  # "Rachel" voice
    output_format=AudioOutputFormat(format="mp3", sample_rate=44100, bitrate=128),
)

response = generate_tts(config, request)
print(response.request_id)    # Unique request ID
print(response.content_type)  # e.g. "audio/mpeg"
# response.audio contains base64-encoded audio bytes
```

## Quick Example — Speech-to-Speech (STS)

```python
from tarash.tarash_gateway import generate_sts
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    AudioOutputFormat,
    STSRequest,
)

config = AudioGenerationConfig(
    provider="elevenlabs",
    model="eleven_multilingual_v2",
    api_key="YOUR_ELEVENLABS_KEY",
)

request = STSRequest(
    audio="https://example.com/input-speech.mp3",  # URL, bytes, or base64
    voice_id="21m00Tcm4TlvDq8ikWAM",
    output_format=AudioOutputFormat(format="mp3", sample_rate=44100, bitrate=128),
)

response = generate_sts(config, request)
print(response.request_id)
# response.audio contains base64-encoded audio bytes
```

Async variants are also available:

```python
from tarash.tarash_gateway import generate_tts_async, generate_sts_async

response = await generate_tts_async(config, tts_request)
response = await generate_sts_async(config, sts_request)
```

---

## TTS Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `text` | Yes | Yes | The text to convert to speech |
| `voice_id` | Yes | Yes | ElevenLabs voice identifier |
| `output_format` | — | Yes | `AudioOutputFormat` with `format`, `sample_rate`, `bitrate` |
| `voice_settings` | — | Yes | Dict with `stability`, `similarity_boost`, etc. |
| `language_code` | — | Yes | Language hint (e.g. `"en"`, `"es"`, `"fr"`) |
| `extra_params` | — | Yes | Any additional ElevenLabs API parameters |

## STS Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `audio` | Yes | Yes | Input audio — URL, raw bytes, or base64 string |
| `voice_id` | Yes | Yes | ElevenLabs voice identifier |
| `output_format` | — | Yes | `AudioOutputFormat` with `format`, `sample_rate`, `bitrate` |
| `voice_settings` | — | Yes | Dict with `stability`, `similarity_boost`, etc. (JSON-encoded for STS multipart form) |
| `extra_params` | — | Yes | Any additional ElevenLabs API parameters |

### Output Format

The `output_format` is converted to an ElevenLabs format string by joining the parts with underscores:

```python
AudioOutputFormat(format="mp3", sample_rate=44100, bitrate=128)  # → "mp3_44100_128"
AudioOutputFormat(format="pcm", sample_rate=16000)               # → "pcm_16000"
AudioOutputFormat(format="mp3")                                  # → "mp3"
```

### Voice Settings

Pass provider-specific voice tuning via `voice_settings`:

```python
request = TTSRequest(
    text="Hello!",
    voice_id="21m00Tcm4TlvDq8ikWAM",
    voice_settings={
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
    },
)
```

---

## Supported Models

ElevenLabs model IDs are not hardcoded — any valid ElevenLabs model ID can be passed in the config. Common models include:

| Model ID | Quality | Latency | Notes |
|---|---|---|---|
| `eleven_multilingual_v2` | Highest | Standard | 29 languages, best quality |
| `eleven_flash_v2_5` | Good | Low | Optimized for low latency |
| `eleven_turbo_v2_5` | High | Medium | Balance of speed and quality |

```python
config = AudioGenerationConfig(
    provider="elevenlabs",
    model="eleven_flash_v2_5",  # Any valid ElevenLabs model ID
    api_key="YOUR_ELEVENLABS_KEY",
)
```

---

## Provider-Specific Notes

**SDK dependency:** This provider requires the `elevenlabs` package. Install with:

```bash
pip install tarash-gateway[elevenlabs]
```

**Authentication:** The `api_key` must always be passed explicitly in the `AudioGenerationConfig`. There is no automatic environment variable fallback.

**No progress callbacks:** ElevenLabs streams audio chunks directly with no status or polling events. Progress callbacks (`on_progress`) are accepted for interface compatibility but are not invoked.

**STS audio input:** The `audio` field in `STSRequest` accepts multiple formats:

- A URL string (e.g. `"https://example.com/audio.mp3"`) — downloaded automatically
- Raw bytes
- A base64-encoded string
- A dict with a `"content"` key containing raw bytes

**Response format:** Audio is returned as a base64-encoded string in `response.audio`. The `content_type` field indicates the MIME type (e.g. `"audio/mpeg"` for MP3).

**Timeout configuration:** Use `AudioGenerationConfig.timeout` to control the maximum wait time in seconds (default: 240).
