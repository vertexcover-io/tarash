# Cartesia

Cartesia provides high-quality text-to-speech (TTS) and speech-to-speech (STS / Voice Changer) audio generation. Tarash uses the official `cartesia` Python SDK.

---

## Installation

```bash
pip install tarash-gateway[cartesia]
```

---

## Text-to-Speech (TTS)

### Quick Example

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, AudioOutputFormat, TTSRequest

config = AudioGenerationConfig(
    provider="cartesia",
    model="sonic-3",
    api_key="YOUR_CARTESIA_KEY",
)

request = TTSRequest(
    text="Hello, welcome to Cartesia text-to-speech!",
    voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
    output_format=AudioOutputFormat(format="mp3", sample_rate=44100, bitrate=128),
    language_code="en",
)

response = generate_tts(config, request)
print(f"Audio size: {len(response.audio)} bytes (base64)")
print(f"Content type: {response.content_type}")
```

### Async Example

```python
from tarash.tarash_gateway import generate_tts_async

response = await generate_tts_async(config, request)
```

### Parameters

| Parameter | TTSRequest field | Required | Notes |
|---|---|:---:|---|
| Text | `text` | Yes | The text to convert to speech |
| Voice | `voice_id` | Yes | Cartesia voice UUID. Converted to `{"mode": "id", "id": voice_id}` internally |
| Output format | `output_format` | -- | `AudioOutputFormat(format, sample_rate, bitrate)`. Defaults to mp3 / 44100 Hz / 128 kbps |
| Language | `language_code` | -- | Language hint passed as `language` to the API |

---

## Speech-to-Speech (Voice Changer)

### Quick Example

```python
from tarash.tarash_gateway import generate_sts
from tarash.tarash_gateway.models import AudioGenerationConfig, AudioOutputFormat, STSRequest

config = AudioGenerationConfig(
    provider="cartesia",
    model="sonic-3",
    api_key="YOUR_CARTESIA_KEY",
)

request = STSRequest(
    audio="https://example.com/input-speech.wav",
    voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
    output_format=AudioOutputFormat(format="wav", sample_rate=44100),
)

response = generate_sts(config, request)
print(f"Audio size: {len(response.audio)} bytes (base64)")
```

### Async Example

```python
from tarash.tarash_gateway import generate_sts_async

response = await generate_sts_async(config, request)
```

### Audio Input

The `audio` field accepts multiple formats:

- **URL** -- `"https://example.com/audio.wav"`
- **Base64 string** -- raw base64-encoded audio bytes
- **Raw bytes** -- `bytes` object
- **MediaContent dict** -- `{"content": b"..."}` with raw bytes

### Parameters

| Parameter | STSRequest field | Required | Notes |
|---|---|:---:|---|
| Audio input | `audio` | Yes | Source audio clip (URL, base64, bytes, or dict) |
| Voice | `voice_id` | Yes | Target voice UUID for the conversion |
| Output format | `output_format` | -- | `AudioOutputFormat(format, sample_rate, bitrate)`. Defaults to mp3 / 44100 Hz / 128 kbps |

STS uses the Cartesia Voice Changer API (`client.voice_changer.change_voice_bytes`) with flattened output format parameters (`output_format_container`, `output_format_sample_rate`, `output_format_encoding`, `output_format_bit_rate`).

---

## Supported Models

| Model ID | Description |
|---|---|
| `sonic-3` | Latest model, highest quality |
| `sonic-turbo` | Faster variant, lower latency |

Models are not hardcoded in the provider -- any model ID accepted by the Cartesia API can be passed via `config.model`.

---

## Output Format Details

The `AudioOutputFormat` fields map to Cartesia's structured output format:

| AudioOutputFormat field | Cartesia API field | Default | Notes |
|---|---|---|---|
| `format` | `container` | -- | `"pcm"` is mapped to `"raw"`. Other values (`mp3`, `wav`) pass through |
| `sample_rate` | `sample_rate` | 44100 | Sample rate in Hz |
| `bitrate` | `bit_rate` | 128000 | Converted from kbps to bps (multiplied by 1000). Only used for non-wav/pcm formats |

For `wav` and `pcm` formats, the encoding is always set to `pcm_s16le`. Bitrate is ignored for these formats (a warning is logged if provided).

---

## Provider-Specific Notes

**Authentication:** Always pass `api_key` explicitly in `AudioGenerationConfig`. There is no automatic environment variable fallback.

**Bitrate limitation:** Cartesia does not support bitrate for `wav` and `pcm` container formats. If `bitrate` is set in `AudioOutputFormat` for these formats, it is silently ignored and a warning is logged.

**No duration metadata:** The Cartesia API does not return audio duration. The `duration` field in `TTSResponse` / `STSResponse` will be `None`.

**Client lifecycle:** A fresh Cartesia client is created per request to avoid event loop issues with the async client.

**Extra parameters:** Both TTS and STS requests support `extra_params` for passing additional provider-specific fields directly to the Cartesia API.
