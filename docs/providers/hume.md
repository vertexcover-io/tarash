# Hume AI

Hume AI provides expressive text-to-speech via **Octave** models with emotional control. Tarash uses the official `hume` Python SDK.

!!! info "TTS Only"
    Hume AI supports **text-to-speech (TTS)** only. Speech-to-speech (STS) is not available.

---

## Installation

Install tarash-gateway with the Hume extra:

```bash
pip install tarash-gateway[hume]
```

This installs the `hume` SDK (`>=0.9.0`).

---

## Quick Example

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, TTSRequest

config = AudioGenerationConfig(
    provider="hume",
    model="octave-2",
    api_key="YOUR_HUME_KEY",
)

request = TTSRequest(
    text="Welcome to expressive speech synthesis.",
    voice_id="Kora",
    voice_settings={"description": "excited and warm", "speed": 1.1},
)

response = generate_tts(config, request)
print(f"Duration: {response.duration}s")
```

---

## Supported Models

| Model ID | Version | Notes |
|---|---|---|
| `octave-2` / `hume-v2` | 2 | Latest Octave model |
| `octave-1` / `hume-v1` | 1 | Previous Octave model |

The model version is extracted automatically from the model name. For example, `octave-2` resolves to version `"2"` and `hume-v1` resolves to version `"1"`.

---

## Parameters

| Parameter | TTSRequest field | Notes |
|---|---|---|
| Text | `text` | The text to synthesize into speech |
| Voice | `voice_id` | Voice name (default) or voice ID (see Voice Lookup Modes below) |
| Output format | `output_format` | `AudioOutputFormat(format="mp3")` — passed as `{"type": format}` to the API |
| Description | `voice_settings={"description": "..."}` | Voice style guidance (e.g., `"excited and warm"`, `"calm and soothing"`) |
| Speed | `voice_settings={"speed": 1.2}` | Speech rate multiplier |
| Trailing silence | `voice_settings={"trailing_silence": 0.5}` | Seconds of silence appended after the utterance |

### Extra Parameters

Advanced or Hume-specific parameters can be passed through `extra_params`:

```python
request = TTSRequest(
    text="Hello world",
    voice_id="Kora",
    extra_params={
        "context": {"text": "Previously on the show..."},
    },
)
```

---

## Voice Lookup Modes

Hume supports three ways to specify a voice, controlled via `voice_settings`:

### Name Lookup (default)

By default, `voice_id` is treated as a voice **name** with the `HUME_AI` provider:

```python
request = TTSRequest(
    text="Hello!",
    voice_id="Kora",  # Looked up by name
)
# Sends: {"name": "Kora", "provider": "HUME_AI"}
```

### ID Lookup

Set `voice_id_mode` to `"id"` to look up the voice by its unique identifier:

```python
request = TTSRequest(
    text="Hello!",
    voice_id="a1b2c3d4-...",
    voice_settings={"voice_id_mode": "id"},
)
# Sends: {"id": "a1b2c3d4-...", "provider": "HUME_AI"}
```

### Custom Voice Provider

Override the voice provider with `voice_provider`:

```python
request = TTSRequest(
    text="Hello!",
    voice_id="my-custom-voice",
    voice_settings={"voice_provider": "CUSTOM"},
)
# Sends: {"name": "my-custom-voice", "provider": "CUSTOM"}
```

You can combine `voice_id_mode` and `voice_provider`:

```python
request = TTSRequest(
    text="Hello!",
    voice_id="a1b2c3d4-...",
    voice_settings={"voice_id_mode": "id", "voice_provider": "CUSTOM"},
)
# Sends: {"id": "a1b2c3d4-...", "provider": "CUSTOM"}
```

---

## Provider-Specific Notes

**SDK requirement:** Install the `hume` package via the extra: `pip install tarash-gateway[hume]`.

**Authentication:** `api_key` must be passed explicitly in `AudioGenerationConfig`. There is no automatic environment variable reading.

**Utterances:** Text is wrapped into a single `PostedUtterance` object internally. Voice specification, description, speed, and trailing silence are all set on the utterance.

**Response metadata:** The response includes `duration` from the Hume generation object, along with `generation_id`, `file_size`, and encoding details in `raw_response`.

**No streaming:** Hume returns the complete audio in a single response. The `on_progress` callback is accepted but unused.
