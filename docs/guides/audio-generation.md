# Audio Generation

Tarash Gateway provides a unified interface for text-to-speech (TTS) and speech-to-speech (STS)
generation across ElevenLabs, Cartesia, Fal.ai, Sarvam, and Hume — using a
single `generate_tts` or `generate_sts` call regardless of provider.

Pass an [`AudioGenerationConfig`](../api-reference/models.md#audiogenerationconfig) and a
[`TTSRequest`](../api-reference/models.md#ttsrequest) to
[`generate_tts()`](../api-reference/gateway.md), or an
[`STSRequest`](../api-reference/models.md#stsrequest) to
[`generate_sts()`](../api-reference/gateway.md).

---

## Text to Speech (TTS)

### Basic TTS

Pass text and a voice ID to generate speech audio.

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, TTSRequest

config = AudioGenerationConfig(
    provider="elevenlabs",
    model="eleven_multilingual_v2",
    api_key="YOUR_ELEVENLABS_KEY",
)
request = TTSRequest(
    text="Welcome to Tarash, the unified AI media gateway.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
)
response = generate_tts(config, request)
print(response.content_type)  # e.g. "audio/mpeg"
```

[ElevenLabs models →](../providers/elevenlabs.md)

---

### With Output Format

Control the audio format, sample rate, and bitrate via `output_format`.

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, TTSRequest, AudioOutputFormat

config = AudioGenerationConfig(
    provider="cartesia",
    model="sonic-3",
    api_key="YOUR_CARTESIA_KEY",
)
request = TTSRequest(
    text="High quality audio with custom format settings.",
    voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
    output_format=AudioOutputFormat(format="wav", sample_rate=44100),
)
response = generate_tts(config, request)
print(response.content_type)  # "audio/wav"
```

[Cartesia models →](../providers/cartesia.md)

---

### With Voice Settings

Some providers accept `voice_settings` to fine-tune the generated speech.

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, TTSRequest

config = AudioGenerationConfig(
    provider="elevenlabs",
    model="eleven_multilingual_v2",
    api_key="YOUR_ELEVENLABS_KEY",
)
request = TTSRequest(
    text="This voice is tuned for maximum expressiveness.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    voice_settings={
        "stability": 0.3,
        "similarity_boost": 0.8,
        "style": 0.7,
        "use_speaker_boost": True,
    },
)
response = generate_tts(config, request)
```

[ElevenLabs voice settings →](../providers/elevenlabs.md)

---

### Multilingual TTS

Providers like Sarvam specialize in specific languages. Pass `language_code` to select the target language.

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, TTSRequest

config = AudioGenerationConfig(
    provider="sarvam",
    model="bulbul:v3",
    api_key="YOUR_SARVAM_KEY",
)
request = TTSRequest(
    text="नमस्ते, तरश गेटवे में आपका स्वागत है।",
    language_code="hi-IN",
)
response = generate_tts(config, request)
```

[Sarvam models →](../providers/sarvam.md)

---

### Expressive TTS

Hume's Octave models support expressive speech with emotion and speed control.

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, TTSRequest

config = AudioGenerationConfig(
    provider="hume",
    model="octave-2",
    api_key="YOUR_HUME_KEY",
)
request = TTSRequest(
    text="I'm so excited to tell you about this!",
    voice_id="Kora",
    voice_settings={
        "description": "excited and enthusiastic",
        "speed": 1.2,
    },
)
response = generate_tts(config, request)
```

[Hume models →](../providers/hume.md)

---

### TTS via Fal.ai

Fal.ai hosts MiniMax Speech and Qwen 3 TTS models.

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, TTSRequest

config = AudioGenerationConfig(
    provider="fal",
    model="fal-ai/minimax/speech-2.8-hd",
    api_key="YOUR_FAL_KEY",
)
request = TTSRequest(
    text="Hello from MiniMax Speech on Fal.",
    voice_id="male-qn-qingse",
)
response = generate_tts(config, request)
```

[MiniMax Speech →](../providers/fal/minimax-speech.md) · [Qwen 3 TTS →](../providers/fal/qwen-tts.md)

---

## Speech to Speech (STS)

STS takes an existing audio clip and re-synthesizes it in a different voice, preserving the
original speech content. Pass an [`STSRequest`](../api-reference/models.md#stsrequest)
with the source audio.

### Basic STS

```python
from tarash.tarash_gateway import generate_sts
from tarash.tarash_gateway.models import AudioGenerationConfig, STSRequest

config = AudioGenerationConfig(
    provider="elevenlabs",
    model="eleven_multilingual_v2",
    api_key="YOUR_ELEVENLABS_KEY",
)
request = STSRequest(
    audio="https://example.com/source-speech.mp3",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
)
response = generate_sts(config, request)
print(response.content_type)
```

[ElevenLabs STS →](../providers/elevenlabs.md)

---

### Voice Changer with Cartesia

Cartesia's STS uses its voice changer API to transform audio into a target voice.

```python
from tarash.tarash_gateway import generate_sts
from tarash.tarash_gateway.models import AudioGenerationConfig, STSRequest, AudioOutputFormat

config = AudioGenerationConfig(
    provider="cartesia",
    model="sonic-3",
    api_key="YOUR_CARTESIA_KEY",
)
request = STSRequest(
    audio="https://example.com/source-speech.wav",
    voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
    output_format=AudioOutputFormat(format="wav", sample_rate=44100),
)
response = generate_sts(config, request)
```

[Cartesia STS →](../providers/cartesia.md)

---

## Async Generation

Every function has an async variant. Use `generate_tts_async` or `generate_sts_async` for
non-blocking workflows.

```python
from tarash.tarash_gateway import generate_tts_async
from tarash.tarash_gateway.models import AudioGenerationConfig, TTSRequest

config = AudioGenerationConfig(
    provider="elevenlabs",
    model="eleven_flash_v2_5",
    api_key="YOUR_ELEVENLABS_KEY",
)
request = TTSRequest(
    text="Fast async generation with ElevenLabs Flash.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
)
response = await generate_tts_async(config, request)
```

---

## Provider Comparison

| Provider | TTS | STS | Highlights |
|----------|:---:|:---:|------------|
| **ElevenLabs** | ✅ | ✅ | 29 languages, voice cloning, fine-grained voice settings |
| **Cartesia** | ✅ | ✅ | Low-latency Sonic models, voice changer API |
| **Fal.ai** | ✅ | — | MiniMax Speech (interjections, pauses), Qwen 3 TTS |
| **Sarvam** | ✅ | — | 11 Indian languages, Bulbul models |
| **Hume** | ✅ | — | Expressive Octave models with emotion control |
