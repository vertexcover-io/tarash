# MiniMax Speech (via Fal.ai)

MiniMax Speech text-to-speech models hosted on Fal.ai, supporting high-quality speech synthesis with interjection support, voice customization, and 34+ languages.

## Quick Example

```python
from tarash.tarash_gateway import generate_tts
from tarash.tarash_gateway.models import AudioGenerationConfig, AudioOutputFormat, TTSRequest

config = AudioGenerationConfig(
    provider="fal",
    model="fal-ai/minimax/speech-2.8-hd",
    api_key="YOUR_FAL_KEY",
)

request = TTSRequest(
    text="Hello! (laughs) This is a test of the MiniMax Speech model.",
    voice_id="Wise_Woman",
    output_format=AudioOutputFormat(format="mp3", sample_rate=44100, bitrate=128),
)

response = generate_tts(config, request)
print(f"Duration: {response.duration}s")
```

---

## Supported Models

| Model | Quality | Notes |
|---|---|---|
| `fal-ai/minimax/speech-2.8-hd` | HD | Highest quality, interjection support |
| `fal-ai/minimax/speech-2.8-turbo` | Turbo | Faster generation with streaming |
| `fal-ai/minimax/speech-2.6-hd` | HD | Previous HD version |
| `fal-ai/minimax/speech-2.6-turbo` | Turbo | Previous turbo version |

---

## Parameters

| Parameter | TTSRequest field | Notes |
|---|---|---|
| Text | `text` | Supports `<#x#>` pauses (0.01–99.99s) and interjections: `(laughs)`, `(sighs)`, `(coughs)`, etc. |
| Voice | `voice_id` | Built-in voices: `Wise_Woman`, `Friendly_Person`, `Inspirational_girl`, etc. |
| Output format | `output_format` | `AudioOutputFormat(format="mp3", sample_rate=44100, bitrate=128)`, etc. |
| Language | `language_code` | Maps to `language_boost`: `English`, `French`, `Japanese`, `Chinese`, etc. (34+ languages) |
| Speed | `voice_settings={"speed": 1.2}` | Range: 0.5–2.0 |
| Volume | `voice_settings={"vol": 0.8}` | Range: 0.01–10 |
| Emotion | `voice_settings={"emotion": "happy"}` | `happy`, `sad`, `angry`, `fearful`, `disgusted`, `surprised`, `neutral` |
| Pitch | `voice_settings={"pitch": 3}` | Range: -12 to 12 |

### Extra Parameters

Advanced features go through `extra_params`:

```python
request = TTSRequest(
    text="Hello world",
    voice_id="Wise_Woman",
    voice_settings={"speed": 1.1, "emotion": "happy", "pitch": 3},
    extra_params={
        "voice_modify": {"pitch": 10, "intensity": 20, "timbre": -5},
        "pronunciation_dict": {"tone_list": ["hello/(heh-loh)"]},
    },
)
```

| Extra param | Type | Notes |
|---|---|---|
| `voice_modify` | `dict` | Fine-grained pitch/intensity/timbre control (-100 to 100) |
| `pronunciation_dict` | `dict` | Custom pronunciation replacements |
| `normalization_setting` | `dict` | Loudness normalization (enabled, target_loudness, etc.) |

---

## Interjections

The model supports natural interjections embedded in text:

- `(laughs)` — laughter
- `(sighs)` — sighing
- `(coughs)` — coughing
- `(clears throat)` — throat clearing
- `(gasps)` — gasping
- `(sniffs)` — sniffing
- `(groans)` — groaning
- `(yawns)` — yawning

```python
request = TTSRequest(
    text="Well (sighs) I suppose we should get started. (clears throat) Hello everyone!",
    voice_id="Wise_Woman",
)
```

---

## Pauses

Insert pauses with `<#x#>` syntax where `x` is seconds (0.01–99.99):

```python
request = TTSRequest(
    text="First point. <#1.5#> Second point. <#0.5#> And finally, the conclusion.",
    voice_id="Wise_Woman",
)
```
