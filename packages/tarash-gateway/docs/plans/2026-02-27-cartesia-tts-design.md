# Cartesia TTS Provider Design

## Summary

Add Cartesia as a TTS and STS (Voice Changer) provider in tarash-gateway, following the proven ElevenLabs provider pattern.

## Decisions

- **Delivery method**: Bytes endpoint (complete audio response, not SSE streaming)
- **STS**: Yes, via Cartesia's Voice Changer API (`/voice-changer/bytes`)
- **Generation config** (emotion/speed/volume): Exposed via `extra_params`, no model changes
- **Output format**: Accept ElevenLabs-style strings (`"mp3_44100_128"`) and parse into Cartesia's structured format
- **Architecture**: Mirror ElevenLabs pattern exactly (Approach A). No field mappers, no base class abstraction.

## Cartesia Capabilities

### Models
| Model | Latency | Notes |
|-------|---------|-------|
| `sonic-3` | ~90ms | Flagship. Supports generation_config (emotion, speed, volume) |
| `sonic-turbo` | ~40ms | Speed-optimized. No generation_config |

### TTS Features
- 37 languages
- Generation config: volume (0.5-2.0), speed (0.6-1.5), 50+ emotions (sonic-3 only)
- Pronunciation dictionaries
- Output: WAV, MP3, raw PCM
- Encodings: pcm_f32le, pcm_s16le, pcm_mulaw, pcm_alaw

### Voice Changer (STS)
- Input: audio file (clip)
- Output: same format options as TTS
- Target voice by ID

### SDK
- Package: `cartesia` (Python 3.9+)
- Clients: `Cartesia`, `AsyncCartesia`
- Error types: BadRequestError (400), AuthenticationError (401), PermissionDeniedError (403), UnprocessableEntityError (422), RateLimitError (429), InternalServerError (>=500), APIConnectionError

## Provider Handler Design

### File: `providers/cartesia.py`

`CartesiaProviderHandler` implements the audio ProviderHandler protocol:

#### Client Management
- Fresh client per call (same as ElevenLabs, avoids event loop issues)
- `Cartesia(api_key=..., timeout=...)` for sync
- `AsyncCartesia(api_key=..., timeout=...)` for async

#### Output Format Parsing
Convert ElevenLabs-style strings to Cartesia structured dicts:
- `"mp3_44100_128"` → `{"container": "mp3", "sample_rate": 44100, "bit_rate": 128000}`
- `"wav_44100"` → `{"container": "wav", "encoding": "pcm_s16le", "sample_rate": 44100}`
- `"pcm_16000"` → `{"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}`
- Default: `{"container": "mp3", "sample_rate": 44100, "bit_rate": 128000}`

#### TTS Request Mapping
| TTSRequest field | Cartesia SDK param |
|------------------|--------------------|
| config.model | model_id |
| request.text | transcript |
| request.voice_id | voice = {"mode": "id", "id": voice_id} |
| request.output_format | output_format (parsed) |
| request.language_code | language |
| extra_params.generation_config | generation_config |
| extra_params.pronunciation_dict_id | pronunciation_dict_id |

#### STS Request Mapping (Voice Changer)
| STSRequest field | Cartesia SDK param |
|------------------|--------------------|
| request.audio | clip (resolved to bytes) |
| request.voice_id | voice = {"id": voice_id} |
| request.output_format | output_format (parsed) |

#### Error Mapping
| Cartesia Exception | Tarash Exception |
|-------------------|------------------|
| BadRequestError (400) | ValidationError |
| UnprocessableEntityError (422) | ValidationError |
| AuthenticationError (401) | HTTPError(401) |
| PermissionDeniedError (403) | ContentModerationError |
| RateLimitError (429) | HTTPError(429), retryable |
| InternalServerError (>=500) | HTTPError(500+), retryable |
| APIConnectionError | HTTPConnectionError, retryable |
| Timeout | TimeoutError, retryable |

#### Generation Flow
1. Create fresh client
2. Convert request to SDK kwargs
3. Generate unique request_id
4. Call `client.tts.generate(...)` → BinaryAPIResponse
5. Read response bytes
6. Base64 encode audio, derive content_type, return TTSResponse

## Registration

- `registry.py`: Add `"cartesia"` → `CartesiaProviderHandler()` in get_handler
- `providers/__init__.py`: Export `CartesiaProviderHandler`
- `conftest.py`: Add `CARTESIA_API_KEY` env var check for e2e skip logic

## Unit Tests (`tests/unit/audio/providers/test_cartesia.py`)

Sections:
1. Output format parsing (parametrized: mp3/wav/pcm/raw string → dict)
2. TTS request conversion (minimal, full with language + generation_config, extra_params)
3. STS request conversion (audio resolution: MediaContent, base64, URL)
4. Response conversion (TTS + STS, content type, base64 encoding)
5. Error mapping (all status codes + connection + timeout + unknown)
6. Integration tests (mocked SDK: sync/async TTS/STS success + error propagation)
7. Client creation (sync/async, with/without api_key)

## E2E Tests (`tests/e2e/test_cartesia.py`)

3 tests for maximum coverage:
1. **test_tts_async_sonic3_with_generation_config**: async, sonic-3, generation_config (emotion+speed+volume via extra_params), language, mp3 output
2. **test_tts_sync_sonic_turbo**: sync, sonic-turbo, minimal params, wav output
3. **test_sts_async_voice_changer**: TTS → Voice Changer pipeline, MediaContent input
