# tarash-silence-remover

Remove silent segments from video and audio files. Uses Silero VAD (default) or FFmpeg silencedetect for detection, and FFmpeg for processing.

## Installation

```bash
pip install tarash-silence-remover           # FFmpeg detector only
pip install tarash-silence-remover[silero]    # With Silero VAD (recommended)
```

## Quick Start

```python
from tarash.tarash_silence_remover import remove_silence, SilenceRemovalConfig, SilenceRemovalRequest

config = SilenceRemovalConfig()
request = SilenceRemovalRequest(input_path="input.mp4")
response = remove_silence(config, request)

print(f"Removed {response.removed_duration:.1f}s of silence")
print(f"Output: {response.output_path}")
```
