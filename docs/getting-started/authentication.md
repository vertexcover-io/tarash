# Authentication

Each provider requires its own API key. Tarash accepts keys either directly in config or via
environment variables.

## Passing keys directly

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="fal",
    api_key="fal-xxxxxxxxxxxxxxxx",
)
```

!!! warning
    Avoid hardcoding keys in source code. Use environment variables instead.

## Using environment variables

Set the key in your environment and read it at runtime:

```bash
export FAL_KEY="fal-xxxxxxxxxxxxxxxx"
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
export RUNWAY_API_KEY="rw-xxxxxxxxxxxxxxxx"
```

```python
import os
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="fal",
    api_key=os.environ["FAL_KEY"],
)
```

## Provider API key reference

| Provider | Env var convention | Where to get it |
|---|---|---|
| Fal.ai | `FAL_KEY` | [fal.ai/dashboard](https://fal.ai/dashboard) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Runway | `RUNWAY_API_KEY` | [app.runwayml.com](https://app.runwayml.com) |
| Google (Veo3) | `GOOGLE_APPLICATION_CREDENTIALS` | GCP Service Account |
| Replicate | `REPLICATE_API_TOKEN` | [replicate.com/account](https://replicate.com/account) |
| Stability AI | `STABILITY_API_KEY` | [platform.stability.ai](https://platform.stability.ai) |
