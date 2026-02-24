# Authentication

Each provider requires its own API key. You can pass it directly or rely on the
provider's standard environment variable — Tarash passes `None` to the provider
SDK which then reads the env var automatically.

## Option 1: Pass the key directly

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    api_key="fal-xxxxxxxxxxxxxxxx",
)
```

!!! warning
    Never hardcode API keys in source files. Use environment variables or a secrets manager.

## Option 2: Use environment variables (recommended)

Set the key in your environment and omit `api_key` — the provider SDK reads it automatically:

```bash
export FAL_KEY="fal-xxxxxxxxxxxxxxxx"
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    # api_key omitted — FAL_KEY is read automatically
)
```

No `os.environ` calls needed. The provider SDK handles it.

## Provider env var reference

| Provider | Env var | Where to get it |
|---|---|---|
| Fal.ai | `FAL_KEY` | [fal.ai/dashboard](https://fal.ai/dashboard) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | Azure Portal |
| Runway | `RUNWAY_API_KEY` | [app.runwayml.com](https://app.runwayml.com) |
| Google (Veo3) | `GOOGLE_APPLICATION_CREDENTIALS` | GCP Service Account JSON path |
| Replicate | `REPLICATE_API_TOKEN` | [replicate.com/account](https://replicate.com/account) |
| Stability AI | `STABILITY_API_KEY` | [platform.stability.ai](https://platform.stability.ai) |
