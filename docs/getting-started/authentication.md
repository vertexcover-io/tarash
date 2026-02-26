# Authentication

Pass your API key via `api_key` in the config:

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast",
    api_key="fal-xxxxxxxxxxxxxxxx",
)
```

!!! warning
    Never hardcode API keys in source files. Use environment variables or a secrets manager.

---

## Provider API key reference

| Provider | Where to get it |
|---|---|
| Fal.ai | [fal.ai/dashboard](https://fal.ai/dashboard) |
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Azure OpenAI | Azure Portal |
| Runway | [app.runwayml.com](https://app.runwayml.com) |
| Google (Gemini API) | [aistudio.google.com](https://aistudio.google.com/apikey) |
| Google (Vertex AI) | GCP project + service account or Application Default Credentials |
| Replicate | [replicate.com/account](https://replicate.com/account) |
| Stability AI | [platform.stability.ai](https://platform.stability.ai) |

!!! tip "Google models support two authentication modes"
    **Gemini Developer API (API key):** Works for both video (Veo) and image (Imagen, Gemini) models. Pass your `AIza...` key directly:

    ```python
    config = VideoGenerationConfig(
        provider="google",
        model="veo-3.0-generate-preview",
        api_key="AIza...",
    )
    ```

    **Vertex AI (Google Cloud):** Use this if you're running on GCP or need enterprise controls. Set `api_key=None` and supply `provider_config`:

    ```python
    config = VideoGenerationConfig(
        provider="google",
        model="veo-3.0-generate-preview",
        api_key=None,
        provider_config={
            "gcp_project": "my-gcp-project",
            "location": "us-central1",               # optional, defaults to us-central1
            "credentials_path": "/path/to/key.json", # optional, omit to use ADC
        },
    )
    ```

    Authentication is handled via a [service account JSON key](https://cloud.google.com/iam/docs/keys-create-delete)
    (`credentials_path`) or [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials)
    if `credentials_path` is omitted.
