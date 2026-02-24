# Azure OpenAI

Azure OpenAI provides Sora video generation through Microsoft Azure's managed OpenAI service. It inherits all capabilities from the [OpenAI provider](openai.md) but authenticates against an Azure endpoint.

## Capabilities

| Feature | Supported |
|---|:---:|
| Video generation | ✅ |
| Image generation | ✅ |
| Image-to-video | ✅ |
| Async | ✅ |
| Progress callbacks | ✅ |

## Configuration

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="azure_openai",
    model="sora-2",             # Deployment name, not model name
    api_key="...",              # Azure API key or AD token
    base_url="https://my-resource.openai.azure.com/",  # Required
    api_version="2024-05-01-preview",                  # Required
    timeout=600,
    max_poll_attempts=120,
    poll_interval=5,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | Azure API key or Azure AD token |
| `base_url` | `str` | — | **Required.** Azure OpenAI endpoint URL |
| `api_version` | `str` | `"2024-05-01-preview"` | Azure API version |
| `timeout` | `int` | `600` | Request timeout in seconds |
| `max_poll_attempts` | `int` | `120` | Max polling iterations |
| `poll_interval` | `int` | `5` | Seconds between status checks |

## Video Models

Azure OpenAI uses **deployment names** rather than model names. The deployment name is whatever name you gave your deployment in the Azure portal.

| Underlying Model | Typical Deployment Name | Duration Options |
|---|---|---|
| Sora 2 | `sora-2` (user-defined) | 4s, 8s, 12s |

## Image Models

Same DALL-E and GPT Image models as the [OpenAI provider](openai.md#image-models), accessed via Azure deployments.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="azure_openai",
    model="my-sora-deployment",   # Your deployment name in Azure
    api_key="YOUR_AZURE_KEY",
    base_url="https://my-resource.openai.azure.com/",
    api_version="2024-05-01-preview",
)

request = VideoGenerationRequest(
    prompt="A timelapse of a city skyline from dawn to dusk",
    duration_seconds=8,
    aspect_ratio="16:9",
)

response = generate_video(config, request)
print(response.video)
```

## Supported Request Parameters

Same as the [OpenAI provider](openai.md#supported-request-parameters). All Sora parameters apply.

## Provider-Specific Notes

**`base_url` is required.** Unlike the standard OpenAI provider, Azure requires an explicit endpoint. The URL should be your Azure OpenAI resource endpoint:

```
https://<resource-name>.openai.azure.com/
```

**`api_version` extraction from URL:** If your `base_url` already contains `?api-version=...` as a query parameter, it will be extracted automatically.

**Deployment names vs. model names:** In Azure, you deploy a model under a custom name. Use that deployment name as `model` in the config, not the underlying OpenAI model name.

**Authentication options:**
- **API key:** Set `api_key` to your Azure resource key.
- **Azure AD token:** Set `api_key` to a bearer token obtained via `azure-identity`.

**Same underlying client:** This provider uses `AzureOpenAI` / `AsyncAzureOpenAI` from the `openai` SDK. All polling and progress tracking behaviour is identical to the standard OpenAI provider.
