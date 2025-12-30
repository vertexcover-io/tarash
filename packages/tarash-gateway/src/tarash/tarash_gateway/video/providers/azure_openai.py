"""Azure OpenAI provider handler for Sora video generation."""

from typing import Any

from typing_extensions import TypedDict

from tarash.tarash_gateway.video.exceptions import ProviderAPIError
from tarash.tarash_gateway.video.models import VideoGenerationConfig
from tarash.tarash_gateway.video.providers.openai import OpenAIProviderHandler
from tarash.tarash_gateway.video.providers.openai import parse_openai_video_status


try:
    from openai import AsyncAzureOpenAI, AzureOpenAI
except ImportError:
    AsyncAzureOpenAI = None  # type: ignore
    AzureOpenAI = None  # type: ignore


class AzureOpenAIVideoParams(TypedDict, total=False):
    """Azure OpenAI Sora-specific parameters."""

    # Additional parameters can be added here as Azure OpenAI expands the API
    pass


class AzureOpenAIProviderHandler(OpenAIProviderHandler):
    """Handler for Azure OpenAI Sora video generation.

    Azure OpenAI uses deployment names instead of model names and requires
    an Azure endpoint URL with API version.

    Inherits all conversion and generation logic from OpenAIProviderHandler,
    only overriding client initialization for Azure-specific authentication.
    """

    # Default API version for Azure OpenAI video generation
    DEFAULT_API_VERSION = "2024-05-01-preview"

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""
        if AzureOpenAI is None:
            raise ImportError(
                "openai is required for Azure OpenAI provider. "
                "Install with: pip install tarash-gateway[openai]"
            )

        self._sync_client_cache: dict[str, Any] = {}
        self._async_client_cache: dict[str, Any] = {}

    def _parse_azure_config(self, config: VideoGenerationConfig) -> dict[str, Any]:
        """Parse Azure-specific configuration from VideoGenerationConfig.

        Azure OpenAI requires:
        - azure_endpoint: The Azure OpenAI resource endpoint
        - api_version: API version string
        - api_key: Azure API key or Azure AD token

        These can be provided via:
        - base_url: Azure endpoint (e.g., https://my-resource.openai.azure.com/)
        - api_version: API version (e.g., "2024-05-01-preview")
        - model: The deployment name

        Args:
            config: Provider configuration

        Returns:
            Dict with Azure-specific client kwargs

        Raises:
            ProviderAPIError: If base_url is not provided
        """
        if not config.base_url:
            raise ProviderAPIError(
                "Azure OpenAI requires base_url to be set to your Azure endpoint "
                "(e.g., https://my-resource.openai.azure.com/)",
                provider=config.provider,
            )

        # Use api_version from config, or extract from base_url, or use default
        api_version = config.api_version or self.DEFAULT_API_VERSION
        azure_endpoint = config.base_url

        # If base_url contains api-version query param, extract it
        if "api-version=" in azure_endpoint:
            import urllib.parse

            parsed = urllib.parse.urlparse(azure_endpoint)
            query_params = urllib.parse.parse_qs(parsed.query)
            if "api-version" in query_params:
                api_version = query_params["api-version"][0]
            # Remove query params from endpoint
            azure_endpoint = urllib.parse.urlunparse(
                (parsed.scheme, parsed.netloc, parsed.path, "", "", "")
            )

        return {
            "azure_endpoint": azure_endpoint.rstrip("/"),
            "api_version": api_version,
            "api_key": config.api_key,
            "timeout": config.timeout,
        }

    def _get_client(
        self, config: VideoGenerationConfig, client_type: str
    ) -> "AsyncAzureOpenAI | AzureOpenAI":
        """
        Get or create Azure OpenAI client for the given config.

        Args:
            config: Provider configuration
            client_type: Type of client to return ("sync" or "async")

        Returns:
            AzureOpenAI (sync) or AsyncAzureOpenAI (async) client instance
        """
        # Use API key + base_url + api_version as cache key
        api_version = config.api_version or self.DEFAULT_API_VERSION
        cache_key = f"{config.api_key}:{config.base_url or 'default'}:{api_version}:{client_type}"

        azure_kwargs = self._parse_azure_config(config)

        if client_type == "async":
            if cache_key not in self._async_client_cache:
                self._async_client_cache[cache_key] = AsyncAzureOpenAI(**azure_kwargs)
            return self._async_client_cache[cache_key]
        else:  # sync
            if cache_key not in self._sync_client_cache:
                self._sync_client_cache[cache_key] = AzureOpenAI(**azure_kwargs)
            return self._sync_client_cache[cache_key]


# Export the parse function for compatibility
def parse_azure_video_status(video: Any):
    """Parse Azure OpenAI video object to VideoGenerationUpdate.

    For compatibility - Azure uses the same format as OpenAI.
    """
    return parse_openai_video_status(video)
