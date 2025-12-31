"""Provider implementations."""

from tarash.tarash_gateway.video.providers.azure_openai import (
    AzureOpenAIProviderHandler,
    AzureOpenAIVideoParams,
    parse_azure_video_status,
)
from tarash.tarash_gateway.video.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
    duration_field_mapper,
    extra_params_field_mapper,
    image_list_field_mapper,
    passthrough_field_mapper,
    single_image_field_mapper,
    video_url_field_mapper,
)
from tarash.tarash_gateway.video.providers.fal import (
    FalProviderHandler,
    FAL_MODEL_REGISTRY,
    get_field_mappers,
    MINIMAX_FIELD_MAPPERS,
    KLING_VIDEO_V26_FIELD_MAPPERS,
    GENERIC_FIELD_MAPPERS,
    parse_fal_status,
)
from tarash.tarash_gateway.video.providers.openai import (
    OpenAIProviderHandler,
    OpenAIVideoParams,
    parse_openai_video_status,
)
from tarash.tarash_gateway.video.providers.veo3 import (
    Veo3ProviderHandler,
    Veo3VideoParams,
    parse_veo3_operation,
)

# Replicate imports are conditional due to pydantic v1 compatibility issues with Python 3.14+
try:
    from tarash.tarash_gateway.video.providers.replicate import (
        ReplicateProviderHandler,
        REPLICATE_MODEL_REGISTRY,
        get_replicate_field_mappers,
        parse_replicate_status,
        KLING_V21_FIELD_MAPPERS,
        LUMA_FIELD_MAPPERS,
        WAN_FIELD_MAPPERS,
        GENERIC_REPLICATE_FIELD_MAPPERS,
    )

    _REPLICATE_AVAILABLE = True
except Exception:
    # Replicate not available (likely due to pydantic v1 incompatibility with Python 3.14+)
    ReplicateProviderHandler = None  # type: ignore
    REPLICATE_MODEL_REGISTRY = {}  # type: ignore
    get_replicate_field_mappers = None  # type: ignore
    parse_replicate_status = None  # type: ignore
    KLING_V21_FIELD_MAPPERS = {}  # type: ignore
    LUMA_FIELD_MAPPERS = {}  # type: ignore
    WAN_FIELD_MAPPERS = {}  # type: ignore
    GENERIC_REPLICATE_FIELD_MAPPERS = {}  # type: ignore
    _REPLICATE_AVAILABLE = False

__all__ = [
    # Common Field Mappers
    "FieldMapper",
    "apply_field_mappers",
    "duration_field_mapper",
    "extra_params_field_mapper",
    "image_list_field_mapper",
    "passthrough_field_mapper",
    "single_image_field_mapper",
    "video_url_field_mapper",
    # Fal
    "FalProviderHandler",
    "FAL_MODEL_REGISTRY",
    "get_field_mappers",
    "MINIMAX_FIELD_MAPPERS",
    "KLING_VIDEO_V26_FIELD_MAPPERS",
    "GENERIC_FIELD_MAPPERS",
    "parse_fal_status",
    # OpenAI
    "OpenAIProviderHandler",
    "OpenAIVideoParams",
    "parse_openai_video_status",
    # Azure OpenAI
    "AzureOpenAIProviderHandler",
    "AzureOpenAIVideoParams",
    "parse_azure_video_status",
    # Replicate
    "ReplicateProviderHandler",
    "REPLICATE_MODEL_REGISTRY",
    "get_replicate_field_mappers",
    "parse_replicate_status",
    "KLING_V21_FIELD_MAPPERS",
    "LUMA_FIELD_MAPPERS",
    "WAN_FIELD_MAPPERS",
    "GENERIC_REPLICATE_FIELD_MAPPERS",
    # Veo3
    "Veo3ProviderHandler",
    "Veo3VideoParams",
    "parse_veo3_operation",
]
