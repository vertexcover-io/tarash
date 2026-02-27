"""Provider implementations."""

from tarash.tarash_gateway.providers.azure_openai import (
    AzureOpenAIProviderHandler,
    AzureOpenAIVideoParams,
)
from tarash.tarash_gateway.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
    duration_field_mapper,
    extra_params_field_mapper,
    image_list_field_mapper,
    passthrough_field_mapper,
    single_image_field_mapper,
    video_url_field_mapper,
)
from tarash.tarash_gateway.providers.fal import (
    FalProviderHandler,
    FAL_MODEL_REGISTRY,
    get_field_mappers,
    MINIMAX_FIELD_MAPPERS,
    KLING_VIDEO_V26_FIELD_MAPPERS,
    WAN_VIDEO_GENERATION_MAPPERS,
    WAN_ANIMATE_MAPPERS,
    KLING_O3_FIELD_MAPPERS,
    WAN_V22_A14B_FIELD_MAPPERS,
    BYTEDANCE_SEEDANCE_FIELD_MAPPERS,
    GENERIC_FIELD_MAPPERS,
    parse_fal_status,
)
from tarash.tarash_gateway.providers.openai import (
    OpenAIProviderHandler,
    OpenAIVideoParams,
    parse_openai_video_status,
)
from tarash.tarash_gateway.providers.replicate import (
    ReplicateProviderHandler,
    REPLICATE_MODEL_REGISTRY,
    get_replicate_field_mappers,
    parse_replicate_status,
    KLING_V21_FIELD_MAPPERS,
    LUMA_FIELD_MAPPERS,
    WAN_FIELD_MAPPERS,
    GENERIC_REPLICATE_FIELD_MAPPERS,
)
from tarash.tarash_gateway.providers.runway import (
    RunwayProviderHandler,
    RunwayVideoParams,
    parse_runway_task_status,
)
from tarash.tarash_gateway.providers.stability import (
    StabilityProviderHandler,
    SD35_LARGE_FIELD_MAPPERS,
    STABLE_IMAGE_ULTRA_FIELD_MAPPERS,
    STABILITY_IMAGE_MODEL_REGISTRY,
    get_stability_image_field_mappers,
)
from tarash.tarash_gateway.providers.google import (
    GoogleProviderHandler,
    # Video exports (formerly Veo3)
    Veo3VideoParams,  # Keep for backwards compatibility
    parse_veo3_operation,  # Keep for backwards compatibility
    # Image exports
    NANO_BANANA_FIELD_MAPPERS,
    IMAGEN3_FIELD_MAPPERS,
    GOOGLE_IMAGE_MODEL_REGISTRY,
    get_google_image_field_mappers,
)
from tarash.tarash_gateway.providers.xai import (
    XaiProviderHandler,
    parse_xai_video_status,
)
from tarash.tarash_gateway.providers.elevenlabs import (
    ElevenLabsProviderHandler,
)
from tarash.tarash_gateway.providers.cartesia import (
    CartesiaProviderHandler,
)

# Backwards compatibility alias
Veo3ProviderHandler = GoogleProviderHandler

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
    "WAN_VIDEO_GENERATION_MAPPERS",
    "WAN_ANIMATE_MAPPERS",
    "KLING_O3_FIELD_MAPPERS",
    "WAN_V22_A14B_FIELD_MAPPERS",
    "BYTEDANCE_SEEDANCE_FIELD_MAPPERS",
    "GENERIC_FIELD_MAPPERS",
    "parse_fal_status",
    # OpenAI
    "OpenAIProviderHandler",
    "OpenAIVideoParams",
    "parse_openai_video_status",
    # Azure OpenAI
    "AzureOpenAIProviderHandler",
    "AzureOpenAIVideoParams",
    # Replicate
    "ReplicateProviderHandler",
    "REPLICATE_MODEL_REGISTRY",
    "get_replicate_field_mappers",
    "parse_replicate_status",
    "KLING_V21_FIELD_MAPPERS",
    "LUMA_FIELD_MAPPERS",
    "WAN_FIELD_MAPPERS",
    "GENERIC_REPLICATE_FIELD_MAPPERS",
    # Runway
    "RunwayProviderHandler",
    "RunwayVideoParams",
    "parse_runway_task_status",
    # Stability
    "StabilityProviderHandler",
    "SD35_LARGE_FIELD_MAPPERS",
    "STABLE_IMAGE_ULTRA_FIELD_MAPPERS",
    "STABILITY_IMAGE_MODEL_REGISTRY",
    "get_stability_image_field_mappers",
    # Google (includes Veo3 for video, Imagen/Nano Banana for image)
    "GoogleProviderHandler",
    "Veo3ProviderHandler",  # Backwards compat alias
    "Veo3VideoParams",  # Backwards compat
    "parse_veo3_operation",  # Backwards compat
    "NANO_BANANA_FIELD_MAPPERS",
    "IMAGEN3_FIELD_MAPPERS",
    "GOOGLE_IMAGE_MODEL_REGISTRY",
    "get_google_image_field_mappers",
    # xAI
    "XaiProviderHandler",
    "parse_xai_video_status",
    # ElevenLabs
    "ElevenLabsProviderHandler",
    # Cartesia
    "CartesiaProviderHandler",
]
