"""Fal.ai provider handler."""

import asyncio
import time
import traceback
from typing import TYPE_CHECKING, cast, Literal, overload

from fal_client import AsyncRequestHandle, SyncRequestHandle
from fal_client.client import FalClientHTTPError

import httpx
from tarash.tarash_gateway.logging import ProviderLogger, log_error
from tarash.tarash_gateway.exceptions import (
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.models import (
    AnyDict,
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageGenerationUpdate,
    ImageProgressCallback,
    ProgressCallback,
    SyncImageProgressCallback,
    SyncProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
    duration_field_mapper,
    extra_params_field_mapper,
    get_field_mappers_from_registry,
    image_list_field_mapper,
    passthrough_field_mapper,
    single_image_field_mapper,
    video_url_field_mapper,
)

try:
    import fal_client
    import httpx
    from fal_client import (
        AsyncClient,
        Status,
        SyncClient,
        Completed,
        Queued,
        InProgress,
    )

    has_fal_client = True
except ImportError:
    has_fal_client = False

if TYPE_CHECKING:
    import fal_client
    from fal_client import (
        AsyncClient,
        SyncClient,
        Status,
        Completed,
        Queued,
        InProgress,
    )

# Logger name constant
_LOGGER_NAME = "tarash.tarash_gateway.providers.fal"

# Provider name constant
_PROVIDER_NAME = "fal"

# ==================== Model Field Mappings ====================


# Minimax models field mappings
MINIMAX_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper(
        field_type="str", allowed_values=["6s", "10s"], provider="fal", model="minimax"
    ),
    "image_url": single_image_field_mapper(),
    "prompt_optimizer": passthrough_field_mapper("enhance_prompt"),
}

# Kling Video v2.6 models field mappings
# Supports both image-to-video and motion-control variants
KLING_VIDEO_V26_FIELD_MAPPERS: dict[str, FieldMapper] = {
    # Required for image-to-video
    "prompt": passthrough_field_mapper("prompt", required=True),
    "image_url": single_image_field_mapper(required=True, image_type="reference"),
    # Optional for image-to-video
    "duration": duration_field_mapper(field_type="str", allowed_values=["5", "10"]),
    "generate_audio": passthrough_field_mapper("generate_audio"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "tail_image_url": single_image_field_mapper(
        required=False, image_type="last_frame"
    ),
    "cfg_scale": extra_params_field_mapper("cfg_scale"),
    "voice_ids": extra_params_field_mapper("voice_ids"),
    # Motion control specific fields
    "video_url": video_url_field_mapper(),
    "character_orientation": extra_params_field_mapper("character_orientation"),
    "keep_original_sound": extra_params_field_mapper("keep_original_sound"),
}


# Kling Video O1 - Unified mapper for all variants
# Supports: image-to-video, reference-to-video, video-to-video/edit
# Reference: https://fal.ai/models/fal-ai/kling-video/o1/*
#
# Usage for elements (reference-to-video and video-to-video/edit):
#   Pass elements directly via extra_params in the exact format expected by Fal:
#   extra_params={
#       "elements": [
#           {
#               "frontal_image_url": "url",
#               "reference_image_urls": ["url1", "url2"]  # Optional, 1-4 additional angles
#           }
#       ]
#   }
KLING_O1_FIELD_MAPPERS: dict[str, FieldMapper] = {
    # Core required field (all variants)
    "prompt": passthrough_field_mapper("prompt", required=True),
    # Image-to-video: start/end frame support
    "start_image_url": single_image_field_mapper(
        required=False, image_type="first_frame"
    ),
    "end_image_url": single_image_field_mapper(required=False, image_type="last_frame"),
    # Reference-to-video and video-to-video/edit: elements support (passed via extra_params)
    "elements": extra_params_field_mapper("elements"),
    "image_urls": image_list_field_mapper(image_type="reference"),
    # Optional parameters (variant-specific)
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=["5", "10"],
        provider="fal",
        model="kling-o1",
        add_suffix=False,  # o1 uses "5", "10" without "s" suffix
    ),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    # Video-to-video/edit specific
    "video_url": video_url_field_mapper(required=False),
    "keep_audio": passthrough_field_mapper("keep_audio"),
}

# Kling Video O3 - Unified mapper for all variants
# Supports: text-to-video, image-to-video, reference-to-video (Pro and Standard tiers)
# Duration: 3-15 seconds (string format without "s" suffix)
#
# Image type conventions (controls which API field is used):
#   - image_type="reference" → image_url (for I2V)
#   - image_type="first_frame" → start_image_url (for R2V first frame)
#   - image_type="last_frame" → end_image_url (for I2V/R2V end frame)
#
# Reference-to-video usage:
#   Pass elements via extra_params in Fal's expected format:
#   extra_params={
#       "elements": [
#           {
#               "frontal_image_url": "url",
#               "reference_image_urls": ["url1", "url2"]  # Optional, 1-3 additional angles
#           }
#       ]
#   }
KLING_O3_FIELD_MAPPERS: dict[str, FieldMapper] = {
    # Core required (all variants)
    "prompt": passthrough_field_mapper("prompt", required=True),
    # Image-to-video: uses image_url (user provides image_type="reference")
    # strict=False allows R2V to use multiple reference images without error
    "image_url": single_image_field_mapper(
        required=False, image_type="reference", strict=False
    ),
    # Reference-to-video: uses start_image_url (user provides image_type="first_frame")
    "start_image_url": single_image_field_mapper(
        required=False, image_type="first_frame"
    ),
    # End frame support (both I2V and R2V)
    "end_image_url": single_image_field_mapper(required=False, image_type="last_frame"),
    # Reference-to-video: style/appearance references (max 4 total with elements)
    "image_urls": image_list_field_mapper(image_type="reference"),
    # Reference-to-video: character/object definitions (passed via extra_params)
    "elements": extra_params_field_mapper("elements"),
    # Common optional parameters
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=[str(i) for i in range(3, 16)],  # "3", "4", ..., "15"
        provider="fal",
        model="kling-o3",
        add_suffix=False,
    ),
    "generate_audio": passthrough_field_mapper("generate_audio"),
    # Text-to-video: voice support (max 2, reference as <<<voice_1>>> in prompt)
    "voice_ids": extra_params_field_mapper("voice_ids"),
    # Multi-shot support (all variants)
    "multi_prompt": extra_params_field_mapper("multi_prompt"),
    "shot_type": extra_params_field_mapper("shot_type"),
}

# Veo 3 and Veo 3.1 models field mappings
# Supports text-to-video, image-to-video, first-last-frame-to-video, and video-to-video (extend-video)
#
# Notes on extend-video:
# - The fal-ai/veo3.1/fast/extend-video endpoint has stricter constraints:
#   - Only supports 7s duration (vs 4s/6s/8s for other variants)
#   - Only supports 720p resolution
#   - Requires both prompt and video_url
# - These API-level constraints are enforced by Fal's API, not by field mappers
# - Use extra_params for extend-video specific values like aspect_ratio: "auto"
VEO3_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=["4s", "6s", "8s", "7s"],  # Added 7s for extend-video
        provider="fal",
        model="veo3",
    ),
    "generate_audio": passthrough_field_mapper("generate_audio"),
    "resolution": passthrough_field_mapper("resolution"),
    "auto_fix": passthrough_field_mapper("auto_fix"),
    "seed": passthrough_field_mapper("seed"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    # Image-to-video support
    "image_url": single_image_field_mapper(required=False, image_type="reference"),
    # First-last-frame-to-video support
    "first_frame_url": single_image_field_mapper(
        required=False, image_type="first_frame"
    ),
    "last_frame_url": single_image_field_mapper(
        required=False, image_type="last_frame"
    ),
    # Video-to-video support (extend-video)
    "video_url": video_url_field_mapper(),
}

# Sora 2 models field mappings
# Supports text-to-video, image-to-video, and video-to-video/remix variants
#
# Notes on video-to-video/remix:
# - The fal-ai/sora-2/video-to-video/remix endpoint remixes Sora-generated videos
# - Requires video_id (not video_url) - can only remix Sora 2 generated videos
# - Pass video_id via extra_params: extra_params={"video_id": "video_123"}
# - Does not use aspect_ratio, resolution, duration (inherited from original video)
SORA2_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "resolution": passthrough_field_mapper("resolution"),
    "duration": duration_field_mapper(
        field_type="int", allowed_values=[4, 8, 12], provider="fal", model="sora-2"
    ),
    "delete_video": passthrough_field_mapper("delete_video"),
    # Image-to-video support (optional - required only for image-to-video variant)
    "image_url": single_image_field_mapper(required=False, image_type="reference"),
    # Video-to-video/remix support (via extra_params)
    "video_id": extra_params_field_mapper("video_id"),
}

# Wan v2.6 and v2.5 - Unified mapper for all video generation endpoints
# Supports: text-to-video, image-to-video, reference-to-video
# Works with both v2.6 (wan/v2.6/*) and v2.5 (fal-ai/wan-25-preview/*)
WAN_VIDEO_GENERATION_MAPPERS: dict[str, FieldMapper] = {
    # Core parameters (all variants)
    "prompt": passthrough_field_mapper("prompt", required=True),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "resolution": passthrough_field_mapper("resolution"),
    "duration": duration_field_mapper(
        field_type="str", add_suffix=False
    ),  # Wan doesn't support "s" suffix
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "seed": passthrough_field_mapper("seed"),
    # Image/Video inputs (optional - used by I2V and R2V variants)
    "image_url": single_image_field_mapper(required=False, image_type="reference"),
    "video_urls": extra_params_field_mapper(
        "video_urls"
    ),  # For R2V with @Video1, @Video2, @Video3
    # Wan-specific features
    "audio_url": extra_params_field_mapper("audio_url"),
    "enable_prompt_expansion": passthrough_field_mapper("enhance_prompt"),
    "multi_shots": extra_params_field_mapper("multi_shots"),
    "enable_safety_checker": extra_params_field_mapper("enable_safety_checker"),
}

# Wan v2.2-14b Animate/Move - Video+Image to Video with motion control
WAN_ANIMATE_MAPPERS: dict[str, FieldMapper] = {
    # Required inputs
    "video_url": video_url_field_mapper(required=True),
    "image_url": single_image_field_mapper(required=True, image_type="reference"),
    # Generation parameters
    "resolution": passthrough_field_mapper("resolution"),
    "seed": passthrough_field_mapper("seed"),
    # Motion control parameters (via extra_params)
    "guidance_scale": extra_params_field_mapper("guidance_scale"),
    "num_inference_steps": extra_params_field_mapper("num_inference_steps"),
    "shift": extra_params_field_mapper("shift"),
    # Quality/output parameters
    "video_quality": extra_params_field_mapper("video_quality"),
    "video_write_mode": extra_params_field_mapper("video_write_mode"),
    "use_turbo": extra_params_field_mapper("use_turbo"),
    "return_frames_zip": extra_params_field_mapper("return_frames_zip"),
    # Safety
    "enable_safety_checker": extra_params_field_mapper("enable_safety_checker"),
    "enable_output_safety_checker": extra_params_field_mapper(
        "enable_output_safety_checker"
    ),
}

# Wan v2.2-a14b - Unified mapper for all v2.2-a14b endpoints
# Supports: text-to-video/lora, image-to-video, image-to-video/lora, video-to-video
# Uses num_frames instead of duration; has LoRA, interpolation, and acceleration support
WAN_V22_A14B_FIELD_MAPPERS: dict[str, FieldMapper] = {
    # Core parameters (all variants)
    "prompt": passthrough_field_mapper("prompt", required=True),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "seed": passthrough_field_mapper("seed"),
    "resolution": passthrough_field_mapper("resolution"),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "enable_prompt_expansion": passthrough_field_mapper("enhance_prompt"),
    # Image/Video inputs (optional - used by i2v and v2v variants)
    "image_url": single_image_field_mapper(required=False, image_type="reference"),
    "end_image_url": single_image_field_mapper(required=False, image_type="last_frame"),
    "video_url": video_url_field_mapper(required=False),
    # Generation parameters (via extra_params)
    "num_frames": extra_params_field_mapper("num_frames"),
    "frames_per_second": extra_params_field_mapper("frames_per_second"),
    "num_inference_steps": extra_params_field_mapper("num_inference_steps"),
    "guidance_scale": extra_params_field_mapper("guidance_scale"),
    "guidance_scale_2": extra_params_field_mapper("guidance_scale_2"),
    "shift": extra_params_field_mapper("shift"),
    "acceleration": extra_params_field_mapper("acceleration"),
    # Interpolation parameters
    "interpolator_model": extra_params_field_mapper("interpolator_model"),
    "num_interpolated_frames": extra_params_field_mapper("num_interpolated_frames"),
    "adjust_fps_for_interpolation": extra_params_field_mapper(
        "adjust_fps_for_interpolation"
    ),
    # Quality/output parameters
    "video_quality": extra_params_field_mapper("video_quality"),
    "video_write_mode": extra_params_field_mapper("video_write_mode"),
    # LoRA and variant-specific
    "loras": extra_params_field_mapper("loras"),
    "reverse_video": extra_params_field_mapper("reverse_video"),
    "strength": extra_params_field_mapper("strength"),  # v2v only
    "resample_fps": extra_params_field_mapper("resample_fps"),  # v2v only
    # Safety
    "enable_safety_checker": extra_params_field_mapper("enable_safety_checker"),
    "enable_output_safety_checker": extra_params_field_mapper(
        "enable_output_safety_checker"
    ),
}

# ByteDance Seedance - Unified mapper for all versions and variants
# Supports v1 (text-to-video, image-to-video, reference-to-video) and v1.5 (text-to-video)
# Works with all ByteDance Seedance models regardless of version
BYTEDANCE_SEEDANCE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    # Core required
    "prompt": passthrough_field_mapper("prompt", required=True),
    # Core optional (all variants)
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "resolution": passthrough_field_mapper("resolution"),
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        provider="fal",
        model="bytedance-seedance",
        add_suffix=False,  # ByteDance uses "2", "3", etc. without "s" suffix
    ),
    "camera_fixed": passthrough_field_mapper("camera_fixed"),
    "seed": passthrough_field_mapper("seed"),
    "generate_audio": passthrough_field_mapper(
        "generate_audio"
    ),  # v1.5 only, ignored by v1
    "enable_safety_checker": extra_params_field_mapper("enable_safety_checker"),
    # Image-to-video support (v1/pro/image-to-video)
    # Using strict=False to allow reference-to-video with multiple reference images
    # When there's 1 reference image, both image_url and reference_image_urls work
    # When there are multiple, image_url returns None and reference_image_urls gets the list
    "image_url": single_image_field_mapper(
        required=False, image_type="reference", strict=False
    ),
    "end_image_url": single_image_field_mapper(required=False, image_type="last_frame"),
    # Reference-to-video support (v1/lite/reference-to-video)
    "reference_image_urls": image_list_field_mapper(image_type="reference"),
}

# Pixverse (v5 and v5.5) - Unified mapper for all variants
# Supports text-to-video, image-to-video, transition, effects, and swap
# All variants share common fields with variant-specific optional fields
PIXVERSE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    # Core fields (text-to-video, image-to-video, transition)
    "prompt": passthrough_field_mapper(
        "prompt", required=False
    ),  # Required for most, but not swap/effects
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "resolution": passthrough_field_mapper("resolution"),
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=["5", "8", "10"],
        provider="fal",
        model="pixverse-v5.5",
        add_suffix=False,  # Pixverse uses "5", "8", "10" without "s" suffix
    ),
    "style": passthrough_field_mapper("style"),
    "thinking_type": passthrough_field_mapper("thinking_type"),
    "seed": passthrough_field_mapper("seed"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    # Audio generation (text-to-video, image-to-video, transition)
    "generate_audio_switch": passthrough_field_mapper("generate_audio_switch"),
    # Multi-clip generation (text-to-video, image-to-video)
    "generate_multi_clip_switch": passthrough_field_mapper(
        "generate_multi_clip_switch"
    ),
    # Image-to-video / transition / effects / swap support
    "image_url": single_image_field_mapper(required=False, image_type="reference"),
    # Transition support (first and end frames)
    "first_image_url": single_image_field_mapper(
        required=False, image_type="first_frame"
    ),
    "end_image_url": single_image_field_mapper(required=False, image_type="last_frame"),
    # Effects variant
    "effect": extra_params_field_mapper("effect"),
    # Swap variant
    "video_url": video_url_field_mapper(required=False),
    "mode": extra_params_field_mapper("mode"),
    "keyframe_id": extra_params_field_mapper("keyframe_id"),
    "original_sound_switch": extra_params_field_mapper("original_sound_switch"),
}

# Generic fallback field mappings
GENERIC_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper(field_type="int"),
    "resolution": passthrough_field_mapper("resolution"),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "image_urls": image_list_field_mapper(),
    "video_url": video_url_field_mapper(),
    "seed": passthrough_field_mapper("seed"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "generate_audio": passthrough_field_mapper("generate_audio"),
}


# ==================== Model Registry ====================


# Registry maps model names (or prefixes) to their field mappers
FAL_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    # Minimax - all variants use same field mappings
    "fal-ai/minimax": MINIMAX_FIELD_MAPPERS,
    # Kling Video O1 - All variants (image-to-video, reference-to-video, video-to-video/edit) use unified mapper
    "fal-ai/kling-video/o1": KLING_O1_FIELD_MAPPERS,
    # Kling Video O3 - All variants (text-to-video, image-to-video, reference-to-video) for Pro and Standard tiers
    "fal-ai/kling-video/o3/": KLING_O3_FIELD_MAPPERS,
    # Kling Video v2.6 - supports both image-to-video and motion-control
    "fal-ai/kling-video/v2.6": KLING_VIDEO_V26_FIELD_MAPPERS,
    # Veo 3.1 - prefix must be registered before veo3 for longest-match precedence
    # Supports all variants: text-to-video, image-to-video, first-last-frame-to-video, extend-video
    "fal-ai/veo3.1": VEO3_FIELD_MAPPERS,
    # Veo 3 - supports text-to-video, image-to-video, first-last-frame-to-video, and extend-video
    "fal-ai/veo3": VEO3_FIELD_MAPPERS,
    # Sora 2 - supports both text-to-video and image-to-video variants
    "fal-ai/sora-2": SORA2_FIELD_MAPPERS,
    # Wan v2.6 - All variants (text-to-video, image-to-video, reference-to-video) use unified mapper
    "wan/v2.6/": WAN_VIDEO_GENERATION_MAPPERS,
    # Wan v2.5 - Uses same unified mapper as v2.6
    "fal-ai/wan-25-preview/": WAN_VIDEO_GENERATION_MAPPERS,
    # Wan v2.2-14b Animate
    "fal-ai/wan/v2.2-14b/animate/": WAN_ANIMATE_MAPPERS,
    # Wan v2.2-a14b - All variants (text-to-video/lora, image-to-video, image-to-video/lora, video-to-video)
    "fal-ai/wan/v2.2-a14b/": WAN_V22_A14B_FIELD_MAPPERS,
    # ByteDance Seedance - Unified mapper for all versions (v1, v1.5) and variants (text-to-video, image-to-video, reference-to-video)
    "fal-ai/bytedance/seedance": BYTEDANCE_SEEDANCE_FIELD_MAPPERS,
    # Pixverse - All variants (text-to-video, image-to-video, transition, effects, swap) use unified mapper
    # Supports both v5 and v5.5 with same field mappings
    "fal-ai/pixverse/v5.5": PIXVERSE_FIELD_MAPPERS,  # v5.5 variants
    "fal-ai/pixverse/v5": PIXVERSE_FIELD_MAPPERS,  # v5 variants (same API)
    "fal-ai/pixverse/swap": PIXVERSE_FIELD_MAPPERS,  # Swap variant (works for both v5 and v5.5)
    # Future models...
    # "fal-ai/hunyuan-video": HUNYUAN_FIELD_MAPPERS,
}

# ==================== Image Generation Field Mappings ====================

# FLUX models field mappings (common for dev, schnell, pro)
FLUX_IMAGE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "seed": passthrough_field_mapper("seed"),
    "image_size": passthrough_field_mapper(
        "size"
    ),  # "square_hd", "landscape_4_3", etc.
    "num_images": passthrough_field_mapper("n"),
}

# Recraft V3 field mappings
RECRAFT_IMAGE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "seed": passthrough_field_mapper("seed"),
    "image_size": passthrough_field_mapper("size"),
    "n": passthrough_field_mapper("n"),
    "style": passthrough_field_mapper(
        "style"
    ),  # "realistic_image", "digital_illustration", etc.
}

# Ideogram V3 field mappings
IDEOGRAM_IMAGE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "seed": passthrough_field_mapper("seed"),
    "image_size": passthrough_field_mapper("size"),
    "num_images": passthrough_field_mapper("n"),
    "style_type": passthrough_field_mapper(
        "style"
    ),  # "AUTO", "GENERAL", "REALISTIC", etc.
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
}

# FLUX.2 Pro/Dev/Flex field mappings
FLUX2_PRO_IMAGE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "seed": passthrough_field_mapper("seed"),
    "size": passthrough_field_mapper("size"),
    "n": passthrough_field_mapper("n"),
    "reference_images": image_list_field_mapper(
        image_type="reference"
    ),  # Multi-reference support
    "guidance_scale": extra_params_field_mapper("guidance_scale"),  # Range: 1.0-20.0
    "num_inference_steps": extra_params_field_mapper(
        "num_inference_steps"
    ),  # Range: 1-50
}


# Custom converter for aspect_ratio with extra_params fallback
def _aspect_ratio_with_extra_params_converter(
    request: ImageGenerationRequest, value: object
) -> str | None:
    """Convert aspect_ratio field, preferring extra_params if present."""
    # First check extra_params for override
    if request.extra_params and "aspect_ratio" in request.extra_params:
        return str(request.extra_params["aspect_ratio"])
    # Fall back to request.aspect_ratio
    return str(value) if value is not None else None


# Flux 1.1 Pro Ultra/Raw field mappings
FLUX_PRO_ULTRA_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "seed": passthrough_field_mapper("seed"),
    "aspect_ratio": FieldMapper(
        source_field="aspect_ratio",
        converter=_aspect_ratio_with_extra_params_converter,
    ),  # Supports 21:9, 16:9, 4:3, 1:1, 3:4, 9:16, 9:21
    "raw": extra_params_field_mapper("raw"),  # Boolean for natural aesthetic
    "safety_tolerance": extra_params_field_mapper("safety_tolerance"),  # Range: 1-6
    "output_format": extra_params_field_mapper("output_format"),  # jpeg or png
}

# Z-Image-Turbo field mappings (distilled model optimized for speed)
ZIMAGE_TURBO_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "seed": passthrough_field_mapper("seed"),
    "size": passthrough_field_mapper("size"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "num_inference_steps": extra_params_field_mapper(
        "num_inference_steps"
    ),  # Default: 8 (distilled)
    "enable_safety_checker": extra_params_field_mapper("enable_safety_checker"),
}

# Generic image field mappings (fallback)
GENERIC_IMAGE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "seed": passthrough_field_mapper("seed"),
    "image_size": passthrough_field_mapper("size"),
    "num_images": passthrough_field_mapper("n"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
}

# Image model registry for Fal
FAL_IMAGE_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    # FLUX models
    "fal-ai/flux": FLUX_IMAGE_FIELD_MAPPERS,  # Prefix match for all flux variants
    # FLUX.2 models (newer generation with multi-reference support)
    "fal-ai/flux-2": FLUX2_PRO_IMAGE_FIELD_MAPPERS,
    # Flux 1.1 Pro Ultra/Raw (ultra high quality with extended aspect ratios)
    "fal-ai/flux-pro/v1.1-ultra": FLUX_PRO_ULTRA_FIELD_MAPPERS,
    "fal-ai/flux-pro/v1.1-raw": FLUX_PRO_ULTRA_FIELD_MAPPERS,
    # Z-Image-Turbo (distilled model optimized for speed)
    "fal-ai/z-image-turbo": ZIMAGE_TURBO_FIELD_MAPPERS,
    # Recraft
    "fal-ai/recraft-v3": RECRAFT_IMAGE_FIELD_MAPPERS,
    "fal-ai/recraft": RECRAFT_IMAGE_FIELD_MAPPERS,
    # Ideogram
    "fal-ai/ideogram": IDEOGRAM_IMAGE_FIELD_MAPPERS,
}


def get_image_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get the field mappers for a given image model.

    Args:
        model_name: Full model name (e.g., "fal-ai/flux/dev")

    Returns:
        Dict mapping API field names to FieldMapper objects
    """
    return get_field_mappers_from_registry(
        model_name, FAL_IMAGE_MODEL_REGISTRY, GENERIC_IMAGE_FIELD_MAPPERS
    )


def parse_fal_image_status(request_id: str, status: Status) -> ImageGenerationUpdate:
    """Parse Fal status update into ImageGenerationUpdate."""
    if isinstance(status, Completed):
        return ImageGenerationUpdate(
            request_id=request_id,
            status="completed",
            progress_percent=100,
            update={
                "metrics": status.metrics,
                "logs": status.logs,
            },
        )
    elif isinstance(status, Queued):
        return ImageGenerationUpdate(
            request_id=request_id,
            status="queued",
            progress_percent=None,
            update={"position": status.position},
        )
    elif isinstance(status, InProgress):
        return ImageGenerationUpdate(
            request_id=request_id,
            status="processing",
            progress_percent=None,
            update={"logs": status.logs},
        )
    else:
        raise ValueError(f"Unknown status: {status}")


def get_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get the field mappers for a given model.

    Lookup Strategy:
    1. Try exact match first
    2. If not found, try prefix matching - find registry keys that are prefixes
       of the model_name, and use the longest matching prefix
    3. If no match found, return generic field mappers as fallback

    Examples:
    --------
    >>> get_field_mappers("fal-ai/minimax")
    MINIMAX_FIELD_MAPPERS  # Exact match

    >>> get_field_mappers("fal-ai/minimax-video")
    MINIMAX_FIELD_MAPPERS  # Prefix match

    >>> get_field_mappers("fal-ai/minimax/hailuo-02-fast/image-to-video")
    MINIMAX_FIELD_MAPPERS  # Prefix match

    >>> get_field_mappers("fal-ai/veo3.1/fast")
    VEO31_FIELD_MAPPERS  # Prefix match

    >>> get_field_mappers("fal-ai/unknown-model")
    GENERIC_FIELD_MAPPERS  # Fallback for unknown models

    Args:
        model_name: Full model name (e.g., "fal-ai/minimax/hailuo-02-fast/image-to-video")

    Returns:
        Dict mapping API field names to FieldMapper objects
    """
    return get_field_mappers_from_registry(
        model_name, FAL_MODEL_REGISTRY, GENERIC_FIELD_MAPPERS
    )


def parse_fal_status(request_id: str, status: Status) -> VideoGenerationUpdate:
    """Parse Fal status update into VideoGenerationUpdate."""
    if isinstance(status, Completed):
        return VideoGenerationUpdate(
            request_id=request_id,
            status="completed",
            progress_percent=100,
            update={
                "metrics": status.metrics,
                "logs": status.logs,
            },
        )
    elif isinstance(status, Queued):
        return VideoGenerationUpdate(
            request_id=request_id,
            status="queued",
            progress_percent=None,
            update={"position": status.position},
        )
    elif isinstance(status, InProgress):
        return VideoGenerationUpdate(
            request_id=request_id,
            status="processing",
            progress_percent=None,
            update={"logs": status.logs},
        )
    else:
        raise ValueError(f"Unknown status: {status}")


# ==================== Provider Handler ====================


class FalProviderHandler:
    """Handler for Fal.ai provider."""

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""
        if not has_fal_client:
            raise ImportError(
                "fal-client is required for Fal provider. "
                + "Install with: pip install tarash-gateway[fal]"
            )

        self._sync_client_cache: dict[str, SyncClient] = {}
        self._async_client_cache: dict[str, AsyncClient] = {}
        # Note: AsyncClient is NOT cached to avoid "Event Loop closed" errors
        # Each async request creates a new client to ensure proper cleanup

    @overload
    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["async"]
    ) -> AsyncClient: ...

    @overload
    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["sync"]
    ) -> SyncClient: ...

    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["sync", "async"]
    ) -> AsyncClient | SyncClient:
        """
        Get or create Fal client for the given config.

        Args:
            config: Provider configuration
            client_type: Type of client to return ("sync" or "async")

        Returns:
            fal_client.SyncClient or fal_client.AsyncClient instance

        Note:
            AsyncClient instances are created fresh for each request to avoid
            "Event Loop closed" errors that occur when cached clients outlive
            the event loop they were created in.
        """
        if not has_fal_client:
            raise ImportError(
                "fal-client is required for Fal provider. "
                + "Install with: pip install tarash-gateway[fal]"
            )

        # Use API key + base_url as cache key
        cache_key = f"{config.api_key}:{config.base_url or 'default'}"

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)

        if client_type == "async":
            # Don't cache AsyncClient - create new instance for each request
            # This prevents "Event Loop closed" errors
            logger.debug(
                "Creating new async Fal client",
                {"base_url": config.base_url or "default"},
            )
            return fal_client.AsyncClient(
                key=config.api_key,
                default_timeout=config.timeout,
            )
        else:  # sync
            if cache_key not in self._sync_client_cache:
                logger.debug(
                    "Creating new sync Fal client",
                    {"base_url": config.base_url or "default"},
                )
                self._sync_client_cache[cache_key] = fal_client.SyncClient(
                    key=config.api_key,
                    default_timeout=config.timeout,
                )
            return self._sync_client_cache[cache_key]

    def _convert_request(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
    ) -> AnyDict:
        """Convert VideoGenerationRequest to model-specific format.

        Process:
        1. Get model-specific field mappers from registry (with prefix matching)
        2. Apply field mappers to convert request to API format
        3. Merge with extra_params (allows manual overrides)

        Args:
            config: Provider configuration
            request: Generic video generation request

        Returns:
            Model-specific validated request dictionary

        Raises:
            ValueError: If validation fails during field mapping
        """
        # Get model-specific field mappers (with prefix matching)
        field_mappers = get_field_mappers(config.model)

        # Apply field mappers to convert request
        api_payload = apply_field_mappers(field_mappers, request)

        # Merge with extra_params (allows manual overrides)
        api_payload.update(request.extra_params)

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.info(
            "Mapped request to provider format",
            {"converted_request": api_payload},
            redact=True,
        )

        return api_payload

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        _request: VideoGenerationRequest,
        request_id: str,
        provider_response: AnyDict,
    ) -> VideoGenerationResponse:
        """
        Convert Fal response to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request: Original video generation request
            request_id: Our request ID
            provider_response: Raw Fal response

        Returns:
            Normalized VideoGenerationResponse
        """
        # Extract video URL from Fal response
        # Fal returns: {"video": {"url": "..."}, ...} or {"video_url": "..."}
        video_url = None
        audio_url = None

        # Try different response formats
        if "video" in provider_response and isinstance(
            provider_response["video"], dict
        ):
            video_dict = cast(dict[str, object], provider_response["video"])
            video_url = cast(str, video_dict.get("url"))
        elif "video_url" in provider_response:
            video_url = cast(str, provider_response["video_url"])

        if "audio" in provider_response and isinstance(
            provider_response["audio"], dict
        ):
            audio_dict = cast(dict[str, object], provider_response["audio"])
            audio_url = cast(str, audio_dict.get("url"))
        elif "audio_url" in provider_response:
            audio_url = cast(str, provider_response["audio_url"])

        if not video_url:
            raise GenerationFailedError(
                f"No video URL found in Fal response: {provider_response}",
                provider=config.provider,
                model=config.model,
                raw_response=provider_response,
            )

        return VideoGenerationResponse(
            request_id=request_id,
            video=video_url,
            audio_url=audio_url,
            duration=provider_response.get("duration"),
            resolution=cast(str, provider_response.get("resolution")),
            aspect_ratio=cast(str, provider_response.get("aspect_ratio")),
            status="completed",
            raw_response=provider_response,
            provider_metadata={},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        _request: VideoGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Handle errors during video generation."""
        if isinstance(ex, TarashException):
            return ex

        # httpx timeout errors
        if isinstance(ex, httpx.TimeoutException):
            return TimeoutError(
                f"Request timed out: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
                timeout_seconds=config.timeout,
            )

        # httpx connection errors
        if isinstance(ex, (httpx.ConnectError, httpx.NetworkError)):
            return HTTPConnectionError(
                f"Connection error: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
            )

        # Fal HTTP errors
        if isinstance(ex, FalClientHTTPError):
            raw_response: dict[str, object] = {
                "status_code": ex.status_code,
                "response_headers": ex.response_headers,
                "response": ex.response.content,
            }

            # Validation errors (400, 422)
            if ex.status_code in (400, 422):
                return ValidationError(
                    ex.message,
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                )

            # All other HTTP errors (401, 403, 429, 500, 503, etc.)
            return HTTPError(
                f"HTTP {ex.status_code}: {ex.message}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=raw_response,
                status_code=ex.status_code,
            )

        # Unknown errors
        log_error(
            f"Fal unknown error: {str(ex)}",
            context={
                "provider": config.provider,
                "model": config.model,
                "request_id": request_id,
                "error_type": type(ex).__name__,
            },
            logger_name=_LOGGER_NAME,
            exc_info=True,
        )
        return GenerationFailedError(
            f"Error while generating video: {str(ex)}",
            provider=config.provider,
            model=config.model,
            request_id=request_id,
            raw_response={
                "error": str(ex),
                "traceback": traceback.format_exc(),
            },
        )

    def _process_event(
        self,
        config: VideoGenerationConfig,
        request_id: str,
        event: Status,
        start_time: float,
    ) -> VideoGenerationUpdate:
        """Process a single event and log it with elapsed time.

        Args:
            config: Provider configuration
            request_id: Request ID
            event: Fal status event
            start_time: Start time for elapsed time calculation

        Returns:
            VideoGenerationUpdate object
        """
        update = parse_fal_status(request_id, event)
        elapsed_time = time.time() - start_time

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME, request_id)
        logger.info(
            "Progress status update",
            {
                "status": update.status,
                "progress_percent": update.progress_percent,
                "time_elapsed_seconds": round(elapsed_time, 2),
            },
        )

        return update

    async def _process_events_async(
        self,
        config: VideoGenerationConfig,
        request_id: str,
        handler: AsyncRequestHandle,
        on_progress: ProgressCallback | None = None,
    ) -> AnyDict:
        """Process events asynchronously and return final result.

        Args:
            config: Provider configuration
            request_id: Request ID
            handler: Fal async handler
            on_progress: Optional async progress callback

        Returns:
            Final result from handler.get()
        """
        start_time = time.time()

        async for event in handler.iter_events(
            with_logs=True, interval=config.poll_interval
        ):
            update = self._process_event(config, request_id, event, start_time)

            if on_progress:
                result = on_progress(update)
                if asyncio.iscoroutine(result):
                    await result

        return await handler.get()

    def _process_events_sync(
        self,
        config: VideoGenerationConfig,
        request_id: str,
        handler: SyncRequestHandle,
        on_progress: SyncProgressCallback | None = None,
    ) -> object:
        """Process events synchronously and return final result.

        Args:
            config: Provider configuration
            request_id: Request ID
            handler: Fal sync handler
            on_progress: Optional progress callback

        Returns:
            Final result from handler.get()
        """
        start_time = time.time()

        for event in handler.iter_events(with_logs=True, interval=config.poll_interval):
            update = self._process_event(config, request_id, event, start_time)

            if on_progress:
                on_progress(update)

        return handler.get()

    @handle_video_generation_errors
    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video asynchronously via Fal with async progress callback.

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional async callback for progress updates

        Returns:
            Final VideoGenerationResponse when complete
        """
        client = self._get_client(config, "async")
        # Build Fal input (let validation errors propagate)
        fal_input = self._convert_request(config, request)

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug("Starting API call")

        # Submit to Fal using async API
        handler = await client.submit(
            config.model,
            arguments=fal_input,
        )

        request_id = handler.request_id
        logger = logger.with_request_id(request_id)
        logger.debug("Request submitted")

        try:
            result = await self._process_events_async(
                config, request_id, handler, on_progress
            )

            logger.debug("Request complete", {"response": result}, redact=True)

            # Parse response
            response = self._convert_response(config, request, request_id, result)

            logger.info("Final generated response", {"response": response}, redact=True)

            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: SyncProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video synchronously (blocking).

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional callback for progress updates

        Returns:
            Final VideoGenerationResponse
        """
        client = self._get_client(config, "sync")

        # Build Fal input (let validation errors propagate)
        fal_input = self._convert_request(config, request)

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug("Starting API call")

        # Submit to Fal
        handler = client.submit(
            config.model,
            arguments=fal_input,
        )
        request_id = handler.request_id
        logger = logger.with_request_id(request_id)
        logger.debug("Request submitted")

        try:
            result = self._process_events_sync(config, request_id, handler, on_progress)

            logger.debug("Request complete", {"response": result}, redact=True)

            # Parse response
            fal_result = cast(AnyDict, result)
            response = self._convert_response(config, request, request_id, fal_result)

            logger.info("Final generated response", {"response": response}, redact=True)

            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)

    # ==================== Image Generation ====================

    def _convert_image_request(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
    ) -> dict[str, object]:
        """Convert ImageGenerationRequest to Fal API format."""
        field_mappers = get_image_field_mappers(config.model)
        return apply_field_mappers(field_mappers, request)

    def _convert_image_response(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        request_id: str,
        fal_result: AnyDict,
    ) -> ImageGenerationResponse:
        """Convert Fal response to ImageGenerationResponse."""
        # Extract image URLs from Fal response
        # Fal typically returns {"images": [{"url": "..."}, ...]}
        images_data = fal_result.get("images", [])
        image_urls: list[str] = []
        for img in images_data:
            if isinstance(img, dict):
                url = img.get("url")
                if url:
                    image_urls.append(str(url))
            elif isinstance(img, str):
                image_urls.append(img)

        # Check for revised prompt (some models return this)
        revised_prompt = fal_result.get("revised_prompt")

        return ImageGenerationResponse(
            request_id=request_id,
            images=image_urls,
            content_type="image/png",
            status="completed",
            is_mock=False,
            revised_prompt=str(revised_prompt) if revised_prompt else None,
            raw_response=fal_result,
            provider_metadata={
                "model": config.model,
                "provider": config.provider,
            },
        )

    def _handle_image_error(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Convert exceptions to TarashException for image generation."""
        # Reuse video error handling logic
        return self._handle_error(
            VideoGenerationConfig(
                model=config.model,
                provider=config.provider,
                api_key=config.api_key,
                base_url=config.base_url,
            ),
            VideoGenerationRequest(prompt=request.prompt),
            request_id,
            ex,
        )

    @handle_video_generation_errors
    async def generate_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate image asynchronously with Fal.

        Args:
            config: Image generation configuration
            request: Image generation request
            on_progress: Optional callback for progress updates

        Returns:
            ImageGenerationResponse with generated image URLs
        """
        client = self._get_client(
            VideoGenerationConfig(
                model=config.model,
                provider=config.provider,
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_poll_attempts=config.max_poll_attempts,
                poll_interval=config.poll_interval,
            ),
            "async",
        )
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)

        # Convert request to Fal format
        fal_kwargs = self._convert_image_request(config, request)

        logger.info(
            "Mapped request to provider format", {"converted_request": fal_kwargs}
        )

        request_id = "unknown"
        try:
            # Submit request to queue
            handle: AsyncRequestHandle[AnyDict] = await client.submit(
                config.model,
                arguments=fal_kwargs,
            )

            request_id = handle.request_id
            logger = logger.with_request_id(request_id)

            logger.debug("Image request submitted")

            # Poll for completion
            poll_attempts = 0
            while poll_attempts < config.max_poll_attempts:
                status = await handle.status()

                # Report progress if callback provided
                if on_progress:
                    update = parse_fal_image_status(request_id, status)
                    if asyncio.iscoroutinefunction(on_progress):
                        await on_progress(update)
                    else:
                        on_progress(update)

                logger.info(
                    "Progress status update",
                    {"status": type(status).__name__},
                )

                # Check if complete
                if isinstance(status, Completed):
                    break

                # Wait before next poll
                await asyncio.sleep(config.poll_interval)
                poll_attempts += 1

            # Get final result
            result = await handle.get()

            logger.debug("Image request complete", {"response": result}, redact=True)

            # Parse response
            fal_result = cast(AnyDict, result)
            response = self._convert_image_response(
                config, request, request_id, fal_result
            )

            logger.info("Final generated response", {"response": response}, redact=True)

            return response

        except Exception as ex:
            raise self._handle_image_error(config, request, request_id, ex)

    @handle_video_generation_errors
    def generate_image(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: SyncImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate image synchronously with Fal.

        Args:
            config: Image generation configuration
            request: Image generation request
            on_progress: Optional callback for progress updates

        Returns:
            ImageGenerationResponse with generated image URLs
        """
        client = self._get_client(
            VideoGenerationConfig(
                model=config.model,
                provider=config.provider,
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_poll_attempts=config.max_poll_attempts,
                poll_interval=config.poll_interval,
            ),
            "sync",
        )
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)

        # Convert request to Fal format
        fal_kwargs = self._convert_image_request(config, request)

        logger.info(
            "Mapped request to provider format", {"converted_request": fal_kwargs}
        )

        request_id = "unknown"
        try:
            # Submit request to queue
            handle: SyncRequestHandle[AnyDict] = client.submit(
                config.model,
                arguments=fal_kwargs,
            )

            request_id = handle.request_id
            logger = logger.with_request_id(request_id)

            logger.debug("Image request submitted")

            # Poll for completion
            poll_attempts = 0
            while poll_attempts < config.max_poll_attempts:
                status = handle.status()

                # Report progress if callback provided
                if on_progress:
                    update = parse_fal_image_status(request_id, status)
                    on_progress(update)

                logger.info(
                    "Progress status update",
                    {"status": type(status).__name__},
                )

                # Check if complete
                if isinstance(status, Completed):
                    break

                # Wait before next poll
                time.sleep(config.poll_interval)
                poll_attempts += 1

            # Get final result
            result = handle.get()

            logger.debug("Image request complete", {"response": result}, redact=True)

            # Parse response
            fal_result = cast(AnyDict, result)
            response = self._convert_image_response(
                config, request, request_id, fal_result
            )

            logger.info("Final generated response", {"response": response}, redact=True)

            return response

        except Exception as ex:
            raise self._handle_image_error(config, request, request_id, ex)
