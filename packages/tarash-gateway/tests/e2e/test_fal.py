"""End-to-end tests for Fal provider.

These tests make actual API calls to the Fal.ai service.
Requires FAL_KEY environment variable to be set.

Run with: pytest tests/e2e/test_fal.py -v -m e2e
Skip with: pytest tests/e2e/test_fal.py -v -m "not e2e"
"""

import os

import pytest

from tarash.tarash_gateway import api
from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)

# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def fal_api_key():
    """Get Fal API key from environment."""
    api_key = os.getenv("FAL_KEY")
    if not api_key:
        pytest.skip("FAL_KEY environment variable not set")
    return api_key


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_comprehensive_async_video_generation(fal_api_key):
    """
    Comprehensive async test combining:
    - Basic video generation
    - Progress tracking
    - Custom parameters (seed, negative_prompt)
    - Different image types
    """
    # Track progress updates
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    # Test with various parameters
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    req = VideoGenerationRequest(
        prompt="A serene lake at sunset with mountains in the background, cinematic quality",
        duration_seconds=4,
        aspect_ratio="16:9",
        resolution="720p",
        seed=42,
        negative_prompt="blur, low quality",
        generate_audio=True,
        auto_fix=True,
    )

    # Generate video using API
    response = await api.generate_video_async(
        config, req, on_progress=progress_callback
    )

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Validate raw_response structure
    assert isinstance(response.raw_response, dict)
    assert "video" in response.raw_response, "raw_response should contain 'video' field"

    # Video should be a URL string (Fal returns URLs)
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )
    video_type = "URL"
    video_info = response.video
    # Validate progress tracking
    assert len(progress_updates) > 0, "Should receive at least one progress update"
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses, "Should receive completed status"

    # Log details
    print("✓ Generated video successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video type: {video_type}")
    print(f"  Video info: {video_info}")
    print(f"  Progress updates: {len(progress_updates)}")
    print(f"  Statuses: {statuses}")
    print(
        f"  Duration: {response.duration}s" if response.duration else "  Duration: N/A"
    )
    print(
        f"  Resolution: {response.resolution}"
        if response.resolution
        else "  Resolution: N/A"
    )


@pytest.mark.e2e
def test_sync_veo31_image_to_video(fal_api_key):
    """
    Sync test for veo3.1 image-to-video:
    - Sync generation path
    - Field mapping: image_list with type="reference" -> image_url
    - Vertical aspect ratio (9:16)
    - Progress callback in sync mode
    """
    progress_updates = []

    def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A calm ocean wave rolling onto a sandy beach with seagulls flying",
        duration_seconds=6,
        aspect_ratio="9:16",
        resolution="720p",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/example_inputs/veo31_i2v_input.jpg",
                "type": "reference",
            }
        ],
        generate_audio=False,
        auto_fix=True,
    )

    response = api.generate_video(config, request, on_progress=progress_callback)

    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"
    assert isinstance(response.video, str)
    assert response.video.startswith("http")

    assert len(progress_updates) > 0
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses

    print(f"✓ Generated veo3.1 image-to-video (sync): {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_minimax_image_to_video(fal_api_key):
    """
    Test Minimax image-to-video model.

    This tests:
    - Minimax-specific model (fal-ai/minimax/hailuo-02-fast/image-to-video)
    - Field mapping: image_list -> image_url
    - Prefix matching in registry
    - Only prompt and image_url fields
    """
    minimax_config = VideoGenerationConfig(
        model="fal-ai/minimax/hailuo-02-fast/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A person walking through a bustling city street",
        image_list=[
            {
                "image": "https://fal.media/files/elephant/8kkhB12hEZI2kkbU8pZPA_test.jpeg",
                "type": "reference",
            }
        ],
    )

    # Generate video using API (async)
    response = await api.generate_video_async(minimax_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )

    print(f"✓ Generated Minimax video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {minimax_config.model}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_kling_motion_control(fal_api_key):
    """
    Test Kling motion control model with image and video inputs.

    This tests:
    - Kling motion control model (fal-ai/kling-video/v2.6/standard/motion-control)
    - Image-to-video with motion guidance from reference video
    - Character orientation set to "video"
    - Keep original sound enabled
    - Field mapping: image_list -> image_url, video -> video_url
    """
    kling_config = VideoGenerationConfig(
        model="fal-ai/kling-video/v2.6/standard/motion-control",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="An african american woman dancing",
        image_list=[
            {
                "image": "https://v3b.fal.media/files/b/0a875302/8NaxQrQxDNHppHtqcchMm.png",
                "type": "reference",
            }
        ],
        video="https://v3b.fal.media/files/b/0a8752bc/2xrNS217ngQ3wzXqA7LXr_output.mp4",
        character_orientation="video",
        keep_original_sound=True,
    )

    # Generate video using API (async)
    response = await api.generate_video_async(kling_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )

    print(f"✓ Generated Kling motion-controlled video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {kling_config.model}")
    print("  Character orientation: video")
    print("  Keep original sound: True")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_veo31_first_last_frame_to_video(fal_api_key):
    """
    Test veo3.1 first-last-frame-to-video.

    This tests:
    - Veo 3.1 first-last-frame model (fal-ai/veo3.1/fast/first-last-frame-to-video)
    - Field mapping: image_list with type="first_frame" and "last_frame"
    - Validates both frames are required and correctly mapped
    - All veo3.1 parameters
    """
    veo31_config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/first-last-frame-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt='A woman looks into the camera, breathes in, then exclaims energetically, "have you guys checked out Veo3.1 First-Last-Frame-to-Video on Fal? It\'s incredible!"',
        duration_seconds=8,
        aspect_ratio="16:9",
        resolution="720p",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/example_inputs/veo31-flf2v-input-1.jpeg",
                "type": "first_frame",
            },
            {
                "image": "https://storage.googleapis.com/falserverless/example_inputs/veo31-flf2v-input-2.jpeg",
                "type": "last_frame",
            },
        ],
        generate_audio=True,
        auto_fix=False,
    )

    # Generate video using API (async)
    response = await api.generate_video_async(veo31_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )

    print(f"✓ Generated veo3.1 first-last-frame-to-video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {veo31_config.model}")
    print("  Duration: 8s")
    print("  Aspect Ratio: 16:9")
    print("  Resolution: 720p")
    print("  Generate Audio: True")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_wan_v26_text_to_video(fal_api_key):
    """
    Test Wan v2.6 text-to-video model.

    This tests:
    - Wan v2.6 text-to-video model (wan/v2.6/text-to-video)
    - Field mapping: duration_seconds -> duration (string without 's' suffix)
    - Wan-specific parameters: enable_prompt_expansion, multi_shots
    - Extra params: audio_url
    - Prefix matching in registry (wan/v2.6/)
    """
    wan_config = VideoGenerationConfig(
        model="wan/v2.6/text-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A cinematic shot of a fox director on a movie set, giving instructions to the crew",
        duration_seconds=5,
        aspect_ratio="16:9",
        resolution="1080p",
        seed=42,
        enhance_prompt=True,  # Maps to enable_prompt_expansion
        negative_prompt="low quality, blurry, distorted",
        extra_params={
            "multi_shots": True,
        },
    )

    # Generate video using API (async)
    response = await api.generate_video_async(wan_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )

    print(f"✓ Generated Wan v2.6 text-to-video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {wan_config.model}")
    print("  Duration: 5s")
    print("  Aspect Ratio: 16:9")
    print("  Resolution: 1080p")
    print("  Prompt expansion: Enabled")
    print("  Multi-shots: Enabled")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_wan_v25_image_to_video(fal_api_key):
    """
    Test Wan v2.5 image-to-video model.

    This tests:
    - Wan v2.5 image-to-video model (fal-ai/wan-25-preview/image-to-video)
    - Field mapping: image_list -> image_url
    - Unified mapper shared between v2.6 and v2.5
    - Duration as string format
    - Prefix matching in registry (fal-ai/wan-25-preview/)
    """
    wan_config = VideoGenerationConfig(
        model="fal-ai/wan-25-preview/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A white dragon warrior standing majestically, with camera slowly moving around",
        duration_seconds=5,
        aspect_ratio="16:9",
        resolution="720p",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/model_tests/wan/dragon-warrior.jpg",
                "type": "reference",
            }
        ],
        enhance_prompt=True,
        negative_prompt="low quality, blurry",
    )

    # Generate video using API (async)
    response = await api.generate_video_async(wan_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )

    print(f"✓ Generated Wan v2.5 image-to-video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {wan_config.model}")
    print("  Duration: 5s")
    print("  Aspect Ratio: 16:9")
    print("  Resolution: 720p")
    print("  With reference image")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_wan_v22_animate_move(fal_api_key):
    """
    Test Wan v2.2-14b animate/move model.

    This tests:
    - Wan v2.2-14b animate/move model (fal-ai/wan/v2.2-14b/animate/move)
    - Field mapping: video -> video_url, image_list -> image_url
    - Wan v2.2-specific parameters: shift, guidance_scale, num_inference_steps
    - Quality parameters: video_quality, video_write_mode, use_turbo
    - Separate mapper from v2.6/v2.5
    - Prefix matching in registry (fal-ai/wan/v2.2-14b/animate/)
    """
    wan_config = VideoGenerationConfig(
        model="fal-ai/wan/v2.2-14b/animate/move",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="",  # Auto-generated by model
        video="https://v3b.fal.media/files/b/panda/a6SvJg96V8eoglMlYFShU_5385885-hd_1080_1920_25fps.mp4",
        image_list=[
            {
                "image": "https://v3b.fal.media/files/b/panda/-oMlZo9Yyj_Nzoza_tgds_GmLF86r5bOt50eMMKCszy_eacc949b3933443c9915a83c98fbe85e.png",
                "type": "reference",
            }
        ],
        resolution="480p",
        seed=42,
        extra_params={
            "shift": 8,
            "guidance_scale": 1,
            "num_inference_steps": 6,
            "use_turbo": True,
            "video_quality": "high",
            "video_write_mode": "balanced",
        },
    )

    # Generate video using API (async)
    response = await api.generate_video_async(wan_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )

    print(f"✓ Generated Wan v2.2-14b animate/move: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {wan_config.model}")
    print("  Resolution: 480p")
    print("  Shift: 8")
    print("  Guidance scale: 1")
    print("  Inference steps: 6")
    print("  Use turbo: True")
    print("  Video quality: high")
    print("  Video write mode: balanced")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_wan_v22_a14b_text_to_video_lora(fal_api_key):
    """
    Test Wan v2.2-a14b text-to-video with LoRA support.

    This tests:
    - Wan v2.2-a14b text-to-video/lora (fal-ai/wan/v2.2-a14b/text-to-video/lora)
    - Progress tracking via async callback
    - LoRA weight support
    - Generation params: guidance_scale, shift, num_inference_steps, acceleration
    - Quality params: video_quality, video_write_mode
    - Prefix matching in registry (fal-ai/wan/v2.2-a14b/)
    """
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    wan_config = VideoGenerationConfig(
        model="fal-ai/wan/v2.2-a14b/text-to-video/lora",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A cinematic scene of a magical forest with glowing fireflies at night",
        aspect_ratio="16:9",
        resolution="480p",
        seed=42,
        negative_prompt="low quality, blurry, distorted",
        enhance_prompt=True,
        extra_params={
            "num_frames": 81,
            "guidance_scale": 3.5,
            "shift": 5.0,
            "num_inference_steps": 27,
            "acceleration": "regular",
            "video_quality": "high",
            "video_write_mode": "balanced",
        },
    )

    response = await api.generate_video_async(
        wan_config, request, on_progress=progress_callback
    )

    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"
    assert isinstance(response.video, str)
    assert response.video.startswith("http")

    assert len(progress_updates) > 0
    statuses = [u.status for u in progress_updates]
    assert "completed" in statuses

    print(f"✓ Generated Wan v2.2-a14b text-to-video/lora: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_wan_v22_a14b_image_to_video_lora(fal_api_key):
    """
    Test Wan v2.2-a14b image-to-video with LoRA support.

    This tests:
    - Wan v2.2-a14b image-to-video/lora (fal-ai/wan/v2.2-a14b/image-to-video/lora)
    - Field mapping: image_list -> image_url
    - enable_prompt_expansion mapping from enhance_prompt
    - Interpolation params: interpolator_model, num_interpolated_frames
    - Safety checkers
    """
    wan_config = VideoGenerationConfig(
        model="fal-ai/wan/v2.2-a14b/image-to-video/lora",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="Camera slowly zooming in on the subject with gentle motion",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg",
                "type": "reference",
            }
        ],
        resolution="480p",
        seed=42,
        enhance_prompt=True,
        extra_params={
            "num_frames": 81,
            "guidance_scale": 3.5,
            "num_inference_steps": 27,
            "interpolator_model": "film",
            "num_interpolated_frames": 1,
            "acceleration": "regular",
            "video_quality": "high",
        },
    )

    response = await api.generate_video_async(wan_config, request)

    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"
    assert isinstance(response.video, str)
    assert response.video.startswith("http")

    print(f"✓ Generated Wan v2.2-a14b image-to-video/lora: {response.request_id}")
    print(f"  Video URL: {response.video}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_veo31_fast_extend_video(fal_api_key):
    """
    Test Veo 3.1 fast extend-video model.

    This tests:
    - Veo 3.1 fast extend-video model (fal-ai/veo3.1/fast/extend-video)
    - Field mapping: video -> video_url (required)
    - 7s duration (only supported duration for extend-video)
    - 720p resolution (only supported resolution for extend-video)
    - Aspect ratio options: auto, 16:9, 9:16
    - Generate audio and auto_fix parameters
    """
    veo31_extend_config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/extend-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    # Track progress updates
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    request = VideoGenerationRequest(
        prompt="A person walking through a park on a sunny day, gentle breeze moving the leaves",
        video="https://v3b.fal.media/files/b/0a8670fe/pY8UGl4_C452wOm9XUBYO_9ae04df8771c4f3f979fa5cabeca6ada.mp4",
        aspect_ratio="16:9",
        duration_seconds=7,
        resolution="720p",
        generate_audio=True,
        auto_fix=False,
    )

    # Generate video using API (async)
    response = await api.generate_video_async(
        veo31_extend_config, request, on_progress=progress_callback
    )

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )

    # Validate progress tracking
    assert len(progress_updates) > 0, "Should receive at least one progress update"
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses, "Should receive completed status"

    print(f"✓ Generated veo3.1 fast extend-video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {veo31_extend_config.model}")
    print("  Duration: 7s")
    print("  Aspect Ratio: 16:9")
    print("  Resolution: 720p")
    print("  Generate Audio: True")
    print(f"  Progress updates: {len(progress_updates)}")
    print(f"  Statuses: {statuses}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sora2_text_to_video(fal_api_key):
    """
    Test Sora 2 text-to-video model.

    This tests:
    - Sora 2 text-to-video model (fal-ai/sora-2/text-to-video)
    - Field mapping: duration_seconds -> duration (int)
    - Sora 2 parameters: aspect_ratio, resolution
    - Duration values: 4, 8, or 12 seconds
    """
    config = VideoGenerationConfig(
        model="fal-ai/sora-2/text-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A serene waterfall in a peaceful forest with morning sunlight filtering through the trees",
        duration_seconds=8,
        aspect_ratio="16:9",
        resolution="720p",
    )

    response = await api.generate_video_async(config, request)

    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"
    assert isinstance(response.video, str)
    assert response.video.startswith("http")

    print(f"✓ Generated Sora 2 text-to-video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {config.model}")
    print("  Duration: 8s")
    print("  Aspect Ratio: 16:9")
    print("  Resolution: 720p")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sora2_image_to_video(fal_api_key):
    """
    Test Sora 2 image-to-video model.

    This tests:
    - Sora 2 image-to-video model (fal-ai/sora-2/image-to-video)
    - Field mapping: image_list with type="reference" -> image_url
    - Different duration (4s) and aspect ratio (1:1)
    """
    config = VideoGenerationConfig(
        model="fal-ai/sora-2/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="The scene comes to life with gentle motion and ambient sounds",
        duration_seconds=4,
        aspect_ratio="1:1",
        resolution="720p",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/example_inputs/veo31_i2v_input.jpg",
                "type": "reference",
            }
        ],
    )

    response = await api.generate_video_async(config, request)

    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"
    assert isinstance(response.video, str)
    assert response.video.startswith("http")

    print(f"✓ Generated Sora 2 image-to-video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {config.model}")
    print("  Duration: 4s")
    print("  Aspect Ratio: 1:1")
