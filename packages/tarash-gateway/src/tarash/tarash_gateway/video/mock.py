"""Mock video generation - single module with all mock logic and models."""

import asyncio
import base64
import hashlib
import random
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator

from tarash.tarash_gateway.video.exceptions import TarashException
from tarash.tarash_gateway.video.models import (
    AspectRatio,
    MediaContent,
    MediaType,
    ProgressCallback,
    ProviderHandler,
    Resolution,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.video.utils import (
    download_media_from_url,
    download_media_from_url_async,
)


# ==================== Mock Models ====================


@dataclass(frozen=True)
class MockVideoSpec:
    """Specification for a mock video in the library."""

    aspect_ratio: AspectRatio
    resolution: Resolution
    duration: float  # seconds
    url: str


# Mock video library - hardcoded sample videos
MOCK_VIDEO_LIBRARY: list[MockVideoSpec] = [
    # 9:16 Portrait
    MockVideoSpec(
        "9:16",
        "720p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/portrait-720p-4s.mp4",
    ),
    MockVideoSpec(
        "9:16",
        "1080p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/portrait-1080p-4s.mp4",
    ),
    MockVideoSpec(
        "9:16",
        "720p",
        8.0,
        "https://storage.googleapis.com/tarash-mock-videos/portrait-720p-8s.mp4",
    ),
    # 16:9 Landscape
    MockVideoSpec(
        "16:9",
        "720p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/landscape-720p-4s.mp4",
    ),
    MockVideoSpec(
        "16:9",
        "1080p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/landscape-1080p-4s.mp4",
    ),
    MockVideoSpec(
        "16:9",
        "4k",
        8.0,
        "https://storage.googleapis.com/tarash-mock-videos/landscape-4k-8s.mp4",
    ),
    # 1:1 Square
    MockVideoSpec(
        "1:1",
        "720p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/square-720p-4s.mp4",
    ),
    MockVideoSpec(
        "1:1",
        "1080p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/square-1080p-4s.mp4",
    ),
    # 4:3 Classic
    MockVideoSpec(
        "4:3",
        "480p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/classic-480p-4s.mp4",
    ),
    # 21:9 Ultrawide
    MockVideoSpec(
        "21:9",
        "1080p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/ultrawide-1080p-4s.mp4",
    ),
]


class MockPollingConfig(BaseModel):
    """Configuration for simulating polling updates."""

    enabled: bool = True
    status_sequence: list[Literal["queued", "processing", "completed", "failed"]] = (
        Field(  # type: ignore[assignment]
            default_factory=lambda: ["queued", "processing", "completed"]
        )
    )
    delay_between_updates: float = 0.5
    progress_percentages: list[int] | None = None
    custom_updates: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_polling_config(self) -> "MockPollingConfig":
        if self.progress_percentages and len(self.progress_percentages) != len(
            self.status_sequence
        ):
            raise ValueError(
                "progress_percentages length must match status_sequence length"
            )
        if self.custom_updates and len(self.custom_updates) != len(
            self.status_sequence
        ):
            raise ValueError("custom_updates length must match status_sequence length")
        return self

    model_config = {"frozen": True}


class MockResponse(BaseModel):
    """A single mock response (success or error) with weight."""

    weight: float = 1.0
    mock_response: VideoGenerationResponse | None = None
    output_video: MediaType | None = None
    output_video_type: Literal["url", "content"] = "url"
    error: Exception | None = None

    @model_validator(mode="after")
    def validate_response(self) -> "MockResponse":
        has_success = self.mock_response is not None or self.output_video is not None
        has_error = self.error is not None

        if has_success and has_error:
            raise ValueError(
                "Cannot specify both success and error in the same MockResponse"
            )
        if self.mock_response and self.output_video:
            raise ValueError("Cannot specify both mock_response and output_video")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        return self

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class MockConfig(BaseModel):
    """Configuration for mocking video generation responses."""

    enabled: bool
    responses: list[MockResponse] | None = None
    polling: MockPollingConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, data: dict) -> dict:
        if isinstance(data, dict):
            # Default to single success response if enabled and no responses provided
            if data.get("enabled") and not data.get("responses"):
                data["responses"] = [MockResponse(weight=1.0)]

            # Validate total weight if responses provided
            if data.get("responses"):
                total_weight = sum(r.weight for r in data["responses"])
                if total_weight <= 0:
                    raise ValueError("Total weight must be positive")

        return data

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


# Rebuild VideoGenerationConfig now that MockConfig is defined
# This resolves the forward reference in models.py
VideoGenerationConfig.model_rebuild()

# Cache directory
MOCK_CACHE_DIR = Path.home() / ".tarash" / "mock_cache"
MOCK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Request ID ====================


def generate_mock_request_id() -> str:
    """Generate unique request ID with mock prefix."""
    return f"mock_{uuid.uuid4()}"


# ==================== Video Matching ====================


def _calculate_match_score(
    request_aspect: AspectRatio,
    request_resolution: Resolution,
    request_duration: float,
    spec: MockVideoSpec,
) -> float:
    """Calculate match score (0-1, higher is better)."""
    score = 0.0

    if request_aspect == spec.aspect_ratio:
        score += 0.4

    resolution_order = ["360p", "480p", "720p", "1080p", "4k"]
    req_idx = resolution_order.index(request_resolution)
    spec_idx = resolution_order.index(spec.resolution)
    score += max(0, 0.3 - abs(req_idx - spec_idx) * 0.15)

    duration_diff = abs(request_duration - spec.duration)
    score += max(0, 0.3 - duration_diff * 0.05)

    return score


def find_best_matching_video(request: VideoGenerationRequest) -> MockVideoSpec:
    """Find best matching video from MOCK_VIDEO_LIBRARY."""
    aspect_ratio = request.aspect_ratio or "9:16"
    duration = float(request.duration_seconds or 4)
    resolution = request.resolution or "720p"

    best_spec, best_score = None, -1.0
    for spec in MOCK_VIDEO_LIBRARY:
        score = _calculate_match_score(aspect_ratio, resolution, duration, spec)
        if score > best_score:
            best_score, best_spec = score, spec

    return best_spec or MOCK_VIDEO_LIBRARY[0]


# ==================== Download & Caching ====================


@lru_cache(maxsize=128)
def _download_cached_sync(url: str) -> bytes:
    """Download with disk + LRU cache (sync).

    Raises:
        HTTPError: If download fails with HTTP error
        TarashException: If download fails for other reasons
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    cache_file = MOCK_CACHE_DIR / f"{url_hash}.mp4"

    if cache_file.exists():
        return cache_file.read_bytes()

    # Let exceptions from download_media_from_url propagate
    content, _ = download_media_from_url(url, provider="mock")
    cache_file.write_bytes(content)
    return content


_async_cache: dict[str, bytes] = {}


async def _download_cached_async(url: str) -> bytes:
    """Download with disk + memory cache (async).

    Raises:
        HTTPError: If download fails with HTTP error
        TarashException: If download fails for other reasons
    """
    if url in _async_cache:
        return _async_cache[url]

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    cache_file = MOCK_CACHE_DIR / f"{url_hash}.mp4"

    if cache_file.exists():
        content = cache_file.read_bytes()
        _async_cache[url] = content
        return content

    # Let exceptions from download_media_from_url_async propagate
    content, _ = await download_media_from_url_async(url, provider="mock")
    cache_file.write_bytes(content)
    _async_cache[url] = content

    # Limit memory cache
    if len(_async_cache) > 128:
        _async_cache.pop(next(iter(_async_cache)))

    return content


# ==================== MediaType Conversion ====================


def _convert_to_content_sync(video: MediaType) -> MediaContent:
    """Convert MediaType to MediaContent (sync)."""
    if isinstance(video, dict) and "content" in video:
        return video

    if isinstance(video, (HttpUrl, str)) and str(video).startswith(
        ("http://", "https://")
    ):
        content = _download_cached_sync(str(video))
        return {"content": content, "content_type": "video/mp4"}

    if isinstance(video, str):
        # Base64
        if video.startswith("data:"):
            header, b64_data = video.split(",", 1)
            content_type = header.split(";")[0].replace("data:", "")
            content = base64.b64decode(b64_data)
        else:
            content = base64.b64decode(video)
            content_type = "video/mp4"
        return {"content": content, "content_type": content_type}

    raise TarashException(f"Unsupported MediaType: {type(video)}", provider="mock")


async def _convert_to_content_async(video: MediaType) -> MediaContent:
    """Convert MediaType to MediaContent (async)."""
    if isinstance(video, dict) and "content" in video:
        return video

    if isinstance(video, (HttpUrl, str)) and str(video).startswith(
        ("http://", "https://")
    ):
        content = await _download_cached_async(str(video))
        return {"content": content, "content_type": "video/mp4"}

    if isinstance(video, str):
        # Base64 (sync operation)
        if video.startswith("data:"):
            header, b64_data = video.split(",", 1)
            content_type = header.split(";")[0].replace("data:", "")
            content = base64.b64decode(b64_data)
        else:
            content = base64.b64decode(video)
            content_type = "video/mp4"
        return {"content": content, "content_type": content_type}

    raise TarashException(f"Unsupported MediaType: {type(video)}", provider="mock")


# ==================== Response Selection ====================


def select_mock_response(responses: list[MockResponse]) -> MockResponse:
    """Select response based on weights."""
    total_weight = sum(r.weight for r in responses)
    weights = [r.weight / total_weight for r in responses]
    return random.choices(responses, weights=weights, k=1)[0]


# ==================== Success Response Creation ====================


def _create_success_sync(
    request: VideoGenerationRequest,
    request_id: str,
    config: MockResponse,
) -> VideoGenerationResponse:
    """Create success response (sync)."""
    if config.mock_response:
        return config.mock_response.model_copy(
            update={"request_id": request_id, "is_mock": True}
        )

    if config.output_video:
        video = (
            _convert_to_content_sync(config.output_video)
            if config.output_video_type == "content"
            else config.output_video
        )
        return VideoGenerationResponse(
            request_id=request_id,
            video=video,
            content_type="video/mp4",
            duration=None,
            resolution=None,
            aspect_ratio=None,
            status="completed",
            is_mock=True,
            raw_response={"mock": True, "output_video_provided": True},
            provider_metadata={"mock_mode": True},
        )

    # Auto-match
    spec = find_best_matching_video(request)
    video = (
        _convert_to_content_sync(spec.url)
        if config.output_video_type == "content"
        else spec.url
    )

    return VideoGenerationResponse(
        request_id=request_id,
        video=video,
        content_type="video/mp4",
        duration=spec.duration,
        resolution=spec.resolution,
        aspect_ratio=spec.aspect_ratio,
        status="completed",
        is_mock=True,
        raw_response={
            "mock": True,
            "matched_spec": {
                "aspect_ratio": spec.aspect_ratio,
                "resolution": spec.resolution,
                "duration": spec.duration,
                "url": spec.url,
            },
        },
        provider_metadata={"mock_mode": True},
    )


async def _create_success_async(
    request: VideoGenerationRequest,
    request_id: str,
    config: MockResponse,
) -> VideoGenerationResponse:
    """Create success response (async)."""
    if config.mock_response:
        return config.mock_response.model_copy(
            update={"request_id": request_id, "is_mock": True}
        )

    if config.output_video:
        video = (
            await _convert_to_content_async(config.output_video)
            if config.output_video_type == "content"
            else config.output_video
        )
        return VideoGenerationResponse(
            request_id=request_id,
            video=video,
            content_type="video/mp4",
            duration=None,
            resolution=None,
            aspect_ratio=None,
            status="completed",
            is_mock=True,
            raw_response={"mock": True, "output_video_provided": True},
            provider_metadata={"mock_mode": True},
        )

    # Auto-match
    spec = find_best_matching_video(request)
    video = (
        await _convert_to_content_async(spec.url)
        if config.output_video_type == "content"
        else spec.url
    )

    return VideoGenerationResponse(
        request_id=request_id,
        video=video,
        content_type="video/mp4",
        duration=spec.duration,
        resolution=spec.resolution,
        aspect_ratio=spec.aspect_ratio,
        status="completed",
        is_mock=True,
        raw_response={
            "mock": True,
            "matched_spec": {
                "aspect_ratio": spec.aspect_ratio,
                "resolution": spec.resolution,
                "duration": spec.duration,
                "url": spec.url,
            },
        },
        provider_metadata={"mock_mode": True},
    )


# ==================== Polling Simulation ====================


def _polling_update(
    request_id: str,
    status: str,
    progress: int | None,
    custom: dict[str, Any] | None,
    result: VideoGenerationResponse | None = None,
    error: str | None = None,
) -> VideoGenerationUpdate:
    """Create polling update."""
    return VideoGenerationUpdate(
        request_id=request_id,
        status=status,  # type: ignore
        progress_percent=progress,
        update=custom or {},
        result=result,
        error=error,
    )


def _simulate_polling_sync(
    request_id: str,
    config: MockPollingConfig,
    result: VideoGenerationResponse | None = None,
    error: str | None = None,
    callback: ProgressCallback | None = None,
) -> None:
    """Simulate polling (sync)."""
    if not callback:
        return

    for idx, status in enumerate(config.status_sequence):
        progress = (
            config.progress_percentages[idx]
            if config.progress_percentages
            else {"queued": 0, "processing": 50, "completed": 100, "failed": None}.get(
                status
            )
        )

        custom = config.custom_updates[idx] if config.custom_updates else {}
        update = _polling_update(
            request_id,
            status,
            progress,
            custom,
            result if status == "completed" else None,
            error if status == "failed" else None,
        )

        callback(update)

        if idx < len(config.status_sequence) - 1:
            time.sleep(config.delay_between_updates)


async def _simulate_polling_async(
    request_id: str,
    config: MockPollingConfig,
    result: VideoGenerationResponse | None = None,
    error: str | None = None,
    callback: ProgressCallback | None = None,
) -> None:
    """Simulate polling (async)."""
    if not callback:
        return

    for idx, status in enumerate(config.status_sequence):
        progress = (
            config.progress_percentages[idx]
            if config.progress_percentages
            else {"queued": 0, "processing": 50, "completed": 100, "failed": None}.get(
                status
            )
        )

        custom = config.custom_updates[idx] if config.custom_updates else {}
        update = _polling_update(
            request_id,
            status,
            progress,
            custom,
            result if status == "completed" else None,
            error if status == "failed" else None,
        )

        if asyncio.iscoroutinefunction(callback):
            await callback(update)
        else:
            callback(update)

        if idx < len(config.status_sequence) - 1:
            await asyncio.sleep(config.delay_between_updates)


# ==================== Main Handlers ====================


def handle_mock_request_sync(
    mock_config: MockConfig,
    request: VideoGenerationRequest,
    on_progress: ProgressCallback | None = None,
) -> VideoGenerationResponse:
    """Handle mock request (sync)."""
    request_id = generate_mock_request_id()
    selected = select_mock_response(mock_config.responses)  # type: ignore[arg-type]

    if selected.error:
        if mock_config.polling:
            # Ensure ends with "failed"
            polling = mock_config.polling
            if polling.status_sequence[-1] != "failed":
                polling = polling.model_copy(
                    update={
                        "status_sequence": [*polling.status_sequence[:-1], "failed"]
                    }
                )
            _simulate_polling_sync(
                request_id, polling, None, str(selected.error), on_progress
            )
        raise selected.error

    response = _create_success_sync(request, request_id, selected)

    if mock_config.polling:
        _simulate_polling_sync(
            request_id, mock_config.polling, response, None, on_progress
        )

    return response


async def handle_mock_request_async(
    mock_config: MockConfig,
    request: VideoGenerationRequest,
    on_progress: ProgressCallback | None = None,
) -> VideoGenerationResponse:
    """Handle mock request (async)."""
    request_id = generate_mock_request_id()
    selected = select_mock_response(mock_config.responses)  # type: ignore[arg-type]

    if selected.error:
        if mock_config.polling:
            # Ensure ends with "failed"
            polling = mock_config.polling
            if polling.status_sequence[-1] != "failed":
                polling = polling.model_copy(
                    update={
                        "status_sequence": [*polling.status_sequence[:-1], "failed"]
                    }
                )
            await _simulate_polling_async(
                request_id, polling, None, str(selected.error), on_progress
            )
        raise selected.error

    response = await _create_success_async(request, request_id, selected)

    if mock_config.polling:
        await _simulate_polling_async(
            request_id, mock_config.polling, response, None, on_progress
        )

    return response


# ==================== Provider Handler ====================


class MockProviderHandler(ProviderHandler):
    """Provider handler for mock video generation.

    This handler wraps the mock logic and presents it as a standard ProviderHandler,
    allowing mock to be treated like any other provider in the execution orchestrator.
    """

    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate mock video asynchronously.

        Args:
            config: Video generation configuration (with mock config)
            request: Video generation request
            on_progress: Optional progress callback

        Returns:
            Mock video generation response

        Raises:
            TarashException: If mock is configured to raise an error
        """
        if not config.mock or not config.mock.enabled:
            raise ValueError("MockProviderHandler requires mock config to be enabled")

        return await handle_mock_request_async(config.mock, request, on_progress)

    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate mock video synchronously.

        Args:
            config: Video generation configuration (with mock config)
            request: Video generation request
            on_progress: Optional progress callback

        Returns:
            Mock video generation response

        Raises:
            TarashException: If mock is configured to raise an error
        """
        if not config.mock or not config.mock.enabled:
            raise ValueError("MockProviderHandler requires mock config to be enabled")

        return handle_mock_request_sync(config.mock, request, on_progress)
