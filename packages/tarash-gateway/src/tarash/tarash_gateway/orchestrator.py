"""Orchestrator for managing generation execution with fallback support."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from tarash.tarash_gateway.logging import log_error, log_info
from tarash.tarash_gateway.exceptions import is_retryable_error
from tarash.tarash_gateway.models import (
    AttemptMetadata,
    AudioGenerationConfig,
    MultiModalGenerationConfig,
    MultiModalGenerationRequest,
    MultiModalGenerationResponse,
    MultiModalProgressCallback,
    ExecutionMetadata,
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageProgressCallback,
    ProgressCallback,
    STSProgressCallback,
    STSRequest,
    STSResponse,
    TTSProgressCallback,
    TTSRequest,
    TTSResponse,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.registry import get_handler

_LOGGER_NAME = "tarash.tarash_gateway.orchestrator"


def _compute_total_cost_usd(attempts: list[AttemptMetadata]) -> Decimal | None:
    """Compute total USD cost across all attempts.

    Returns ``None`` if any attempt lacks cost data or has
    ``amount_usd`` set to ``None``.
    """
    attempt_costs = [a.cost for a in attempts]
    if any(c is None for c in attempt_costs):
        return None
    if any(c.amount_usd is None for c in attempt_costs):  # type: ignore[union-attr]
        return None
    return sum((c.amount_usd for c in attempt_costs), Decimal("0"))  # type: ignore[union-attr, misc]


def _collect_fallback_chain(config: BaseModel) -> list[Any]:
    """Collect the full fallback chain using depth-first traversal.

    Works for any config type that has an optional ``fallback_configs`` list.
    """
    chain: list[Any] = [config]
    fallbacks = getattr(config, "fallback_configs", None)
    if fallbacks:
        for fallback in fallbacks:
            chain.extend(_collect_fallback_chain(fallback))
    return chain


class ExecutionOrchestrator:
    """Manages provider execution with automatic fallback and metadata tracking.

    Traverses the fallback chain depth-first, retrying with each successive
    provider on retryable errors. Attaches ``ExecutionMetadata`` to every
    response for observability.
    """

    # ------------------------------------------------------------------
    # Generic core executors
    # ------------------------------------------------------------------

    async def _execute_chain_async(
        self,
        fallback_chain: list[Any],
        request: Any,
        on_progress: Any,
        handler_method: str,
        error_label: str,
    ) -> Any:
        """Execute a generation request asynchronously through the fallback chain.

        Args:
            fallback_chain: Ordered list of configs to try.
            request: Generation parameters.
            on_progress: Optional progress callback forwarded to the handler.
            handler_method: Name of the async handler method to invoke.
            error_label: Label for error messages (e.g. "Video", "Image").
        """
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

        log_info(
            f"Starting {error_label.lower()} fallback chain execution",
            context={
                "configs_in_chain": len(fallback_chain),
                "primary_provider": fallback_chain[0].provider,
                "primary_model": fallback_chain[0].model,
            },
            logger_name=_LOGGER_NAME,
        )

        for attempt_number, cfg in enumerate(fallback_chain, start=1):
            started_at = datetime.now()
            attempt_metadata = AttemptMetadata(
                provider=cfg.provider,
                model=cfg.model,
                attempt_number=attempt_number,
                started_at=started_at,
                ended_at=None,
                status="failed",
                error_type=None,
                error_message=None,
                is_retryable=None,
                request_id=None,
            )

            try:
                log_info(
                    f"Attempting with provider (attempt {attempt_number}/{len(fallback_chain)})",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "attempt_number": attempt_number,
                    },
                    logger_name=_LOGGER_NAME,
                )

                handler = get_handler(cfg)
                method = getattr(handler, handler_method)
                response = await method(cfg, request, on_progress=on_progress)

                ended_at = datetime.now()
                attempt_metadata.ended_at = ended_at
                attempt_metadata.status = "success"
                attempt_metadata.request_id = response.request_id
                attempt_metadata.cost = response.cost
                attempts.append(attempt_metadata)

                execution_metadata = ExecutionMetadata(
                    total_attempts=len(attempts),
                    successful_attempt=attempt_number,
                    attempts=attempts,
                    fallback_triggered=attempt_number > 1,
                    configs_in_chain=len(fallback_chain),
                    total_cost_usd=_compute_total_cost_usd(attempts),
                )

                log_info(
                    f"Successfully generated on attempt {attempt_number}",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "request_id": response.request_id,
                        "total_attempts": len(attempts),
                    },
                    logger_name=_LOGGER_NAME,
                )

                return response.model_copy(
                    update={"execution_metadata": execution_metadata}
                )

            except NotImplementedError:
                raise

            except Exception as ex:
                ended_at = datetime.now()
                attempt_metadata.ended_at = ended_at
                attempt_metadata.error_type = type(ex).__name__
                attempt_metadata.error_message = str(ex)
                attempt_metadata.is_retryable = is_retryable_error(ex)
                attempts.append(attempt_metadata)
                last_exception = ex

                log_error(
                    f"Attempt {attempt_number} failed",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "error_type": type(ex).__name__,
                        "error_message": str(ex),
                        "is_retryable": attempt_metadata.is_retryable,
                    },
                    logger_name=_LOGGER_NAME,
                )

                if not attempt_metadata.is_retryable:
                    log_info(
                        "Non-retryable error encountered, stopping fallback chain",
                        context={"error_type": type(ex).__name__},
                        logger_name=_LOGGER_NAME,
                    )
                    raise ex

                if attempt_number == len(fallback_chain):
                    log_error(
                        "All fallback attempts exhausted",
                        context={"total_attempts": len(attempts)},
                        logger_name=_LOGGER_NAME,
                    )
                    raise ex

                log_info(
                    f"Retryable error, continuing to next fallback ({attempt_number + 1}/{len(fallback_chain)})",
                    logger_name=_LOGGER_NAME,
                )

        if last_exception:
            raise last_exception
        raise RuntimeError(f"{error_label} fallback chain execution failed unexpectedly")

    def _execute_chain_sync(
        self,
        fallback_chain: list[Any],
        request: Any,
        on_progress: Any,
        handler_method: str,
        error_label: str,
    ) -> Any:
        """Execute a generation request synchronously through the fallback chain.

        Blocking version of ``_execute_chain_async``.

        Args:
            fallback_chain: Ordered list of configs to try.
            request: Generation parameters.
            on_progress: Optional progress callback forwarded to the handler.
            handler_method: Name of the sync handler method to invoke.
            error_label: Label for error messages (e.g. "Video", "Image").
        """
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

        log_info(
            f"Starting {error_label.lower()} fallback chain execution (sync)",
            context={
                "configs_in_chain": len(fallback_chain),
                "primary_provider": fallback_chain[0].provider,
                "primary_model": fallback_chain[0].model,
            },
            logger_name=_LOGGER_NAME,
        )

        for attempt_number, cfg in enumerate(fallback_chain, start=1):
            started_at = datetime.now()
            attempt_metadata = AttemptMetadata(
                provider=cfg.provider,
                model=cfg.model,
                attempt_number=attempt_number,
                started_at=started_at,
                ended_at=None,
                status="failed",
                error_type=None,
                error_message=None,
                is_retryable=None,
                request_id=None,
            )

            try:
                log_info(
                    f"Attempting with provider (attempt {attempt_number}/{len(fallback_chain)})",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "attempt_number": attempt_number,
                    },
                    logger_name=_LOGGER_NAME,
                )

                handler = get_handler(cfg)
                method = getattr(handler, handler_method)
                response = method(cfg, request, on_progress=on_progress)

                ended_at = datetime.now()
                attempt_metadata.ended_at = ended_at
                attempt_metadata.status = "success"
                attempt_metadata.request_id = response.request_id
                attempt_metadata.cost = response.cost
                attempts.append(attempt_metadata)

                execution_metadata = ExecutionMetadata(
                    total_attempts=len(attempts),
                    successful_attempt=attempt_number,
                    attempts=attempts,
                    fallback_triggered=attempt_number > 1,
                    configs_in_chain=len(fallback_chain),
                    total_cost_usd=_compute_total_cost_usd(attempts),
                )

                log_info(
                    f"Successfully generated on attempt {attempt_number}",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "request_id": response.request_id,
                        "total_attempts": len(attempts),
                    },
                    logger_name=_LOGGER_NAME,
                )

                return response.model_copy(
                    update={"execution_metadata": execution_metadata}
                )

            except NotImplementedError:
                raise

            except Exception as ex:
                ended_at = datetime.now()
                attempt_metadata.ended_at = ended_at
                attempt_metadata.error_type = type(ex).__name__
                attempt_metadata.error_message = str(ex)
                attempt_metadata.is_retryable = is_retryable_error(ex)
                attempts.append(attempt_metadata)
                last_exception = ex

                log_error(
                    f"Attempt {attempt_number} failed",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "error_type": type(ex).__name__,
                        "error_message": str(ex),
                        "is_retryable": attempt_metadata.is_retryable,
                    },
                    logger_name=_LOGGER_NAME,
                )

                if not attempt_metadata.is_retryable:
                    log_info(
                        "Non-retryable error encountered, stopping fallback chain",
                        context={"error_type": type(ex).__name__},
                        logger_name=_LOGGER_NAME,
                    )
                    raise ex

                if attempt_number == len(fallback_chain):
                    log_error(
                        "All fallback attempts exhausted",
                        context={"total_attempts": len(attempts)},
                        logger_name=_LOGGER_NAME,
                    )
                    raise ex

                log_info(
                    f"Retryable error, continuing to next fallback ({attempt_number + 1}/{len(fallback_chain)})",
                    logger_name=_LOGGER_NAME,
                )

        if last_exception:
            raise last_exception
        raise RuntimeError(f"{error_label} fallback chain execution failed unexpectedly")

    # ------------------------------------------------------------------
    # Fallback chain collectors (kept as static helpers for backward compat)
    # ------------------------------------------------------------------

    @staticmethod
    def collect_fallback_chain(
        config: VideoGenerationConfig,
    ) -> list[VideoGenerationConfig]:
        """Collect the full fallback chain for video generation."""
        return _collect_fallback_chain(config)

    @staticmethod
    def collect_image_fallback_chain(
        config: ImageGenerationConfig,
    ) -> list[ImageGenerationConfig]:
        """Collect fallback chain for image generation."""
        return _collect_fallback_chain(config)

    @staticmethod
    def collect_audio_fallback_chain(
        config: AudioGenerationConfig,
    ) -> list[AudioGenerationConfig]:
        """Collect fallback chain for audio generation."""
        return _collect_fallback_chain(config)

    @staticmethod
    def collect_multi_modal_fallback_chain(
        config: MultiModalGenerationConfig,
    ) -> list[MultiModalGenerationConfig]:
        """Collect fallback chain for multi-modal generation."""
        return _collect_fallback_chain(config)

    # ------------------------------------------------------------------
    # Video Generation
    # ------------------------------------------------------------------

    async def execute_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Execute video generation asynchronously with fallback support."""
        return await self._execute_chain_async(
            self.collect_fallback_chain(config),
            request,
            on_progress,
            handler_method="generate_video_async",
            error_label="Video",
        )

    def execute_sync(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Execute video generation synchronously with fallback support."""
        return self._execute_chain_sync(
            self.collect_fallback_chain(config),
            request,
            on_progress,
            handler_method="generate_video",
            error_label="Video",
        )

    # ------------------------------------------------------------------
    # Image Generation
    # ------------------------------------------------------------------

    async def execute_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Execute image generation with fallback support (async)."""
        return await self._execute_chain_async(
            self.collect_image_fallback_chain(config),
            request,
            on_progress,
            handler_method="generate_image_async",
            error_label="Image",
        )

    def execute_image_sync(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Execute image generation with fallback support (sync)."""
        return self._execute_chain_sync(
            self.collect_image_fallback_chain(config),
            request,
            on_progress,
            handler_method="generate_image",
            error_label="Image",
        )

    # ------------------------------------------------------------------
    # TTS Generation
    # ------------------------------------------------------------------

    async def execute_tts_async(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback | None = None,
    ) -> TTSResponse:
        """Execute TTS generation with fallback support (async)."""
        return await self._execute_chain_async(
            self.collect_audio_fallback_chain(config),
            request,
            on_progress,
            handler_method="generate_tts_async",
            error_label="TTS",
        )

    def execute_tts_sync(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback | None = None,
    ) -> TTSResponse:
        """Execute TTS generation with fallback support (sync)."""
        return self._execute_chain_sync(
            self.collect_audio_fallback_chain(config),
            request,
            on_progress,
            handler_method="generate_tts",
            error_label="TTS",
        )

    # ------------------------------------------------------------------
    # STS Generation
    # ------------------------------------------------------------------

    async def execute_sts_async(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback | None = None,
    ) -> STSResponse:
        """Execute STS generation with fallback support (async)."""
        return await self._execute_chain_async(
            self.collect_audio_fallback_chain(config),
            request,
            on_progress,
            handler_method="generate_sts_async",
            error_label="STS",
        )

    def execute_sts_sync(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback | None = None,
    ) -> STSResponse:
        """Execute STS generation with fallback support (sync)."""
        return self._execute_chain_sync(
            self.collect_audio_fallback_chain(config),
            request,
            on_progress,
            handler_method="generate_sts",
            error_label="STS",
        )

    # ------------------------------------------------------------------
    # Multi-Modal Generation
    # ------------------------------------------------------------------

    async def execute_multi_modal_async(
        self,
        config: MultiModalGenerationConfig,
        request: MultiModalGenerationRequest,
        on_progress: MultiModalProgressCallback | None = None,
    ) -> MultiModalGenerationResponse:
        """Execute multi-modal generation with fallback support (async)."""
        return await self._execute_chain_async(
            self.collect_multi_modal_fallback_chain(config),
            request,
            on_progress,
            handler_method="generate_multi_modal_async",
            error_label="Multi-modal",
        )

    def execute_multi_modal_sync(
        self,
        config: MultiModalGenerationConfig,
        request: MultiModalGenerationRequest,
        on_progress: MultiModalProgressCallback | None = None,
    ) -> MultiModalGenerationResponse:
        """Execute multi-modal generation with fallback support (sync)."""
        return self._execute_chain_sync(
            self.collect_multi_modal_fallback_chain(config),
            request,
            on_progress,
            handler_method="generate_multi_modal",
            error_label="Multi-modal",
        )
