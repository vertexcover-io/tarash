"""Orchestrator for managing video and image generation execution with fallback support."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from tarash.tarash_gateway.logging import log_error, log_info
from tarash.tarash_gateway.exceptions import is_retryable_error
from tarash.tarash_gateway.models import (
    AttemptMetadata,
    AudioGenerationConfig,
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


class ExecutionOrchestrator:
    """Manages provider execution with automatic fallback and metadata tracking.

    Traverses the fallback chain depth-first, retrying with each successive
    provider on retryable errors. Attaches ``ExecutionMetadata`` to every
    response for observability.
    """

    # ------------------------------------------------------------------
    # Generic fallback infrastructure (Phase 1)
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_fallback_chain(config: Any) -> list[Any]:
        """Collect the full fallback chain using depth-first traversal.

        Uses duck typing — reads ``config.fallback_configs`` via
        :func:`getattr` so it works with any config type
        (``VideoGenerationConfig``, ``ImageGenerationConfig``,
        ``AudioGenerationConfig``).

        Args:
            config: Root config (primary provider). Its ``fallback_configs``
                are recursively traversed depth-first.

        Returns:
            Ordered list of config objects to try, starting with *config*
            itself.
        """
        chain: list[Any] = [config]
        fallbacks = getattr(config, "fallback_configs", None)
        if fallbacks:
            for fallback in fallbacks:
                chain.extend(
                    ExecutionOrchestrator._collect_fallback_chain(fallback)
                )
        return chain

    async def _execute_with_fallback_async(
        self,
        chain: list[Any],
        invoke_handler: Callable[[Any, Any], Awaitable[Any]],
        label: str,
    ) -> Any:
        """Execute an async handler across a fallback chain.

        Args:
            chain: Pre-collected fallback chain (from ``_collect_fallback_chain``).
            invoke_handler: Async callable ``(handler, cfg) -> response``.
                The caller binds request / on_progress into this via closure.
            label: Modality name used in log messages (e.g. ``"video"``).

        Returns:
            The first successful response, with ``ExecutionMetadata`` attached.

        Raises:
            NotImplementedError: Re-raised immediately if a handler raises it.
            Exception: The last exception if all providers fail.
        """
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

        log_info(
            f"Starting {label} fallback chain execution",
            context={
                "configs_in_chain": len(chain),
                "primary_provider": chain[0].provider,
                "primary_model": chain[0].model,
            },
            logger_name="tarash.tarash_gateway.orchestrator",
        )

        for attempt_number, cfg in enumerate(chain, start=1):
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
                    f"Attempting {label} with provider "
                    f"(attempt {attempt_number}/{len(chain)})",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "attempt_number": attempt_number,
                    },
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

                handler = get_handler(cfg)
                response = await invoke_handler(handler, cfg)

                # Success
                ended_at = datetime.now()
                attempt_metadata.ended_at = ended_at
                attempt_metadata.status = "success"
                attempt_metadata.request_id = response.request_id
                attempts.append(attempt_metadata)

                execution_metadata = ExecutionMetadata(
                    total_attempts=len(attempts),
                    successful_attempt=attempt_number,
                    attempts=attempts,
                    fallback_triggered=attempt_number > 1,
                    configs_in_chain=len(chain),
                )

                log_info(
                    f"Successfully generated {label} on attempt {attempt_number}",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "request_id": response.request_id,
                        "total_attempts": len(attempts),
                    },
                    logger_name="tarash.tarash_gateway.orchestrator",
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
                    f"{label.capitalize()} attempt {attempt_number} failed",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "error_type": type(ex).__name__,
                        "error_message": str(ex),
                        "is_retryable": attempt_metadata.is_retryable,
                    },
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

                if not attempt_metadata.is_retryable:
                    log_info(
                        "Non-retryable error encountered, stopping fallback chain",
                        context={"error_type": type(ex).__name__},
                        logger_name="tarash.tarash_gateway.orchestrator",
                    )
                    raise

                if attempt_number == len(chain):
                    log_error(
                        f"All {label} fallback attempts exhausted",
                        context={"total_attempts": len(attempts)},
                        logger_name="tarash.tarash_gateway.orchestrator",
                    )
                    raise

                log_info(
                    f"Retryable error, continuing to next fallback "
                    f"({attempt_number + 1}/{len(chain)})",
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

        if last_exception:
            raise last_exception
        raise RuntimeError(f"{label.capitalize()} fallback chain execution failed unexpectedly")

    def _execute_with_fallback_sync(
        self,
        chain: list[Any],
        invoke_handler: Callable[[Any, Any], Any],
        label: str,
    ) -> Any:
        """Execute a sync handler across a fallback chain.

        Synchronous counterpart of ``_execute_with_fallback_async``.

        Args:
            chain: Pre-collected fallback chain (from ``_collect_fallback_chain``).
            invoke_handler: Sync callable ``(handler, cfg) -> response``.
            label: Modality name used in log messages (e.g. ``"video"``).

        Returns:
            The first successful response, with ``ExecutionMetadata`` attached.

        Raises:
            NotImplementedError: Re-raised immediately if a handler raises it.
            Exception: The last exception if all providers fail.
        """
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

        log_info(
            f"Starting {label} fallback chain execution (sync)",
            context={
                "configs_in_chain": len(chain),
                "primary_provider": chain[0].provider,
                "primary_model": chain[0].model,
            },
            logger_name="tarash.tarash_gateway.orchestrator",
        )

        for attempt_number, cfg in enumerate(chain, start=1):
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
                    f"Attempting {label} with provider "
                    f"(attempt {attempt_number}/{len(chain)})",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "attempt_number": attempt_number,
                    },
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

                handler = get_handler(cfg)
                response = invoke_handler(handler, cfg)

                # Success
                ended_at = datetime.now()
                attempt_metadata.ended_at = ended_at
                attempt_metadata.status = "success"
                attempt_metadata.request_id = response.request_id
                attempts.append(attempt_metadata)

                execution_metadata = ExecutionMetadata(
                    total_attempts=len(attempts),
                    successful_attempt=attempt_number,
                    attempts=attempts,
                    fallback_triggered=attempt_number > 1,
                    configs_in_chain=len(chain),
                )

                log_info(
                    f"Successfully generated {label} on attempt {attempt_number}",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "request_id": response.request_id,
                        "total_attempts": len(attempts),
                    },
                    logger_name="tarash.tarash_gateway.orchestrator",
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
                    f"{label.capitalize()} attempt {attempt_number} failed",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "error_type": type(ex).__name__,
                        "error_message": str(ex),
                        "is_retryable": attempt_metadata.is_retryable,
                    },
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

                if not attempt_metadata.is_retryable:
                    log_info(
                        "Non-retryable error encountered, stopping fallback chain",
                        context={"error_type": type(ex).__name__},
                        logger_name="tarash.tarash_gateway.orchestrator",
                    )
                    raise

                if attempt_number == len(chain):
                    log_error(
                        f"All {label} fallback attempts exhausted",
                        context={"total_attempts": len(attempts)},
                        logger_name="tarash.tarash_gateway.orchestrator",
                    )
                    raise

                log_info(
                    f"Retryable error, continuing to next fallback "
                    f"({attempt_number + 1}/{len(chain)})",
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

        if last_exception:
            raise last_exception
        raise RuntimeError(f"{label.capitalize()} fallback chain execution failed unexpectedly")

    # ------------------------------------------------------------------
    # Public execute methods (thin wrappers)
    # ------------------------------------------------------------------

    async def execute_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Execute video generation asynchronously with fallback support."""
        chain = self._collect_fallback_chain(config)
        return await self._execute_with_fallback_async(
            chain,
            lambda handler, cfg: handler.generate_video_async(cfg, request, on_progress=on_progress),
            "video",
        )

    def execute_sync(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Execute video generation synchronously with fallback support."""
        chain = self._collect_fallback_chain(config)
        return self._execute_with_fallback_sync(
            chain,
            lambda handler, cfg: handler.generate_video(cfg, request, on_progress=on_progress),
            "video",
        )

    async def execute_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Execute image generation with fallback support (async)."""
        chain = self._collect_fallback_chain(config)
        return await self._execute_with_fallback_async(
            chain,
            lambda handler, cfg: handler.generate_image_async(cfg, request, on_progress=on_progress),
            "image",
        )

    def execute_image_sync(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Execute image generation with fallback support (sync)."""
        chain = self._collect_fallback_chain(config)
        return self._execute_with_fallback_sync(
            chain,
            lambda handler, cfg: handler.generate_image(cfg, request, on_progress=on_progress),
            "image",
        )

    async def execute_tts_async(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback | None = None,
    ) -> TTSResponse:
        """Execute TTS generation with fallback support (async)."""
        chain = self._collect_fallback_chain(config)
        return await self._execute_with_fallback_async(
            chain,
            lambda handler, cfg: handler.generate_tts_async(cfg, request, on_progress=on_progress),
            "tts",
        )

    def execute_tts_sync(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback | None = None,
    ) -> TTSResponse:
        """Execute TTS generation with fallback support (sync)."""
        chain = self._collect_fallback_chain(config)
        return self._execute_with_fallback_sync(
            chain,
            lambda handler, cfg: handler.generate_tts(cfg, request, on_progress=on_progress),
            "tts",
        )

    async def execute_sts_async(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback | None = None,
    ) -> STSResponse:
        """Execute STS generation with fallback support (async)."""
        chain = self._collect_fallback_chain(config)
        return await self._execute_with_fallback_async(
            chain,
            lambda handler, cfg: handler.generate_sts_async(cfg, request, on_progress=on_progress),
            "sts",
        )

    def execute_sts_sync(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback | None = None,
    ) -> STSResponse:
        """Execute STS generation with fallback support (sync)."""
        chain = self._collect_fallback_chain(config)
        return self._execute_with_fallback_sync(
            chain,
            lambda handler, cfg: handler.generate_sts(cfg, request, on_progress=on_progress),
            "sts",
        )
