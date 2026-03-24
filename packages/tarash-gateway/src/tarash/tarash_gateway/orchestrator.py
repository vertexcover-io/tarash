"""Orchestrator for managing video and image generation execution with fallback support."""

from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from tarash.tarash_gateway.logging import log_error, log_info
from tarash.tarash_gateway.exceptions import is_retryable_error
from tarash.tarash_gateway.models import (
    AttemptMetadata,
    ExecutionMetadata,
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageProgressCallback,
    ProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.registry import get_handler

ConfigT = TypeVar("ConfigT", VideoGenerationConfig, ImageGenerationConfig)
ResponseT = TypeVar("ResponseT", VideoGenerationResponse, ImageGenerationResponse)


def _collect_fallback_chain(config: ConfigT) -> list[ConfigT]:
    """Collect fallback chain using depth-first traversal.

    Works for any config type that has a fallback_configs attribute.
    """
    chain: list[ConfigT] = [config]
    if config.fallback_configs:
        for fallback in config.fallback_configs:
            chain.extend(_collect_fallback_chain(fallback))  # type: ignore[arg-type]
    return chain


class ExecutionOrchestrator:
    """Orchestrator for managing fallback chain execution with metadata tracking."""

    @staticmethod
    def collect_fallback_chain(
        config: VideoGenerationConfig,
    ) -> list[VideoGenerationConfig]:
        """Collect fallback chain for video generation configs."""
        return _collect_fallback_chain(config)

    @staticmethod
    def collect_image_fallback_chain(
        config: ImageGenerationConfig,
    ) -> list[ImageGenerationConfig]:
        """Collect fallback chain for image generation configs."""
        return _collect_fallback_chain(config)

    async def _execute_with_fallback_async(
        self,
        config: ConfigT,
        request: BaseModel,
        handler_method: str,
        on_progress: Any = None,
    ) -> ResponseT:
        """Execute generation with fallback support (async).

        Args:
            config: Primary configuration (video or image)
            request: Generation request
            handler_method: Name of the async handler method to call
            on_progress: Optional progress callback
        """
        fallback_chain = _collect_fallback_chain(config)
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

        log_info(
            "Starting fallback chain execution",
            context={
                "configs_in_chain": len(fallback_chain),
                "primary_provider": config.provider,
                "primary_model": config.model,
            },
            logger_name="tarash.tarash_gateway.orchestrator",
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
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

                handler = get_handler(cfg)
                method = getattr(handler, handler_method)
                response = await method(cfg, request, on_progress=on_progress)

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
                    configs_in_chain=len(fallback_chain),
                )

                log_info(
                    f"Successfully generated on attempt {attempt_number}",
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
                    f"Attempt {attempt_number} failed",
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
                    raise ex

                if attempt_number == len(fallback_chain):
                    log_error(
                        "All fallback attempts exhausted",
                        context={"total_attempts": len(attempts)},
                        logger_name="tarash.tarash_gateway.orchestrator",
                    )
                    raise ex

                log_info(
                    f"Retryable error, continuing to next fallback ({attempt_number + 1}/{len(fallback_chain)})",
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

        if last_exception:
            raise last_exception
        raise RuntimeError("Fallback chain execution failed unexpectedly")

    def _execute_with_fallback_sync(
        self,
        config: ConfigT,
        request: BaseModel,
        handler_method: str,
        on_progress: Any = None,
    ) -> ResponseT:
        """Execute generation with fallback support (sync).

        Args:
            config: Primary configuration (video or image)
            request: Generation request
            handler_method: Name of the sync handler method to call
            on_progress: Optional progress callback
        """
        fallback_chain = _collect_fallback_chain(config)
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

        log_info(
            "Starting fallback chain execution (sync)",
            context={
                "configs_in_chain": len(fallback_chain),
                "primary_provider": config.provider,
                "primary_model": config.model,
            },
            logger_name="tarash.tarash_gateway.orchestrator",
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
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

                handler = get_handler(cfg)
                method = getattr(handler, handler_method)
                response = method(cfg, request, on_progress=on_progress)

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
                    configs_in_chain=len(fallback_chain),
                )

                log_info(
                    f"Successfully generated on attempt {attempt_number}",
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
                    f"Attempt {attempt_number} failed",
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
                    raise ex

                if attempt_number == len(fallback_chain):
                    log_error(
                        "All fallback attempts exhausted",
                        context={"total_attempts": len(attempts)},
                        logger_name="tarash.tarash_gateway.orchestrator",
                    )
                    raise ex

                log_info(
                    f"Retryable error, continuing to next fallback ({attempt_number + 1}/{len(fallback_chain)})",
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

        if last_exception:
            raise last_exception
        raise RuntimeError("Fallback chain execution failed unexpectedly")

    # ==================== Video Generation ====================

    async def execute_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Execute video generation with fallback support (async)."""
        return await self._execute_with_fallback_async(
            config, request, "generate_video_async", on_progress
        )

    def execute_sync(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Execute video generation with fallback support (sync)."""
        return self._execute_with_fallback_sync(
            config, request, "generate_video", on_progress
        )

    # ==================== Image Generation ====================

    async def execute_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Execute image generation with fallback support (async)."""
        return await self._execute_with_fallback_async(
            config, request, "generate_image_async", on_progress
        )

    def execute_image_sync(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Execute image generation with fallback support (sync)."""
        return self._execute_with_fallback_sync(
            config, request, "generate_image", on_progress
        )
