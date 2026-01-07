"""Orchestrator for managing video generation execution with fallback support."""

from datetime import datetime

from tarash.tarash_gateway.logging import log_error, log_info
from tarash.tarash_gateway.exceptions import is_retryable_error
from tarash.tarash_gateway.models import (
    AttemptMetadata,
    ExecutionMetadata,
    ProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.registry import get_handler


class ExecutionOrchestrator:
    """Orchestrator for managing fallback chain execution with metadata tracking."""

    @staticmethod
    def collect_fallback_chain(
        config: VideoGenerationConfig,
    ) -> list[VideoGenerationConfig]:
        """Collect fallback chain using depth-first traversal.

        Args:
            config: Primary configuration (root of fallback tree)

        Returns:
            List of configs in execution order (depth-first)
        """
        chain = [config]

        if config.fallback_configs:
            for fallback in config.fallback_configs:
                # Recursively collect fallbacks (depth-first)
                chain.extend(ExecutionOrchestrator.collect_fallback_chain(fallback))

        return chain

    async def execute_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Execute video generation with fallback support (async).

        Args:
            config: Primary video generation configuration
            request: Video generation request
            on_progress: Optional progress callback

        Returns:
            VideoGenerationResponse with execution metadata

        Raises:
            Last exception if all attempts fail
        """
        fallback_chain = self.collect_fallback_chain(config)
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

                # Get handler and execute
                handler = get_handler(cfg)
                response = await handler.generate_video_async(
                    cfg, request, on_progress=on_progress
                )

                # Success!
                ended_at = datetime.now()
                attempt_metadata.ended_at = ended_at
                attempt_metadata.status = "success"
                attempt_metadata.request_id = response.request_id
                attempts.append(attempt_metadata)

                # Attach execution metadata to response
                execution_metadata = ExecutionMetadata(
                    total_attempts=len(attempts),
                    successful_attempt=attempt_number,
                    attempts=attempts,
                    fallback_triggered=attempt_number > 1,
                    configs_in_chain=len(fallback_chain),
                )

                log_info(
                    f"Successfully generated video on attempt {attempt_number}",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "request_id": response.request_id,
                        "total_attempts": len(attempts),
                    },
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

                # Return response with metadata (need to create new instance since frozen)
                return response.model_copy(
                    update={"execution_metadata": execution_metadata}
                )

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

                # If error is not retryable, stop immediately
                if not attempt_metadata.is_retryable:
                    log_info(
                        "Non-retryable error encountered, stopping fallback chain",
                        context={"error_type": type(ex).__name__},
                        logger_name="tarash.tarash_gateway.orchestrator",
                    )
                    raise ex

                # If this was the last config, raise the error
                if attempt_number == len(fallback_chain):
                    log_error(
                        "All fallback attempts exhausted",
                        context={"total_attempts": len(attempts)},
                        logger_name="tarash.tarash_gateway.orchestrator",
                    )
                    raise ex

                # Otherwise, continue to next fallback
                log_info(
                    f"Retryable error, continuing to next fallback ({attempt_number + 1}/{len(fallback_chain)})",
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

        # Should never reach here, but raise last exception if we do
        if last_exception:
            raise last_exception
        raise RuntimeError("Fallback chain execution failed unexpectedly")

    def execute_sync(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Execute video generation with fallback support (sync).

        Args:
            config: Primary video generation configuration
            request: Video generation request
            on_progress: Optional progress callback

        Returns:
            VideoGenerationResponse with execution metadata

        Raises:
            Last exception if all attempts fail
        """
        fallback_chain = self.collect_fallback_chain(config)
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

                # Get handler and execute
                handler = get_handler(cfg)
                response = handler.generate_video(cfg, request, on_progress=on_progress)

                # Success!
                ended_at = datetime.now()
                attempt_metadata.ended_at = ended_at
                attempt_metadata.status = "success"
                attempt_metadata.request_id = response.request_id
                attempts.append(attempt_metadata)

                # Attach execution metadata to response
                execution_metadata = ExecutionMetadata(
                    total_attempts=len(attempts),
                    successful_attempt=attempt_number,
                    attempts=attempts,
                    fallback_triggered=attempt_number > 1,
                    configs_in_chain=len(fallback_chain),
                )

                log_info(
                    f"Successfully generated video on attempt {attempt_number}",
                    context={
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "request_id": response.request_id,
                        "total_attempts": len(attempts),
                    },
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

                # Return response with metadata (need to create new instance since frozen)
                return response.model_copy(
                    update={"execution_metadata": execution_metadata}
                )

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

                # If error is not retryable, stop immediately
                if not attempt_metadata.is_retryable:
                    log_info(
                        "Non-retryable error encountered, stopping fallback chain",
                        context={"error_type": type(ex).__name__},
                        logger_name="tarash.tarash_gateway.orchestrator",
                    )
                    raise ex

                # If this was the last config, raise the error
                if attempt_number == len(fallback_chain):
                    log_error(
                        "All fallback attempts exhausted",
                        context={"total_attempts": len(attempts)},
                        logger_name="tarash.tarash_gateway.orchestrator",
                    )
                    raise ex

                # Otherwise, continue to next fallback
                log_info(
                    f"Retryable error, continuing to next fallback ({attempt_number + 1}/{len(fallback_chain)})",
                    logger_name="tarash.tarash_gateway.orchestrator",
                )

        # Should never reach here, but raise last exception if we do
        if last_exception:
            raise last_exception
        raise RuntimeError("Fallback chain execution failed unexpectedly")
