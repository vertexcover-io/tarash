"""Orchestrator for managing video and image generation execution with fallback support."""

from datetime import datetime

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

    @staticmethod
    def collect_fallback_chain(
        config: VideoGenerationConfig,
    ) -> list[VideoGenerationConfig]:
        """Collect the full fallback chain using depth-first traversal.

        Args:
            config: Root config (primary provider). Its ``fallback_configs``
                are recursively traversed depth-first.

        Returns:
            Ordered list of ``VideoGenerationConfig`` objects to try, starting
            with ``config`` itself.
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
        """Execute video generation asynchronously with fallback support.

        Iterates the fallback chain in order. On a retryable error the next
        provider is tried; on a non-retryable error execution stops immediately.

        Args:
            config: Primary configuration. Fallbacks are read from
                ``config.fallback_configs`` recursively.
            request: Video generation parameters.
            on_progress: Optional callback forwarded to the active provider.

        Returns:
            ``VideoGenerationResponse`` with ``execution_metadata`` attached.

        Raises:
            TarashException: The last exception raised if all providers fail.
                Non-retryable errors are re-raised immediately without trying
                further fallbacks.
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
        """Execute video generation synchronously with fallback support.

        Blocking version of ``execute_async``. Iterates the fallback chain in
        order, stopping on non-retryable errors.

        Args:
            config: Primary configuration. Fallbacks are read from
                ``config.fallback_configs`` recursively.
            request: Video generation parameters.
            on_progress: Optional callback forwarded to the active provider.

        Returns:
            ``VideoGenerationResponse`` with ``execution_metadata`` attached.

        Raises:
            TarashException: The last exception raised if all providers fail.
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

    # ==================== Image Generation ====================

    @staticmethod
    def collect_image_fallback_chain(
        config: ImageGenerationConfig,
    ) -> list[ImageGenerationConfig]:
        """Collect fallback chain for image generation."""
        chain = [config]
        if config.fallback_configs:
            for fallback in config.fallback_configs:
                chain.extend(
                    ExecutionOrchestrator.collect_image_fallback_chain(fallback)
                )
        return chain

    async def execute_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Execute image generation with fallback support (async)."""
        fallback_chain = self.collect_image_fallback_chain(config)
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

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
                handler = get_handler(cfg)
                response = await handler.generate_image_async(
                    cfg, request, on_progress=on_progress
                )

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

                if not attempt_metadata.is_retryable or attempt_number == len(
                    fallback_chain
                ):
                    raise ex

        if last_exception:
            raise last_exception
        raise RuntimeError("Image fallback chain execution failed")

    def execute_image_sync(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Execute image generation with fallback support (sync)."""
        fallback_chain = self.collect_image_fallback_chain(config)
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

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
                handler = get_handler(cfg)
                response = handler.generate_image(cfg, request, on_progress=on_progress)

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

                if not attempt_metadata.is_retryable or attempt_number == len(
                    fallback_chain
                ):
                    raise ex

        if last_exception:
            raise last_exception
        raise RuntimeError("Image fallback chain execution failed")

    # ==================== TTS Generation ====================

    @staticmethod
    def collect_audio_fallback_chain(
        config: AudioGenerationConfig,
    ) -> list[AudioGenerationConfig]:
        """Collect fallback chain for audio generation."""
        chain = [config]
        if config.fallback_configs:
            for fallback in config.fallback_configs:
                chain.extend(
                    ExecutionOrchestrator.collect_audio_fallback_chain(fallback)
                )
        return chain

    async def execute_tts_async(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback | None = None,
    ) -> TTSResponse:
        """Execute TTS generation with fallback support (async)."""
        fallback_chain = self.collect_audio_fallback_chain(config)
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

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
                handler = get_handler(cfg)
                response = await handler.generate_tts_async(
                    cfg, request, on_progress=on_progress
                )

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

                if not attempt_metadata.is_retryable or attempt_number == len(
                    fallback_chain
                ):
                    raise ex

        if last_exception:
            raise last_exception
        raise RuntimeError("TTS fallback chain execution failed")

    def execute_tts_sync(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback | None = None,
    ) -> TTSResponse:
        """Execute TTS generation with fallback support (sync)."""
        fallback_chain = self.collect_audio_fallback_chain(config)
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

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
                handler = get_handler(cfg)
                response = handler.generate_tts(cfg, request, on_progress=on_progress)

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

                if not attempt_metadata.is_retryable or attempt_number == len(
                    fallback_chain
                ):
                    raise ex

        if last_exception:
            raise last_exception
        raise RuntimeError("TTS fallback chain execution failed")

    # ==================== STS Generation ====================

    async def execute_sts_async(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback | None = None,
    ) -> STSResponse:
        """Execute STS generation with fallback support (async)."""
        fallback_chain = self.collect_audio_fallback_chain(config)
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

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
                handler = get_handler(cfg)
                response = await handler.generate_sts_async(
                    cfg, request, on_progress=on_progress
                )

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

                if not attempt_metadata.is_retryable or attempt_number == len(
                    fallback_chain
                ):
                    raise ex

        if last_exception:
            raise last_exception
        raise RuntimeError("STS fallback chain execution failed")

    def execute_sts_sync(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback | None = None,
    ) -> STSResponse:
        """Execute STS generation with fallback support (sync)."""
        fallback_chain = self.collect_audio_fallback_chain(config)
        attempts: list[AttemptMetadata] = []
        last_exception: Exception | None = None

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
                handler = get_handler(cfg)
                response = handler.generate_sts(cfg, request, on_progress=on_progress)

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

                if not attempt_metadata.is_retryable or attempt_number == len(
                    fallback_chain
                ):
                    raise ex

        if last_exception:
            raise last_exception
        raise RuntimeError("STS fallback chain execution failed")
