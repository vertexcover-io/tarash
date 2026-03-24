"""Orchestrator for managing video, image, and audio generation execution with fallback support."""

from datetime import datetime

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

# Union of all config types that share provider, model, fallback_configs fields
_AnyConfig = VideoGenerationConfig | ImageGenerationConfig | AudioGenerationConfig

# Union of all request types
_AnyRequest = VideoGenerationRequest | ImageGenerationRequest | TTSRequest | STSRequest

# Union of all progress callback types
_AnyProgressCallback = (
    ProgressCallback
    | ImageProgressCallback
    | TTSProgressCallback
    | STSProgressCallback
    | None
)

# Union of all response types
_AnyResponse = (
    VideoGenerationResponse | ImageGenerationResponse | TTSResponse | STSResponse
)


class ExecutionOrchestrator:
    """Manages provider execution with automatic fallback and metadata tracking.

    Traverses the fallback chain depth-first, retrying with each successive
    provider on retryable errors. Attaches ``ExecutionMetadata`` to every
    response for observability.
    """

    @staticmethod
    def _collect_fallback_chain(
        config: _AnyConfig,
    ) -> list[_AnyConfig]:
        """Collect the full fallback chain using depth-first traversal.

        Args:
            config: Root config (primary provider). Its ``fallback_configs``
                are recursively traversed depth-first.

        Returns:
            Ordered list of config objects to try, starting with ``config`` itself.
        """
        chain = [config]

        if config.fallback_configs:
            for fallback in config.fallback_configs:
                chain.extend(ExecutionOrchestrator._collect_fallback_chain(fallback))

        return chain

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
        return ExecutionOrchestrator._collect_fallback_chain(config)  # type: ignore[return-value]

    async def _execute_with_fallback_async(
        self,
        fallback_chain: list[_AnyConfig],
        handler_method: str,
        config: _AnyConfig,
        request: _AnyRequest,
        on_progress: _AnyProgressCallback = None,
    ) -> _AnyResponse:
        """Execute generation asynchronously with fallback support.

        Iterates the fallback chain in order. On a retryable error the next
        provider is tried; on a non-retryable error execution stops immediately.
        ``NotImplementedError`` is always re-raised immediately.

        Args:
            fallback_chain: Ordered list of configs to try.
            handler_method: Name of the async handler method to call.
            config: The original config (used for logging context).
            request: Generation request parameters.
            on_progress: Optional callback forwarded to the active provider.

        Returns:
            Response with ``execution_metadata`` attached.

        Raises:
            NotImplementedError: If the handler does not support this method.
            TarashException: The last exception raised if all providers fail.
        """
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
                response = await getattr(handler, handler_method)(
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
        raise RuntimeError("Fallback chain execution failed unexpectedly")

    def _execute_with_fallback_sync(
        self,
        fallback_chain: list[_AnyConfig],
        handler_method: str,
        config: _AnyConfig,
        request: _AnyRequest,
        on_progress: _AnyProgressCallback = None,
    ) -> _AnyResponse:
        """Execute generation synchronously with fallback support.

        Blocking version of ``_execute_with_fallback_async``. Iterates the
        fallback chain in order, stopping on non-retryable errors.
        ``NotImplementedError`` is always re-raised immediately.

        Args:
            fallback_chain: Ordered list of configs to try.
            handler_method: Name of the sync handler method to call.
            config: The original config (used for logging context).
            request: Generation request parameters.
            on_progress: Optional callback forwarded to the active provider.

        Returns:
            Response with ``execution_metadata`` attached.

        Raises:
            NotImplementedError: If the handler does not support this method.
            TarashException: The last exception raised if all providers fail.
        """
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
                response = getattr(handler, handler_method)(
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
        raise RuntimeError("Fallback chain execution failed unexpectedly")

    # ==================== Video Generation ====================

    async def execute_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Execute video generation asynchronously with fallback support."""
        chain = self._collect_fallback_chain(config)
        return await self._execute_with_fallback_async(
            chain, "generate_video_async", config, request, on_progress
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
            chain, "generate_video", config, request, on_progress
        )

    # ==================== Image Generation ====================

    @staticmethod
    def collect_image_fallback_chain(
        config: ImageGenerationConfig,
    ) -> list[ImageGenerationConfig]:
        """Collect fallback chain for image generation."""
        return ExecutionOrchestrator._collect_fallback_chain(config)  # type: ignore[return-value]

    async def execute_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Execute image generation with fallback support (async)."""
        chain = self._collect_fallback_chain(config)
        return await self._execute_with_fallback_async(
            chain, "generate_image_async", config, request, on_progress
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
            chain, "generate_image", config, request, on_progress
        )

    # ==================== TTS Generation ====================

    @staticmethod
    def collect_audio_fallback_chain(
        config: AudioGenerationConfig,
    ) -> list[AudioGenerationConfig]:
        """Collect fallback chain for audio generation."""
        return ExecutionOrchestrator._collect_fallback_chain(config)  # type: ignore[return-value]

    async def execute_tts_async(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback | None = None,
    ) -> TTSResponse:
        """Execute TTS generation with fallback support (async)."""
        chain = self._collect_fallback_chain(config)
        return await self._execute_with_fallback_async(
            chain, "generate_tts_async", config, request, on_progress
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
            chain, "generate_tts", config, request, on_progress
        )

    # ==================== STS Generation ====================

    async def execute_sts_async(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback | None = None,
    ) -> STSResponse:
        """Execute STS generation with fallback support (async)."""
        chain = self._collect_fallback_chain(config)
        return await self._execute_with_fallback_async(
            chain, "generate_sts_async", config, request, on_progress
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
            chain, "generate_sts", config, request, on_progress
        )
