"""Batch execution engine for concurrent generation requests.

Provides an internal async batch executor that dispatches multiple
generation requests concurrently, gated by an asyncio.Semaphore.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from tarash.tarash_gateway.exceptions import TarashException, ValidationError
from tarash.tarash_gateway.logging import log_warning
from tarash.tarash_gateway.models import (
    BatchCompletionUpdate,
    BatchItem,
    BatchItemResult,
    BatchResponse,
)

ConfigT = TypeVar("ConfigT")
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


async def _execute_batch_async(
    items: list[BatchItem[ConfigT, RequestT]],
    default_config: ConfigT,
    execute_fn: Callable[..., Awaitable[ResponseT]],
    max_concurrent: int = 5,
    on_item_progress: Any | None = None,  # Modality-specific progress callback
    on_batch_progress: Callable[[BatchCompletionUpdate[ResponseT]], None] | None = None,
) -> BatchResponse[ResponseT]:
    """Execute a batch of generation requests concurrently.

    Args:
        items: List of BatchItem objects to process.
        default_config: Default config used when item.config is None.
        execute_fn: Async function to call for each item (e.g. generate_video_async).
        max_concurrent: Maximum number of concurrent requests (1-50).
        on_item_progress: Progress callback forwarded to each single-request call.
        on_batch_progress: Callback invoked after each item completes.

    Returns:
        BatchResponse with results in original submission order.

    Raises:
        ValidationError: If max_concurrent is outside [1, 50].
    """
    # REQ-11: Validate max_concurrent range
    if max_concurrent < 1 or max_concurrent > 50:
        raise ValidationError(
            f"max_concurrent must be between 1 and 50, got {max_concurrent}"
        )

    # EDGE-1: Empty batch
    if not items:
        return BatchResponse(results=[], total=0, succeeded=0, failed=0)

    # REQ-9: Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    # Shared mutable state for tracking completion
    completed_count = 0
    count_lock = asyncio.Lock()
    results: list[BatchItemResult[ResponseT]] = [None] * len(items)  # type: ignore[list-item]

    async def _execute_single(index: int, item: BatchItem[ConfigT, RequestT]) -> None:
        nonlocal completed_count

        # REQ-12: Resolve config
        effective_config = item.config if item.config is not None else default_config

        item_result: BatchItemResult[ResponseT]

        async with semaphore:
            try:
                # REQ-14, REQ-18: Call execute_fn with config, request, on_progress
                response = await execute_fn(
                    effective_config, item.request, on_progress=on_item_progress
                )
                # REQ-15: Success
                item_result = BatchItemResult(
                    index=index, status="completed", response=response
                )
            except TarashException as e:
                # REQ-16: TarashException -> failed result
                item_result = BatchItemResult(index=index, status="failed", error=e)
            except Exception as e:
                # REQ-17: Wrap non-TarashException
                wrapped = TarashException(str(e))
                item_result = BatchItemResult(
                    index=index, status="failed", error=wrapped
                )

        # REQ-20: Record result before calling on_batch_progress
        results[index] = item_result

        # Update completed count
        async with count_lock:
            completed_count += 1
            current_completed = completed_count

        # REQ-19, REQ-21: Fire batch progress callback
        if on_batch_progress is not None:
            try:
                update = BatchCompletionUpdate(
                    index=index,
                    item_result=item_result,
                    completed_count=current_completed,
                    total_count=len(items),
                )
                on_batch_progress(update)
            except Exception as cb_err:
                log_warning(
                    f"on_batch_progress callback raised an exception: {cb_err}",
                    context={"index": index, "error": str(cb_err)},
                    logger_name="tarash.tarash_gateway.batch",
                )

    # REQ-13: Dispatch all items as concurrent tasks
    tasks = [
        asyncio.create_task(_execute_single(i, item)) for i, item in enumerate(items)
    ]
    await asyncio.gather(*tasks)

    # REQ-22: Results already ordered by index
    # REQ-23, REQ-24, REQ-25, REQ-26: Compute aggregates
    succeeded = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status == "failed")

    return BatchResponse(
        results=results,
        total=len(items),
        succeeded=succeeded,
        failed=failed,
    )
