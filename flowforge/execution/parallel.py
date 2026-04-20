"""anyio TaskGroup wrapper for parallel flow/task execution."""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine


async def run_parallel(
    coroutines: list[Coroutine[Any, Any, Any]],
) -> list[Any]:
    """Run a list of coroutines concurrently and return results in order."""
    try:
        import anyio

        results: list[Any] = [None] * len(coroutines)

        async def _run(idx: int, coro: Coroutine[Any, Any, Any]) -> None:
            results[idx] = await coro

        async with anyio.create_task_group() as tg:
            for i, coro in enumerate(coroutines):
                tg.start_soon(_run, i, coro)

        return results
    except ImportError:
        # Fallback to asyncio.gather
        return list(await asyncio.gather(*coroutines))


async def run_with_timeout(
    coro: Coroutine[Any, Any, Any],
    timeout_seconds: float,
) -> Any:
    """Run a coroutine with a timeout."""
    return await asyncio.wait_for(coro, timeout=timeout_seconds)
