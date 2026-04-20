"""Pre/post execution callback hooks."""
from __future__ import annotations

import inspect
from typing import Any, Callable, Awaitable


HookFn = Callable[..., Any | Awaitable[Any]]


class HookRegistry:
    """Registry for pre/post execution hooks."""

    def __init__(self) -> None:
        self._pre_hooks: list[HookFn] = []
        self._post_hooks: list[HookFn] = []

    def on_before(self, fn: HookFn) -> HookFn:
        """Register a pre-execution hook."""
        self._pre_hooks.append(fn)
        return fn

    def on_after(self, fn: HookFn) -> HookFn:
        """Register a post-execution hook."""
        self._post_hooks.append(fn)
        return fn

    async def fire_pre(self, node_id: str, input_data: Any) -> None:
        for hook in self._pre_hooks:
            result = hook(node_id, input_data)
            if inspect.isawaitable(result):
                await result

    async def fire_post(self, node_id: str, output_data: Any) -> None:
        for hook in self._post_hooks:
            result = hook(node_id, output_data)
            if inspect.isawaitable(result):
                await result
