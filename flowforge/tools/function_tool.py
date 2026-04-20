"""Python function → ToolAdapter wrapper."""
from __future__ import annotations

import inspect
from typing import Any, Callable

from flowforge.tools.base import ToolAdapter


class FunctionToolAdapter(ToolAdapter):
    """Wraps a Python async (or sync) function as a tool."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self._func = func
        self._name = name or func.__name__
        self._description = description or (inspect.getdoc(func) or f"Call {func.__name__}")
        self._schema = schema or {"type": "object", "properties": {}, "required": []}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def schema(self) -> dict[str, Any]:
        return self._schema

    async def call(self, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(self._func):
            return await self._func(**kwargs)
        return self._func(**kwargs)
