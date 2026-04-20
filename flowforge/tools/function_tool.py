"""Python function → ToolAdapter wrapper."""
from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from flowforge.tools.base import ToolAdapter


def _infer_schema_from_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Auto-generate a JSON Schema ``input_schema`` from *func*'s type hints.

    Inspects ``inspect.signature`` and ``typing.get_type_hints`` to build a
    schema with ``properties`` and ``required`` lists.  Only parameters with
    recognised scalar types are included; unknown types are left as
    ``{"type": "string"}`` (a safe default for LLM tool calls).

    Parameters without a default value are added to ``required``.
    """
    _TYPE_MAP: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls", "ctx"):
            continue

        hint = hints.get(name)
        prop: dict[str, Any] = {}

        if hint is not None:
            # Handle Optional[X] / X | None → unwrap to X.
            origin = getattr(hint, "__origin__", None)
            args = getattr(hint, "__args__", ())
            if origin is type(int | str):  # types.UnionType (3.10+)
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    hint = non_none[0]
            elif origin is not None:
                # typing.Union
                import typing
                if origin is typing.Union:
                    non_none = [a for a in args if a is not type(None)]
                    if len(non_none) == 1:
                        hint = non_none[0]

            # list[X] → {"type": "array", "items": ...}
            list_origin = getattr(hint, "__origin__", None)
            if list_origin is list:
                item_args = getattr(hint, "__args__", ())
                if item_args and item_args[0] in _TYPE_MAP:
                    prop = {"type": "array", "items": {"type": _TYPE_MAP[item_args[0]]}}
                else:
                    prop = {"type": "array"}
            elif hint in _TYPE_MAP:
                prop = {"type": _TYPE_MAP[hint]}
            else:
                prop = {"type": "string"}
        else:
            prop = {"type": "string"}

        # Default value → "default" field.
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)

        # Use docstring-style description if we ever add support; skip for now.
        properties[name] = prop

    return {"type": "object", "properties": properties, "required": required}


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
        self._schema = schema or _infer_schema_from_hints(func)

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
