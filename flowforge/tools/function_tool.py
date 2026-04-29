"""Python function → ToolAdapter wrapper."""
from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Any, Callable, get_args, get_origin, get_type_hints

from flowforge.tools.base import ToolAdapter


_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _schema_for_hint(hint: Any) -> dict[str, Any]:
    """Return a compact JSON Schema fragment for a Python type hint."""
    if hint is Any:
        return {}

    origin = get_origin(hint)
    args = get_args(hint)

    if origin in {typing.Union, types.UnionType}:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _schema_for_hint(non_none[0])
        options = [_schema_for_hint(arg) for arg in non_none]
        options = [option for option in options if option]
        return {"anyOf": options} if options else {"type": "string"}

    if origin is typing.Literal:
        values = list(args)
        prop: dict[str, Any] = {"enum": values}
        value_types = {type(value) for value in values if value is not None}
        if len(value_types) == 1 and next(iter(value_types)) in _TYPE_MAP:
            prop["type"] = _TYPE_MAP[next(iter(value_types))]
        return prop

    if origin in {list, tuple, set, frozenset, Sequence}:
        prop: dict[str, Any] = {"type": "array"}
        if args:
            item_schema = _schema_for_hint(args[0])
            if item_schema:
                prop["items"] = item_schema
        return prop

    if origin in {dict, Mapping}:
        prop = {"type": "object"}
        if len(args) >= 2:
            value_schema = _schema_for_hint(args[1])
            if value_schema:
                prop["additionalProperties"] = value_schema
        return prop

    if hint in _TYPE_MAP:
        return {"type": _TYPE_MAP[hint]}

    if hasattr(hint, "model_json_schema"):
        try:
            return hint.model_json_schema()
        except Exception:
            pass

    return {"type": "string"}


def _infer_schema_from_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Auto-generate a JSON Schema ``input_schema`` from *func*'s type hints.

    Inspects ``inspect.signature`` and ``typing.get_type_hints`` to build a
    schema with ``properties`` and ``required`` lists.  Only parameters with
    recognised scalar types are included; unknown types are left as
    ``{"type": "string"}`` (a safe default for LLM tool calls).

    Parameters without a default value are added to ``required``.
    """
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
        if param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        hint = hints.get(name)
        prop = _schema_for_hint(hint) if hint is not None else {"type": "string"}

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
