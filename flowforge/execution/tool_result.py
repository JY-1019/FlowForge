"""Forgiving wrapper around tool return values.

Tool functions return a ``dict`` like
``{"ok": True, "body": "<html>...", ...}`` or
``{"ok": True, "stdout": "...", "stderr": "..."}``.

Generated dynamic flows often confuse the dict for the *primary text* it
carries — e.g. ``result[:40000]`` (slicing), ``f"...{result}"`` (string
formatting), or ``str(result)``.  These should "just work" even though the
canonical access is ``result["body"]`` / ``result["content"]`` / etc.

``ToolResult`` is a ``dict`` subclass that *also* behaves like a string for
slicing, indexing by integer, and string conversion.  When asked for slice
/ str / int-index access, it transparently delegates to the first
non-empty value among a small list of conventional text fields.

Existing callers that use ``result.get("body")`` or ``result["body"]`` are
unaffected — those keep dict semantics.
"""
from __future__ import annotations

from typing import Any


_PRIMARY_TEXT_KEYS: tuple[str, ...] = (
    "body",
    "content",
    "text",
    "output",
    "stdout",
    "result",
    "stderr",
    "raw",
    "html",
)


class ToolResult(dict):
    """Dict that doubles as a string for the convenience of generated code.

    * Dict access (``result["body"]``, ``result.get("ok")``) — unchanged.
    * String access (``result[:N]``, ``result[i]``, ``str(result)``,
      ``f"{result}"``) — operates on the primary text field
      (first non-empty among ``body``, ``content``, ``stdout``, etc.).

    Designed to be tolerant of the slack typing the LLM produces in
    dynamic flows; callers that follow the catalog's documented shape see
    no change in behaviour.
    """

    __slots__ = ()

    @property
    def primary_text(self) -> str:
        """First non-empty conventional text field in this result."""
        for key in _PRIMARY_TEXT_KEYS:
            value = dict.get(self, key)
            if isinstance(value, str) and value:
                return value
        return ""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, slice)):
            return self.primary_text[key]
        if isinstance(key, str) and key in _PRIMARY_TEXT_KEYS:
            value = dict.get(self, key)
            if isinstance(value, str) and value:
                return value
            # Conventional text-key fallback: if the LLM guessed the
            # wrong synonym (e.g. ``result["content"]`` for a tool that
            # actually populates ``"body"``), return whichever sibling
            # text field is non-empty instead of raising ``KeyError``.
            sibling = self.primary_text
            if sibling:
                return sibling
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, str) and key in _PRIMARY_TEXT_KEYS:
            value = dict.get(self, key)
            if isinstance(value, str) and value:
                return value
            sibling = self.primary_text
            if sibling:
                return sibling
            return default
        return dict.get(self, key, default)

    def __str__(self) -> str:
        text = self.primary_text
        if text:
            return text
        return super().__repr__()


def wrap_tool_result(value: Any) -> Any:
    """Wrap a tool's return value in :class:`ToolResult` when applicable.

    Only plain ``dict`` instances are wrapped.  ``ToolResult`` instances
    pass through unchanged.  Lists, scalars, and Pydantic models are
    returned as-is.
    """
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, dict):
        return ToolResult(value)
    return value
