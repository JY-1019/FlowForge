"""ToolRegistry — holds all registered tool adapters."""
from __future__ import annotations

from typing import Any

from flowforge.tools.base import ToolAdapter


class ToolRegistry:
    """Registry for all available tools in a FlowForge agent."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolAdapter] = {}

    def register(self, adapter: ToolAdapter) -> None:
        self._tools[adapter.name] = adapter

    def get(self, name: str) -> ToolAdapter | None:
        return self._tools.get(name)

    def all(self) -> list[ToolAdapter]:
        return list(self._tools.values())

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_tool() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
