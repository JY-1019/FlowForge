"""Abstract base for tool adapters."""
from __future__ import annotations

import abc
from typing import Any


class ToolAdapter(abc.ABC):
    """Abstract base class for all tool adapters."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique tool name."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description."""

    @property
    def schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's input parameters."""
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @abc.abstractmethod
    async def call(self, **kwargs: Any) -> Any:
        """Execute the tool with the given parameters."""

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Return the Anthropic tool_use schema for this adapter."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }
