"""MCP server adapter."""
from __future__ import annotations

from typing import Any

from flowforge.tools.base import ToolAdapter


class MCPToolAdapter(ToolAdapter):
    """Adapter that proxies calls to an MCP server endpoint."""

    def __init__(self, url: str, tool_name: str, description: str = "", schema: dict[str, Any] | None = None) -> None:
        self._url = url
        self._tool_name = tool_name
        self._description = description
        self._schema = schema or {"type": "object", "properties": {}, "required": []}

    @property
    def name(self) -> str:
        return self._tool_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def schema(self) -> dict[str, Any]:
        return self._schema

    async def call(self, **kwargs: Any) -> Any:
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("httpx is required for MCPToolAdapter") from e

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._url,
                json={"tool": self._tool_name, "input": kwargs},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
