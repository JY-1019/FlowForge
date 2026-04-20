"""HTTP API tool adapter."""
from __future__ import annotations

from typing import Any

from flowforge.tools.base import ToolAdapter


class HTTPToolAdapter(ToolAdapter):
    """Adapter that calls an HTTP API endpoint as a tool."""

    def __init__(
        self,
        url: str,
        method: str = "POST",
        name: str = "http_tool",
        description: str = "",
        headers: dict[str, str] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self._url = url
        self._method = method.upper()
        self._name = name
        self._description = description
        self._headers = headers or {}
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
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("httpx is required for HTTPToolAdapter") from e

        async with httpx.AsyncClient() as client:
            if self._method == "GET":
                response = await client.get(self._url, params=kwargs, headers=self._headers, timeout=30.0)
            else:
                response = await client.request(
                    self._method,
                    self._url,
                    json=kwargs,
                    headers=self._headers,
                    timeout=30.0,
                )
            response.raise_for_status()
            return response.json()
