"""Output sink adapters for FlowForge results."""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any


class OutputAdapter(abc.ABC):
    """Abstract base for output sinks."""

    @abc.abstractmethod
    async def write(self, result: Any, metadata: dict[str, Any] | None = None) -> None:
        """Write the result to the sink."""


class FileOutputAdapter(OutputAdapter):
    """Write results to a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    async def write(self, result: Any, metadata: dict[str, Any] | None = None) -> None:
        data: dict[str, Any] = {"result": result}
        if metadata:
            data["metadata"] = metadata
        if hasattr(result, "model_dump"):
            data["result"] = result.model_dump()
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


class WebhookOutputAdapter(OutputAdapter):
    """POST results to a webhook URL."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._url = url
        self._headers = headers or {}

    async def write(self, result: Any, metadata: dict[str, Any] | None = None) -> None:
        import httpx

        payload: dict[str, Any] = {}
        if hasattr(result, "model_dump"):
            payload["result"] = result.model_dump()
        else:
            payload["result"] = result
        if metadata:
            payload["metadata"] = metadata

        async with httpx.AsyncClient() as client:
            await client.post(self._url, json=payload, headers=self._headers, timeout=30.0)


class PrintOutputAdapter(OutputAdapter):
    """Print results to stdout (useful for testing/CLI)."""

    async def write(self, result: Any, metadata: dict[str, Any] | None = None) -> None:
        if hasattr(result, "model_dump"):
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        else:
            print(result)
