"""Tool execution for the tool_use loop.

When the LLM returns ``tool_use`` blocks, the executor dispatches each call
to the correct handler:

* **MCPServer** → JSON-RPC over Streamable HTTP (MCP 2025-03-26 protocol)
* **FunctionTool** → Direct Python function call (sync or async)
* **HTTPTool** → HTTP request via httpx
"""
from __future__ import annotations

import inspect
import json
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.types import ToolConfig, MCPServer, FunctionTool, HTTPTool

logger = logging.getLogger(__name__)

# MCP Streamable HTTP spec requires both content types in Accept.
_MCP_ACCEPT = "application/json, text/event-stream"


def _parse_sse_json(text: str) -> dict[str, Any] | None:
    """Extract the JSON-RPC result from an SSE response body.

    SSE format::

        event: message
        data: {"jsonrpc":"2.0","id":1,"result":{...}}

    We look for lines starting with ``data:`` and try to parse the first
    valid JSON object.
    """
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                continue
    return None


class ToolExecutor:
    """Executes tool calls returned by LLM tool_use responses.

    Reuses a single ``httpx.AsyncClient`` across all MCP/HTTP calls for
    connection pooling and performance.  Call :meth:`close` (or use as an
    async context manager) when the executor is no longer needed.
    """

    def __init__(self, tool_configs: list[ToolConfig]) -> None:
        from flowforge.types import MCPServer, FunctionTool, HTTPTool

        self._configs: dict[str, ToolConfig] = {}
        for tc in tool_configs:
            if isinstance(tc, MCPServer):
                name = tc.name
            elif isinstance(tc, FunctionTool):
                name = tc.name or (tc.func.__name__ if hasattr(tc.func, "__name__") else "")
            elif isinstance(tc, HTTPTool):
                name = tc.name or "http_tool"
            else:
                continue
            if name:
                self._configs[name] = tc

        # MCP session state (shared across calls to the same server URL).
        self._mcp_sessions: dict[str, str] = {}   # url → session_id
        self._mcp_initialized: set[str] = set()
        self._mcp_request_id = 0

        # Shared HTTP client — lazily created on first use.
        self._http_client: Any | None = None

    async def _get_client(self) -> Any:
        """Return the shared ``httpx.AsyncClient``, creating it on first use."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def close(self) -> None:
        """Close the shared HTTP client and release resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> "ToolExecutor":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_tool(self, name: str) -> bool:
        return name in self._configs

    async def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool by *name* and return the result as a string."""
        from flowforge.types import MCPServer, FunctionTool, HTTPTool

        config = self._configs.get(name)
        if config is None:
            return f"Error: unknown tool '{name}'"

        try:
            if isinstance(config, MCPServer):
                return await self._execute_mcp(config, name, tool_input)
            elif isinstance(config, FunctionTool):
                return await self._execute_function(config, tool_input)
            elif isinstance(config, HTTPTool):
                return await self._execute_http(config, tool_input)
            return f"Error: unsupported tool type for '{name}'"
        except Exception as e:
            logger.warning("Tool execution failed: %s(%s) → %s", name, tool_input, e)
            return f"Error executing tool '{name}': {e}"

    async def fetch_mcp_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Fetch real tool schemas from all MCP servers.

        Returns a mapping of ``tool_name → anthropic_tool_schema`` with the
        real ``input_schema`` from the server (instead of the empty stub).
        """
        from flowforge.types import MCPServer

        schemas: dict[str, dict[str, Any]] = {}
        seen_urls: set[str] = set()

        for config in self._configs.values():
            if not isinstance(config, MCPServer):
                continue
            url = config.url
            if url in seen_urls:
                continue
            seen_urls.add(url)

            if url not in self._mcp_initialized:
                await self._mcp_initialize(url)

            try:
                data = await self._mcp_rpc(url, "tools/list", {})
                tools = data.get("tools", [])
                for tool in tools:
                    name = tool.get("name", "")
                    if name and name in self._configs:
                        schemas[name] = {
                            "name": name,
                            "description": tool.get("description", ""),
                            "input_schema": tool.get("inputSchema", {
                                "type": "object", "properties": {},
                            }),
                        }
                        logger.debug("MCP tool schema fetched: %s", name)
            except Exception as e:
                logger.warning("Failed to fetch MCP tool list from %s: %s", url, e)

        return schemas

    # ------------------------------------------------------------------
    # MCP (Streamable HTTP transport)
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._mcp_request_id += 1
        return self._mcp_request_id

    def _mcp_headers(self, url: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": _MCP_ACCEPT,
        }
        session_id = self._mcp_sessions.get(url)
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        return headers

    def _parse_mcp_response(self, resp: Any) -> dict[str, Any]:
        """Parse an MCP HTTP response — JSON or SSE."""
        content_type = resp.headers.get("content-type", "")

        # Update session ID if provided.
        sid = resp.headers.get("mcp-session-id")
        if sid:
            # Store for the caller to pick up (url not known here).
            self._last_session_id = sid

        # SSE response
        if "text/event-stream" in content_type:
            data = _parse_sse_json(resp.text)
            if data is None:
                raise RuntimeError(
                    f"MCP SSE response contained no parseable JSON-RPC data. "
                    f"Body: {resp.text[:500]}"
                )
            if "error" in data:
                raise RuntimeError(f"MCP error: {data['error']}")
            return data.get("result", {})

        # Regular JSON response
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    async def _mcp_rpc(
        self, url: str, method: str, params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a single JSON-RPC request and return ``result``."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._next_id(),
        }
        self._last_session_id = None

        client = await self._get_client()
        resp = await client.post(
            url, json=payload, headers=self._mcp_headers(url),
        )
        resp.raise_for_status()
        result = self._parse_mcp_response(resp)

        # Persist session ID from response.
        if self._last_session_id:
            self._mcp_sessions[url] = self._last_session_id

        return result

    async def _mcp_initialize(self, url: str) -> None:
        """Perform the MCP initialize handshake + initialized notification."""
        headers = {
            "Content-Type": "application/json",
            "Accept": _MCP_ACCEPT,
        }

        # 1) initialize
        init_payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "flowforge", "version": "0.1.0"},
            },
            "id": self._next_id(),
        }

        client = await self._get_client()
        resp = await client.post(url, json=init_payload, headers=headers, timeout=10.0)
        resp.raise_for_status()

        sid = resp.headers.get("mcp-session-id")
        if sid:
            self._mcp_sessions[url] = sid

        # 2) notifications/initialized (no id → notification, no response expected)
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        notif_headers = dict(headers)
        if sid:
            notif_headers["Mcp-Session-Id"] = sid
        try:
            await client.post(url, json=notif, headers=notif_headers, timeout=10.0)
        except Exception:
            # Some servers don't respond to notifications — that's fine.
            pass

        self._mcp_initialized.add(url)
        logger.info(
            "MCP session initialized: %s (session=%s)",
            url, self._mcp_sessions.get(url, "none"),
        )

    async def _execute_mcp(
        self, config: MCPServer, tool_name: str, tool_input: dict[str, Any],
    ) -> str:
        url = config.url

        if url not in self._mcp_initialized:
            await self._mcp_initialize(url)

        result = await self._mcp_rpc(url, "tools/call", {
            "name": tool_name,
            "arguments": tool_input,
        })

        # Extract text from MCP content blocks.
        content = result.get("content", [])
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    texts.append(f"[Image: {block.get('mimeType', 'image')}]")
                else:
                    texts.append(json.dumps(block, ensure_ascii=False))
            else:
                texts.append(str(block))
        return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------
    # FunctionTool
    # ------------------------------------------------------------------

    async def _execute_function(self, config: FunctionTool, tool_input: dict[str, Any]) -> str:
        import asyncio

        func = config.func
        if inspect.iscoroutinefunction(func):
            result = await func(**tool_input)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: func(**tool_input))

        if isinstance(result, str):
            return result
        return json.dumps(result, default=str, ensure_ascii=False)

    # ------------------------------------------------------------------
    # HTTPTool
    # ------------------------------------------------------------------

    async def _execute_http(self, config: HTTPTool, tool_input: dict[str, Any]) -> str:
        client = await self._get_client()
        if config.method.upper() == "GET":
            resp = await client.get(
                config.url, params=tool_input,
                headers=config.headers, timeout=30.0,
            )
        else:
            resp = await client.request(
                config.method.upper(), config.url,
                json=tool_input, headers=config.headers, timeout=30.0,
            )
        resp.raise_for_status()
        return resp.text
