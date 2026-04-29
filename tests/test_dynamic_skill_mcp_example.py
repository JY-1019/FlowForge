"""Executable examples for dynamic Skills + built-in MCP provisioning.

These tests keep the "dynamic Skill + Playwright MCP" workflow honest without
depending on Anthropic tokens, npm, or a real browser.  A tiny local HTTP
server mimics the Streamable HTTP MCP protocol enough to prove that
``mcp_register_server`` can auto-start a declared server and that registered
MCP tool names are usable by FlowForge's tool executor.
"""
from __future__ import annotations

import os
import signal
import socket
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from flowforge import FlowForge, flow, global_config, step, task
from flowforge.dynamic.generator import DynamicFlowGenerator
from flowforge.execution.context import GlobalContext, FlowContext, StepContext, TaskContext
from flowforge.execution.tool_executor import ToolExecutor
from flowforge.tools.builtin import create_builtin_tool_pack
from flowforge.tools.registry import ToolRegistry
from flowforge.types import AgentSkill, ClaudeSkill, DynamicRunOptions, LLMConfig, MCPServer


@flow(name="baseline", prompt="baseline flow for dynamic generator tests")
class _BaselineFlow:
    @task(name="noop", prompt="return a deterministic baseline result")
    class _NoopTask:
        @step(order=1, prompt="return baseline")
        async def run(ctx):
            return {"ok": True}


@global_config(prompt="baseline agent for dynamic generator tests")
class _BaselineAgent:
    BaselineFlow = _BaselineFlow


_FAKE_MCP_SERVER = r'''
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


TOOLS = [
    {
        "name": "browser_navigate",
        "description": "Fake Playwright navigate tool.",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "browser_snapshot",
        "description": "Fake Playwright snapshot tool.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(length) or b"{}")
        method = payload.get("method")
        request_id = payload.get("id")

        if method == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if method == "initialize":
            body = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-playwright", "version": "0.1"},
                },
            }
        elif method == "tools/list":
            body = {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            params = payload.get("params", {})
            result = {
                "called": params.get("name"),
                "arguments": params.get("arguments", {}),
            }
            body = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, sort_keys=True),
                        }
                    ]
                },
            }
        else:
            body = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"unknown method: {method}"},
            }

        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Mcp-Session-Id", "fake-session")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
'''


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_fake_mcp_server(tmp_path: Path) -> Path:
    server_path = tmp_path / "fake_playwright_mcp.py"
    server_path.write_text(_FAKE_MCP_SERVER, encoding="utf-8")
    return server_path


def _dynamic_options(tmp_path: Path, server_path: Path, port: int) -> DynamicRunOptions:
    return DynamicRunOptions.for_playwright_mcp(
        project_root=str(tmp_path),
        port=port,
        command=[sys.executable, str(server_path), "--port", str(port)],
        tools=["browser_navigate", "browser_snapshot"],
        persist_generated=False,
        auto_load_generated=False,
        mcp_start_timeout_seconds=5,
    )


def _skill_tools() -> list[AgentSkill | ClaudeSkill]:
    skill_dir = (
        Path(__file__).resolve().parents[1]
        / "flowforge"
        / "skills"
        / "anthropic"
        / "frontend-design"
    )
    return [
        AgentSkill(
            path=str(skill_dir),
            name="frontend-design",
            description="Local frontend design Skill instructions.",
        ),
        ClaudeSkill(
            name="pptx",
            description="Anthropic native PowerPoint Skill.",
        ),
    ]


@pytest.mark.asyncio
async def test_dynamic_codegen_accepts_skills_builtins_and_declared_mcp_tools(tmp_path):
    """Dynamic code can combine Claude/Agent Skills, built-ins, and MCP tools."""
    port = _free_port()
    options = _dynamic_options(tmp_path, _write_fake_mcp_server(tmp_path), port)
    tools = [*create_builtin_tool_pack(options), *_skill_tools()]
    generator = DynamicFlowGenerator(
        llm_config=LLMConfig.for_claude(model="test"),
        dag=FlowForge.compile(_BaselineAgent).dag,
        tool_configs=tools,
        dynamic_options=options,
    )

    generated_code = textwrap.dedent(
        '''
        from flowforge import flow, task, step

        @flow(name="dynamic_skill_mcp_probe", prompt="Register Playwright MCP, use prompt-only Skills, and write a compact brief.")
        class DynamicSkillMcpProbeFlow:
            @task(name="inspect", prompt="Prepare MCP tools, inspect the target, and persist a markdown brief.")
            class InspectTask:
                @step(order=1, prompt="Register the declared Playwright MCP server and normalise target input.", tools=["mcp_register_server"])
                async def register_playwright(ctx):
                    raw = ctx.input
                    if hasattr(raw, "model_dump"):
                        raw = raw.model_dump()
                    params = raw if isinstance(raw, dict) else {}
                    target_url = params.get("target_url", "https://example.com")
                    ctx.shared_data["target_url"] = target_url
                    registered = await ctx.call_tool(
                        "mcp_register_server",
                        server_name="playwright",
                        tool_names="browser_navigate,browser_snapshot",
                    )
                    if not registered.get("ok"):
                        raise RuntimeError(registered.get("stderr", "MCP registration failed"))
                    ctx.shared_data["registered_tools"] = registered.get("registered_tools", [])
                    return {
                        "target_url": target_url,
                        "registered_tools": registered.get("registered_tools", []),
                    }

                @step(order=2, prompt="Use the registered browser tool plus Skills to create a concise implementation brief.", tools=["browser_navigate", "frontend-design", "pptx", "markdown_write"])
                async def write_brief(ctx):
                    target_url = ctx.shared_data.get("target_url", "https://example.com")
                    notes = await ctx.call_llm(
                        "Navigate to the target URL with <browser_navigate>. "
                        "Use <frontend-design> for UI judgment and <pptx> only if a slide-style summary is useful."
                    )
                    content = str(notes).strip()
                    if not content:
                        raise RuntimeError("brief content was empty")
                    written = await ctx.call_tool(
                        "markdown_write",
                        path="reports/dynamic_skill_mcp_probe.md",
                        content=content,
                    )
                    if not written.get("ok"):
                        raise RuntimeError(written.get("error", "markdown_write failed"))
                    return {
                        "target_url": target_url,
                        "registered_tools": ctx.shared_data.get("registered_tools", []),
                        "artifact_path": written.get("path"),
                    }
        '''
    ).strip()

    with patch(
        "flowforge.execution.llm.call_llm_api",
        new_callable=AsyncMock,
    ) as mock_api:
        mock_api.return_value = generated_code
        meta, code = await generator.generate_and_compile(
            flow_name="dynamic_skill_mcp_probe",
            flow_prompt="Register Playwright MCP, use Skills, and write a brief.",
            user_query={
                "target_url": "https://example.com",
                "required_tools": [
                    "mcp_register_server",
                    "browser_navigate",
                    "frontend-design",
                    "pptx",
                    "markdown_write",
                ],
            },
        )

    assert meta.name == "dynamic_skill_mcp_probe"
    assert code == generated_code
    assert generator.check_tool_ref_validity(generated_code) is None
    assert generator.check_required_tool_usage(
        generated_code,
        {
            "required_tools": [
                "mcp_register_server",
                "browser_navigate",
                "frontend-design",
                "pptx",
                "markdown_write",
            ]
        },
    ) is None


@pytest.mark.asyncio
async def test_mcp_register_server_auto_starts_and_registered_tool_executes(tmp_path):
    """The built-in MCP registration tool starts a declared server on demand."""
    port = _free_port()
    options = _dynamic_options(tmp_path, _write_fake_mcp_server(tmp_path), port)
    tools = create_builtin_tool_pack(options)

    global_ctx = GlobalContext(
        llm_config=LLMConfig(model="test"),
        global_prompt="dynamic MCP test",
        tool_registry=ToolRegistry(),
        global_tools=tools,
        dynamic_options=options,
    )
    flow_ctx = FlowContext(
        global_ctx=global_ctx,
        flow_name="dynamic_skill_mcp_probe",
        flow_prompt="probe dynamic MCP registration",
    )
    task_ctx = TaskContext(
        flow_ctx=flow_ctx,
        task_name="inspect",
        task_prompt="register and call MCP",
    )
    step_ctx = StepContext(
        task_ctx=task_ctx,
        step_prompt="register Playwright MCP",
    )

    registration = None
    executor = None
    try:
        registration = await step_ctx.call_tool(
            "mcp_register_server",
            server_name="playwright",
        )

        assert registration["ok"] is True
        assert registration["registered_tools"] == [
            "browser_navigate",
            "browser_snapshot",
        ]
        assert registration["start"]["ok"] is True
        assert registration["start"]["already_running"] is False

        mcp_tools = [
            tool
            for tool in step_ctx.merged_tools
            if isinstance(tool, MCPServer)
        ]
        assert [tool.name for tool in mcp_tools] == [
            "browser_navigate",
            "browser_snapshot",
        ]

        executor = ToolExecutor(mcp_tools)
        schemas = await executor.fetch_mcp_tool_schemas()
        assert schemas["browser_navigate"]["input_schema"]["required"] == ["url"]

        result = await executor.execute(
            "browser_navigate",
            {"url": "https://example.com"},
        )
        assert '"called": "browser_navigate"' in result
        assert '"url": "https://example.com"' in result
    finally:
        if executor is not None:
            await executor.close()
        pid = None
        if registration:
            pid = (registration.get("start") or {}).get("pid")
        if pid:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
