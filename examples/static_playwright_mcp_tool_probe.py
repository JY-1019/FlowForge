"""Static Playwright MCP tool probe with on-demand server startup.

Run:

    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/static_playwright_mcp_tool_probe.py https://example.com

This file intentionally uses a hand-written flow.  It is a small probe for
FlowForge's built-in MCP registration tool, not a dynamic generation demo.
Unlike ``examples/playwright_agent.py``, it does not pre-register ``MCPServer``
tools that require an already-running server.  Instead:

1. ``DynamicRunOptions.for_playwright_mcp`` declares how to start Playwright MCP.
2. The agent starts with only built-in dynamic tools.
3. ``mcp_register_server`` auto-starts the declared server when needed.
4. The next LLM call receives the freshly registered Playwright tools and uses
   ``<browser_navigate>`` / ``<browser_snapshot>`` through FlowForge tool use.

The default command is:

    npx -y @playwright/mcp@latest --port <port>

Set ``FLOWFORGE_PLAYWRIGHT_MCP_COMMAND`` to override it, for example:

    FLOWFORGE_PLAYWRIGHT_MCP_COMMAND="npx -y @playwright/mcp@latest --port 8931"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from flowforge import DynamicRunOptions, FlowForge, flow, global_config, step, task
from flowforge.types import DependencyPolicy, LLMConfig


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "_artifacts" / "static_playwright_mcp_tool_probe"
PROJECT_ROOT = ARTIFACT_DIR / "project"

PLAYWRIGHT_TOOLS = [
    "browser_navigate",
    "browser_snapshot",
]


class BrowserProbeInput(BaseModel):
    target_url: str = Field(description="URL to inspect with Playwright MCP")
    goal: str = Field(
        default="Summarize the page and call out the most important visible content.",
        description="Natural-language inspection goal",
    )


class BrowserProbeResult(BaseModel):
    target_url: str
    mcp_url: str
    registered_tools: list[str] = Field(default_factory=list)
    snapshot_summary: str = ""
    artifact_path: str = ""
    mcp_start: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _llm_config_from_env() -> LLMConfig:
    provider = os.getenv("FLOWFORGE_PROVIDER", "").strip().lower()
    model = os.getenv("FLOWFORGE_MODEL", "").strip()
    max_tokens = int(os.getenv("FLOWFORGE_MAX_TOKENS", "4096"))
    kwargs = {"temperature": 0.2, "max_tokens": max_tokens}

    if provider == "openai":
        return LLMConfig.for_openai(model=model or "gpt-4o", **kwargs)
    if provider == "google":
        return LLMConfig.for_gemini(model=model or "gemini-2.0-flash", **kwargs)
    return LLMConfig.for_claude(model=model or "claude-sonnet-4-6", **kwargs)


def _dynamic_options(port: int) -> DynamicRunOptions:
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    command_env = os.getenv("FLOWFORGE_PLAYWRIGHT_MCP_COMMAND", "").strip()
    command = shlex.split(command_env) if command_env else None
    return DynamicRunOptions.for_playwright_mcp(
        project_root=str(PROJECT_ROOT),
        port=port,
        command=command,
        tools=PLAYWRIGHT_TOOLS,
        generated_dir="examples/_artifacts/static_playwright_mcp_tool_probe/generated",
        persist_generated=False,
        auto_load_generated=False,
        mcp_start_timeout_seconds=int(
            os.getenv("FLOWFORGE_PLAYWRIGHT_MCP_START_TIMEOUT", "15")
        ),
        dependency_policy=DependencyPolicy(
            allow_install=True,
            allowed_managers=["npm", "pnpm", "yarn"],
        ),
    )


@global_config(
    prompt=(
        "You are a dynamic browser-inspection agent. Do not assume a "
        "Playwright MCP server is already running. Register the declared MCP "
        "server first; registration may start it. After registration, use the "
        "registered browser tools through FlowForge tool use."
    ),
    llm_config=_llm_config_from_env(),
    tools=[],
    dynamic_flow=True,
    include_builtin_tools=True,
)
class DynamicPlaywrightAutoMcpAgent:
    @flow(
        name="auto_playwright_probe",
        prompt=(
            "Start/register Playwright MCP on demand, inspect a URL with "
            "registered browser tools, and write a compact markdown report."
        ),
        input_schema=BrowserProbeInput,
        output_schema=BrowserProbeResult,
    )
    class AutoPlaywrightProbeFlow:
        @task(name="inspect", prompt="Inspect a page through auto-started MCP")
        class InspectTask:
            @step(
                order=1,
                prompt=(
                    "Register the declared Playwright MCP server. If the HTTP "
                    "endpoint is not reachable, mcp_register_server should "
                    "start the configured command before registering tools."
                ),
                input_schema=BrowserProbeInput,
                tools=["mcp_register_server"],
                timeout_seconds=90,
            )
            async def register_playwright(ctx):
                data = ctx.input
                target_url = data.target_url.strip()
                if not target_url.startswith(("http://", "https://")):
                    target_url = f"https://{target_url}"

                registered = await ctx.call_tool(
                    "mcp_register_server",
                    server_name="playwright",
                    tool_names=",".join(PLAYWRIGHT_TOOLS),
                    auto_start=True,
                )
                if not registered.get("ok"):
                    raise RuntimeError(
                        registered.get("stderr")
                        or registered.get("error")
                        or "Playwright MCP registration failed"
                    )

                return {
                    "target_url": target_url,
                    "goal": data.goal,
                    "mcp_url": registered.get("url", ""),
                    "registered_tools": registered.get("registered_tools", []),
                    "mcp_start": registered.get("start"),
                }

            @step(
                order=2,
                prompt=(
                    "Use the registered Playwright MCP tools to navigate and "
                    "capture a page snapshot. Then write a markdown artifact."
                ),
                output_schema=BrowserProbeResult,
                tools=[
                    "browser_navigate",
                    "browser_snapshot",
                    "markdown_write",
                ],
                timeout_seconds=120,
            )
            async def browse_and_report(ctx):
                params = _as_dict(ctx.input)
                target_url = params.get("target_url", "https://example.com")
                goal = params.get("goal", "")

                snapshot = await ctx.call_llm(
                    f"""
                    Inspect {target_url}.

                    Goal: {goal}

                    Required sequence:
                    1. Use <browser_navigate> to open the target URL.
                    2. Use <browser_snapshot> to read the visible page state.
                    3. Return a concise summary with important visible text,
                       page title if available, and any notable issues.
                    """
                )
                summary = str(snapshot).strip()
                content = (
                    "# Dynamic Playwright Auto MCP Report\n\n"
                    f"- target_url: {target_url}\n"
                    f"- mcp_url: {params.get('mcp_url', '')}\n"
                    f"- registered_tools: {', '.join(params.get('registered_tools', []))}\n\n"
                    "## Snapshot Summary\n\n"
                    f"{summary}\n"
                )
                written = await ctx.call_tool(
                    "markdown_write",
                    path="reports/playwright_auto_mcp_report.md",
                    content=content,
                )
                if not written.get("ok"):
                    raise RuntimeError(written.get("error", "markdown_write failed"))

                return BrowserProbeResult(
                    target_url=target_url,
                    mcp_url=str(params.get("mcp_url", "")),
                    registered_tools=list(params.get("registered_tools", [])),
                    snapshot_summary=summary,
                    artifact_path=str(written.get("path", "")),
                    mcp_start=params.get("mcp_start"),
                    notes=[
                        "Playwright MCP was registered at runtime.",
                        "If mcp_start is not null, FlowForge started the server.",
                    ],
                )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_url", nargs="?", default="https://example.com")
    parser.add_argument(
        "--goal",
        default="Summarize the visible page content and report anything useful for QA.",
    )
    parser.add_argument("--port", type=int, default=8931)
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="leave an auto-started Playwright MCP process running after the demo",
    )
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    options = _dynamic_options(args.port)
    engine = FlowForge.compile(
        DynamicPlaywrightAutoMcpAgent,
        dynamic_options=options,
    )

    request = BrowserProbeInput(target_url=args.target_url, goal=args.goal)
    print("Static Playwright MCP tool probe")
    print(f"  Target: {request.target_url}")
    print(f"  MCP URL: {options.mcp_server_urls['playwright']}")
    print(f"  MCP command: {' '.join(options.mcp_server_commands['playwright'])}")
    print()

    result = await engine.run(request, route="auto_playwright_probe")
    data = result.model_dump() if hasattr(result, "model_dump") else _as_dict(result)
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))

    start = data.get("mcp_start") or {}
    pid = start.get("pid")
    if pid and not args.keep_server:
        try:
            os.killpg(int(pid), signal.SIGTERM)
            print(f"\nStopped auto-started Playwright MCP process group: {pid}")
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
