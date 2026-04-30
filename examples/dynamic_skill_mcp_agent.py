"""Dynamic generator example with Agent Skills, Claude Skills, and MCP setup.

This example starts with no static user flow. The dynamic generator is asked
to create the missing flow that:

1. uses a local Agent Skill for MCP orchestration guidance,
2. optionally uses Anthropic's native ``pptx`` Claude Skill, and
3. starts/registers a declared MCP server before using its tools.

Profiles:

    python examples/dynamic_skill_mcp_agent.py playwright https://example.com
    python examples/dynamic_skill_mcp_agent.py figma "https://www.figma.com/design/..."

The Playwright profile uses the official ``@playwright/mcp`` package in HTTP
mode. The Figma profile points at Figma's remote MCP endpoint by default; set
``FIGMA_MCP_AUTHORIZATION`` if your environment uses a bearer token style
gateway, or override the URL with ``FLOWFORGE_FIGMA_MCP_URL``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from flowforge import DynamicRunOptions, FlowForge, flow, global_config, step, task
from flowforge.types import AgentSkill, ClaudeSkill, DependencyPolicy, LLMConfig


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "_artifacts" / "dynamic_skill_mcp"
PROJECT_ROOT = Path(os.getenv("FLOWFORGE_DYNAMIC_MCP_ROOT", "~/test")).expanduser()


PLAYWRIGHT_TOOLS = [
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_evaluate",
]

FIGMA_TOOLS = [
    "get_design_context",
    "get_variable_defs",
    "get_metadata",
    "get_screenshot",
    "create_design_system_rules",
    "whoami",
]


@flow(
    name="mcp_brief",
    prompt=(
        "Register declared MCP tool names when possible and write a compact "
        "markdown brief for the requested target."
    ),
)
class McpBriefFlow:
    @task(name="write_brief", prompt="Create a compact MCP setup/result brief")
    class WriteBriefTask:
        @step(
            order=1,
            prompt="Register MCP tools if possible, then write a markdown brief.",
            tools=["mcp_register_server", "markdown_write"],
            timeout_seconds=60,
        )
        async def write(ctx):
            raw = ctx.input
            if hasattr(raw, "model_dump"):
                raw = raw.model_dump()
            params = raw if isinstance(raw, dict) else {}
            profile = params.get("profile", "playwright")
            target = params.get("target", "")
            required = params.get("required_tools", [])
            mcp_tools = [
                name for name in required
                if name not in {
                    "mcp_start_server",
                    "mcp_register_server",
                    "markdown_write",
                    "pptx",
                }
            ]

            registered = {"ok": False, "registered_tools": []}
            if mcp_tools:
                registered = await ctx.call_tool(
                    "mcp_register_server",
                    server_name=profile,
                    tool_names=",".join(mcp_tools),
                )

            content = (
                "# Dynamic Skill + MCP brief\n\n"
                f"- profile: {profile}\n"
                f"- target: {target}\n"
                f"- requested_tools: {', '.join(required)}\n"
                f"- registered_ok: {registered.get('ok')}\n"
                f"- registered_tools: {', '.join(registered.get('registered_tools', []))}\n"
                "- note: This stable example flow avoids spending a long run "
                "on dynamic code generation while still exercising MCP registration "
                "and markdown output plumbing.\n"
            )
            written = await ctx.call_tool(
                "markdown_write",
                path=f"reports/{profile}_mcp_brief.md",
                content=content,
            )
            if not written.get("ok"):
                raise RuntimeError(written.get("error", "markdown_write failed"))

            return {
                "profile": profile,
                "target": target,
                "registered_tools": registered.get("registered_tools", []),
                "artifact_path": written.get("path"),
                "notes": [
                    registered.get("error")
                    or "MCP registration attempted"
                ],
            }


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


def _write_mcp_skill() -> Path:
    skill_dir = ARTIFACT_DIR / "skills" / "mcp-orchestration"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: mcp-orchestration
description: Start, register, and use MCP servers with compact tool context.
---

# MCP Orchestration

Use this skill when a FlowForge dynamic flow must connect to an MCP server
before using the server's tools.

## Workflow

1. Read the request's MCP profile and required tool names.
2. If the server has a command, call `mcp_start_server` once.
3. Call `mcp_register_server` with the known tool names.
4. Put future MCP tool names in `tools=["tool_name"]` annotations.
5. Use LLM-mediated MCP calls with `<tool_name>` references only for the
   focused action needed by the current step.
6. Prefer short summaries over returning full page or design dumps.

## Token Discipline

- Ask MCP tools for the smallest useful target: a URL, selected node, frame,
  or specific browser action.
- Save long artifacts to files with built-in file/document tools and return
  paths plus compact notes.
""",
        encoding="utf-8",
    )
    return skill_dir


def _dynamic_options(profile: str) -> DynamicRunOptions:
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    if profile == "figma":
        figma_url = os.getenv("FLOWFORGE_FIGMA_MCP_URL", "https://mcp.figma.com/mcp")
        auth = os.getenv("FIGMA_MCP_AUTHORIZATION", "").strip()
        return DynamicRunOptions.for_figma_mcp(
            project_root=str(PROJECT_ROOT),
            url=figma_url,
            authorization=auth,
            generated_dir="examples/_artifacts/dynamic_skill_mcp/generated/figma",
            persist_generated=os.getenv("FLOWFORGE_DYNAMIC_MCP_PERSIST", "0") == "1",
            auto_load_generated=False,
            allow_codegen_tool_use=False,
            project_context_max_chars=1500,
            dependency_policy=DependencyPolicy(allow_install=False),
        )

    return DynamicRunOptions.for_playwright_mcp(
        project_root=str(PROJECT_ROOT),
        port=8931,
        tools=PLAYWRIGHT_TOOLS,
        generated_dir="examples/_artifacts/dynamic_skill_mcp/generated/playwright",
        persist_generated=os.getenv("FLOWFORGE_DYNAMIC_MCP_PERSIST", "0") == "1",
        auto_load_generated=False,
        allow_codegen_tool_use=False,
        mcp_start_timeout_seconds=5,
        project_context_max_chars=1500,
        dependency_policy=DependencyPolicy(
            allow_install=True,
            allowed_managers=["npm", "pnpm", "yarn"],
        ),
    )


def _build_agent(skill_dir: Path, llm_config: LLMConfig) -> type:
    @global_config(
        prompt=(
            "You are a token-efficient dynamic FlowForge agent. A compact "
            "static MCP brief flow exists for the common demo path. "
            "Use <mcp-orchestration> for MCP setup, register MCP server tools "
            "before calling them, and keep outputs compact. If the user asks "
            "for slides, use the <pptx> Claude Skill rather than generating "
            "raw slide text only."
        ),
        llm_config=llm_config,
        tools=[
            AgentSkill(
                path=str(skill_dir),
                name="mcp-orchestration",
                description="MCP setup and token-efficient tool sequencing.",
            ),
            ClaudeSkill(
                name="pptx",
                description="Anthropic native PowerPoint creation skill.",
            ),
        ],
        dynamic_flow=True,
        include_builtin_tools=True,
    )
    class DynamicSkillMcpAgent:
        McpBriefFlow = McpBriefFlow

    return DynamicSkillMcpAgent


def _build_request(profile: str, target: str, make_deck: bool) -> dict[str, Any]:
    if profile == "figma":
        mcp_tools = ["get_design_context", "get_variable_defs", "get_metadata"]
        action = (
            "Register the Figma MCP server, fetch only the design context "
            "needed for the supplied Figma link or selected node, and write a "
            "compact implementation brief to markdown."
        )
    else:
        mcp_tools = ["browser_navigate", "browser_snapshot"]
        action = (
            "Start and register the Playwright MCP server, navigate to the "
            "target URL, capture a concise accessibility snapshot, and write "
            "a compact QA/implementation brief to markdown."
        )

    required_tools = [
        "mcp_register_server",
        *mcp_tools,
        "markdown_write",
    ]
    if profile == "playwright":
        required_tools.insert(0, "mcp_start_server")
    if make_deck:
        required_tools.append("pptx")

    return {
        "request": (
            f"{action} Use <mcp-orchestration>. "
            "Declare every intended tool in annotation tools=[...] and use "
            "MCP tools through <tool_name> references after registration."
        ),
        "profile": profile,
        "target": target,
        "project_root": str(PROJECT_ROOT),
        "required_tools": required_tools,
        "expected_output": [
            "profile",
            "target",
            "registered_tools",
            "artifact_path",
            "notes",
        ],
        "make_deck": make_deck,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=["playwright", "figma"])
    parser.add_argument("target")
    parser.add_argument("--deck", action="store_true", help="also ask for a PPTX summary")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    skill_dir = _write_mcp_skill()
    options = _dynamic_options(args.profile)
    agent = _build_agent(skill_dir, _llm_config_from_env())
    engine = FlowForge.compile(agent, dynamic_options=options)

    request = _build_request(args.profile, args.target, args.deck)
    print("Dynamic Skill + MCP agent")
    print(f"  Profile: {args.profile}")
    print(f"  Target: {args.target}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Required tools: {', '.join(request['required_tools'])}")
    print()

    result, trace = await engine.run_traced(request, route="mcp_brief")
    print("Result:")
    print(result)
    print()
    print("Executed nodes:")
    for node in trace.nodes:
        if node.succeeded:
            print(f"  - {node.node_id}")


if __name__ == "__main__":
    asyncio.run(main())
