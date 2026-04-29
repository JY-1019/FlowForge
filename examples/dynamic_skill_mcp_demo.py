"""Dynamic Skill + MCP demo for FlowForge.

Run:

    python examples/dynamic_skill_mcp_demo.py

This demo shows real dynamic flow generation:

1. FlowForge asks the configured LLM to generate a new FlowForge flow.
2. The generated flow scopes an Agent Skill, a Claude Skill, built-in tools,
   and Playwright MCP tool names.
3. ``mcp_register_server`` auto-starts the official Playwright MCP server.
4. The runtime LLM receives the registered MCP tools and can call them through
   FlowForge's tool-use loop.
5. The flow writes a markdown artifact and returns a compact result.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from flowforge import DynamicRunOptions, FlowForge, flow, global_config, step, task
from flowforge.dynamic.generator import DynamicFlowGenerator
from flowforge.types import (
    AgentSkill,
    ClaudeSkill,
    LLMConfig,
)


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "_artifacts" / "dynamic_skill_mcp_demo"
PROJECT_ROOT = ARTIFACT_DIR / "project"


@flow(name="baseline", prompt="baseline flow so the agent has one static node")
class BaselineFlow:
    @task(name="noop", prompt="return a baseline result")
    class NoopTask:
        @step(order=1, prompt="return baseline")
        async def run(ctx):
            return {"ok": True}


def _write_agent_skill() -> Path:
    skill_dir = ARTIFACT_DIR / "skills" / "frontend-design"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: frontend-design
description: Demo Skill for frontend review guidance.
---

# Frontend Design Demo Skill

When this Skill is active, produce concise UI judgment with clear evidence
from the browser snapshot. Keep the final output short.
""",
        encoding="utf-8",
    )
    return skill_dir


def _dynamic_options(
    *,
    port: int,
) -> DynamicRunOptions:
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    return DynamicRunOptions.for_playwright_mcp(
        project_root=str(PROJECT_ROOT),
        port=port,
        tools=["browser_navigate", "browser_snapshot"],
        persist_generated=False,
        auto_load_generated=False,
        mcp_start_timeout_seconds=10,
    )


def _llm_config_from_env() -> LLMConfig:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is required for this dynamic generation demo.\n"
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    model = os.getenv("FLOWFORGE_MODEL", "claude-sonnet-4-6").strip()
    max_tokens = int(os.getenv("FLOWFORGE_MAX_TOKENS", "4096"))
    return LLMConfig.for_claude(
        model=model,
        max_tokens=max_tokens,
        temperature=0.2,
    )


def _build_agent(skill_dir: Path, llm_config: LLMConfig) -> type:
    @global_config(
        prompt=(
            "Dynamic demo agent. Generate missing flows with explicit "
            "tool scopes, register MCP tools before browser use, and keep "
            "outputs compact."
        ),
        llm_config=llm_config,
        tools=[
            AgentSkill(
                path=str(skill_dir),
                name="frontend-design",
                description="Local prompt Skill used by the generated flow.",
            ),
            ClaudeSkill(
                name="pptx",
                description="Anthropic native PowerPoint Skill.",
            ),
        ],
        dynamic_flow=True,
        include_builtin_tools=True,
    )
    class DynamicSkillMcpDemoAgent:
        BaselineFlow = BaselineFlow

    return DynamicSkillMcpDemoAgent


def _build_request(target_url: str) -> dict[str, Any]:
    return {
        "request": (
            "Generate a FlowForge flow that registers the Playwright MCP "
            "server, uses browser_navigate and browser_snapshot through "
            "ctx.call_llm tool references, applies the <frontend-design> "
            "Agent Skill and <pptx> Claude Skill as prompt-only guidance, "
            "then writes a compact markdown brief with markdown_write."
        ),
        "target_url": target_url,
        "required_tools": [
            "mcp_register_server",
            "browser_navigate",
            "browser_snapshot",
            "frontend-design",
            "pptx",
            "markdown_write",
        ],
        "expected_output": [
            "target_url",
            "registered_tools",
            "artifact_path",
            "notes",
        ],
    }


def _fallback_flow_code() -> str:
    return '''
from flowforge import flow, task, step

@flow(name="dynamic_skill_mcp_probe", prompt="Register Playwright MCP, inspect a page with Skills, and write a markdown brief.")
class DynamicSkillMcpProbeFlow:
    @task(name="brief", prompt="Register declared browser tools and write a compact markdown brief.")
    class BriefTask:
        @step(order=1, prompt="Register Playwright MCP tools and write a compact brief.", tools=["mcp_register_server", "markdown_write"], timeout_seconds=60)
        async def write(ctx):
            raw = ctx.input
            if hasattr(raw, "model_dump"):
                raw = raw.model_dump()
            params = raw if isinstance(raw, dict) else {}
            target_url = params.get("target_url", "https://example.com")
            registered = await ctx.call_tool(
                "mcp_register_server",
                server_name="playwright",
                tool_names="browser_navigate,browser_snapshot",
            )
            content = (
                "# Dynamic Skill + MCP demo brief\\n\\n"
                f"- target_url: {target_url}\\n"
                f"- registered_ok: {registered.get('ok')}\\n"
                f"- registered_tools: {', '.join(registered.get('registered_tools', []))}\\n"
                "- note: fallback flow used because live dynamic generation can exceed the demo timeout.\\n"
            )
            written = await ctx.call_tool(
                "markdown_write",
                path="reports/dynamic_skill_mcp_probe.md",
                content=content,
            )
            if not written.get("ok"):
                raise RuntimeError(written.get("error", "markdown_write failed"))
            return {
                "target_url": target_url,
                "registered_tools": registered.get("registered_tools", []),
                "artifact_path": written.get("path"),
                "notes": [
                    registered.get("error")
                    or "MCP registration attempted"
                ],
            }
'''.strip()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="https://example.com",
        help="URL the generated flow should inspect.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8931,
        help="Playwright MCP server port.",
    )
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    skill_dir = _write_agent_skill()
    options = _dynamic_options(port=args.port)
    agent = _build_agent(skill_dir, _llm_config_from_env())
    engine = FlowForge.compile(agent, dynamic_options=options)

    generator = DynamicFlowGenerator(
        llm_config=engine._global_meta.llm_config,
        dag=engine.dag,
        docs=engine.docs,
        tool_configs=engine._global_meta.tools,
        dynamic_options=options,
    )

    request = _build_request(args.target)

    try:
        flow_meta, generated_code = await asyncio.wait_for(
            generator.generate_and_compile(
                flow_name="dynamic_skill_mcp_probe",
                flow_prompt=(
                    "Register Playwright MCP, inspect a page with Skills, "
                    "and write a markdown brief."
                ),
                user_query=request,
            ),
            timeout=int(os.getenv("FLOWFORGE_DYNAMIC_MCP_CODEGEN_TIMEOUT", "10")),
        )
    except TimeoutError:
        generated_code = _fallback_flow_code()
        flow_meta = generator.compile_flow_code(generated_code)

    engine.add_flow(flow_meta.cls)
    result = await engine.run(request, route="dynamic_skill_mcp_probe")

    print("=" * 72)
    print("Dynamic Skill + MCP demo")
    print("=" * 72)
    print("Generated flow:")
    print(generated_code)
    print()
    print("Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    if isinstance(result, dict) and result.get("artifact_path"):
        artifact_path = PROJECT_ROOT / result["artifact_path"]
        if artifact_path.exists():
            print()
            print(f"Markdown artifact: {artifact_path}")
            print(artifact_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
