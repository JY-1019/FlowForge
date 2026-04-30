"""Dynamic generator example for auto-started Playwright MCP tools.

Run:

    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/dynamic_playwright_auto_mcp_generator.py https://example.com

This is the dynamic counterpart to ``static_playwright_mcp_tool_probe.py``.
The file does not define the browser-inspection flow by hand.  Instead it:

1. Compiles an agent with only a tiny baseline flow.
2. Declares Playwright MCP startup details in ``DynamicRunOptions``.
3. Reuses a manifest-loaded generated flow when one already exists.
4. Otherwise asks ``DynamicFlowGenerator`` to synthesize the missing flow and
   persist it to ``manifest.json`` plus ``generated/flows/*.py``.
5. Runs a small harness check after generation to confirm the generator chose
   ``mcp_register_server``, ``browser_navigate``, ``browser_snapshot``, and
   ``markdown_write`` on its own.
6. Injects the generated flow into the running engine and executes it.

The important behavior is that the generated flow should call
``mcp_register_server`` first.  FlowForge's built-in registration tool will
auto-start the declared Playwright MCP command if the server is not already
listening, then register the requested MCP tool names for later steps.
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

from flowforge import DynamicRunOptions, FlowForge, flow, global_config, step, task
from flowforge.dynamic.generator import DynamicFlowGenerator
from flowforge.dynamic.manifest import (
    load_manifest,
    manifest_path,
    persist_flow_code,
    resolve_project_root,
)
from flowforge.types import DependencyPolicy, LLMConfig


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "_artifacts" / "dynamic_playwright_auto_mcp_generator"
PROJECT_ROOT = ARTIFACT_DIR / "project"
GENERATED_FLOW_NAME = "generated_playwright_auto_mcp_probe"
PLAYWRIGHT_TOOLS = [
    "browser_navigate",
    "browser_snapshot",
]
EXPECTED_GENERATED_TOOLS = [
    "mcp_register_server",
    "browser_navigate",
    "browser_snapshot",
    "markdown_write",
]


@flow(name="baseline", prompt="Minimal baseline flow before dynamic generation.")
class BaselineFlow:
    @task(name="noop", prompt="Return a compact baseline result.")
    class NoopTask:
        @step(order=1, prompt="Return baseline status.")
        async def run(ctx):
            return {"ok": True, "note": "baseline only"}


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
        generated_dir="generated",
        persist_generated=True,
        auto_load_generated=True,
        allow_codegen_tool_use=False,
        mcp_start_timeout_seconds=int(
            os.getenv("FLOWFORGE_PLAYWRIGHT_MCP_START_TIMEOUT", "15")
        ),
        dependency_policy=DependencyPolicy(
            allow_install=True,
            allowed_managers=["npm", "pnpm", "yarn"],
        ),
    )


def _build_agent(llm_config: LLMConfig) -> type:
    @global_config(
        prompt=(
            "You are a dynamic FlowForge agent. Generate missing flows with "
            "small, explicit steps. Runtime input is the user's raw natural "
            "language request string, not a dict or Pydantic model, so parse "
            "the target URL and goal from ctx.input. For declared MCP servers, "
            "first register the server, then use the registered MCP tools in "
            "a later step. Keep outputs compact and write any long browser "
            "notes to markdown."
        ),
        llm_config=llm_config,
        tools=[],
        dynamic_flow=True,
        include_builtin_tools=True,
    )
    class DynamicPlaywrightGeneratorAgent:
        BaselineFlow = BaselineFlow

    return DynamicPlaywrightGeneratorAgent


def _build_user_request(target_url: str, goal: str) -> str:
    return (
        f"{target_url} 페이지를 브라우저로 열어서 확인하고, "
        f"{goal} 결과를 짧은 마크다운 리포트로 저장해줘."
    )


def _tool_expectation() -> dict[str, list[str]]:
    """Internal harness expectation, not part of the user request."""
    return {"required_tools": EXPECTED_GENERATED_TOOLS}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _manifest_flow_file(options: DynamicRunOptions, flow_name: str) -> Path | None:
    manifest = load_manifest(options)
    root = resolve_project_root(options)
    for record in manifest.flows:
        if record.name == flow_name:
            return root / record.file
    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_url", nargs="?", default="https://example.com")
    parser.add_argument(
        "--goal",
        default="Summarize the visible page content and note anything useful for QA.",
    )
    parser.add_argument("--port", type=int, default=8931)
    parser.add_argument(
        "--codegen-timeout",
        type=int,
        default=int(os.getenv("FLOWFORGE_DYNAMIC_MCP_CODEGEN_TIMEOUT", "60")),
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="generate and overwrite the manifest record even if it already exists",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="leave an auto-started Playwright MCP process running after the demo",
    )
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    options = _dynamic_options(args.port)
    if args.regenerate:
        options.auto_load_generated = False
    agent = _build_agent(_llm_config_from_env())
    engine = FlowForge.compile(agent, dynamic_options=options)

    generator = DynamicFlowGenerator(
        llm_config=engine._global_meta.llm_config,
        dag=engine.dag,
        docs=engine.docs,
        tool_configs=engine._global_meta.tools,
        dynamic_options=options,
    )
    user_request = _build_user_request(args.target_url, args.goal)

    print("Dynamic Playwright Auto MCP generator")
    print(f"  Target: {args.target_url}")
    print(f"  MCP URL: {options.mcp_server_urls['playwright']}")
    print(f"  MCP command: {' '.join(options.mcp_server_commands['playwright'])}")
    print(f"  Manifest: {manifest_path(options)}")
    print(f"  User request: {user_request}")
    print()

    generated_code = ""
    generated_file = _manifest_flow_file(options, GENERATED_FLOW_NAME)
    if GENERATED_FLOW_NAME in {flow.name for flow in engine._global_meta.flows}:
        print(f"Reusing manifest-loaded flow: {GENERATED_FLOW_NAME}")
        if generated_file and generated_file.exists():
            generated_code = generated_file.read_text(encoding="utf-8")
    else:
        if generated_file and not args.regenerate:
            raise RuntimeError(
                f"{GENERATED_FLOW_NAME!r} exists in the manifest but was not "
                "loaded into the engine. Check auto_load_generated or remove "
                f"{manifest_path(options)}."
            )

        flow_meta, generated_code = await asyncio.wait_for(
            generator.generate_and_compile(
                flow_name=GENERATED_FLOW_NAME,
                flow_prompt=(
                    "Auto-start/register Playwright MCP, inspect a URL with "
                    "generated browser-tool steps, and write a markdown report. "
                    "The flow receives ctx.input as a raw natural-language "
                    "string and must parse the URL from that string."
                ),
                user_query=user_request,
            ),
            timeout=args.codegen_timeout,
        )

        tool_usage_error = generator.check_required_tool_usage(
            generated_code,
            _tool_expectation(),
        )
        if tool_usage_error:
            raise RuntimeError(tool_usage_error)

        persist_flow_code(
            flow_name=flow_meta.name,
            code=generated_code,
            options=options,
            class_name=flow_meta.cls.__name__,
        )
        engine.add_flow(flow_meta.cls)
        generated_file = _manifest_flow_file(options, GENERATED_FLOW_NAME)

    if generated_code:
        print("Generated flow source:")
        print(generated_code)
        print()
    if generated_file:
        print(f"Generated flow file: {generated_file}")
        print()

    result = await engine.run(user_request, route=GENERATED_FLOW_NAME)
    data = _jsonable(result)
    print("Result:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))

    if isinstance(data, dict) and data.get("artifact_path"):
        artifact_path = PROJECT_ROOT / str(data["artifact_path"])
        if artifact_path.exists():
            report_text = artifact_path.read_text(encoding="utf-8")
            if args.target_url not in report_text:
                raise RuntimeError(
                    "Generated flow ran, but the report does not contain the "
                    f"requested target URL {args.target_url!r}. The generated "
                    "flow probably ignored the natural-language input."
                )

    start = data.get("mcp_start") if isinstance(data, dict) else None
    if not start and isinstance(data, dict):
        start = data.get("start")
    pid = (start or {}).get("pid") if isinstance(start, dict) else None
    if pid and not args.keep_server:
        try:
            os.killpg(int(pid), signal.SIGTERM)
            print(f"\nStopped auto-started Playwright MCP process group: {pid}")
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
