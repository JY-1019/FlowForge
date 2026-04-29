"""Bare agent — zero static flows, zero tools, dynamic flow reuse enabled.

This is the simplest possible dynamic flow example.  The agent has:
- No ``@flow`` classes
- No ``FunctionTool`` registrations
- No external API dependencies

When no persisted generated flow exists, the planner reports a gap and the
``_dynamic_generator`` meta-flow creates the required flow.  On later runs,
the manifest-loaded generated flow is reused so the example remains stable
even when the planning LLM is temporarily unavailable.

Run:

    python examples/dynamic_bare_agent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from flowforge import DynamicRunOptions, FlowForge, global_config
from flowforge.schema.dag import NodeType
from flowforge.types import LLMConfig


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "_artifacts" / "dynamic_bare"
GENERATED_DIR = "examples/_artifacts/dynamic_bare/generated"


def _llm_config_from_env() -> LLMConfig:
    provider = os.getenv("FLOWFORGE_PROVIDER", "").strip().lower()
    model = os.getenv("FLOWFORGE_MODEL", "").strip()

    if provider == "openai":
        return LLMConfig.for_openai(model=model or "gpt-4o")
    if provider == "google":
        return LLMConfig.for_gemini(model=model or "gemini-2.0-flash")
    return LLMConfig.for_claude(model=model or "claude-sonnet-4-6")


def _dynamic_options() -> DynamicRunOptions:
    return DynamicRunOptions(
        project_root=str(ROOT_DIR.parent),
        generated_dir=GENERATED_DIR,
        auto_load_generated=True,
        persist_generated=True,
        include_builtin_tools=False,
        allow_codegen_tool_use=False,
    )


def _user_flow_names(engine: Any) -> list[str]:
    return [
        node.name
        for node in engine.dag.get_children("global")
        if node.type == NodeType.FLOW and not node.name.startswith("_")
    ]


# ---------------------------------------------------------------------------
# Agent — completely empty: no flows, no tools
# ---------------------------------------------------------------------------

@global_config(
    prompt=(
        "너는 범용 AI 에이전트다. "
        "사전 정의된 Flow나 Tool이 하나도 없다. "
        "사용자의 요청을 분석하여 필요한 Flow를 직접 생성하고 실행해야 한다. "
        "생성되는 Flow는 ctx.call_llm()만 사용하여 동작해야 한다."
    ),
    llm_config=_llm_config_from_env(),
    dynamic_flow=True,
)
class BareAgent:
    """No flows, no tools. Everything is generated dynamically."""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    options = _dynamic_options()
    engine = FlowForge.compile(BareAgent, dynamic_options=options)
    await engine.generate_docs(planning_only=True)

    print("=" * 50)
    print("Bare Agent — zero static flows, zero tools")
    print("=" * 50)
    print(f"  User flows: {_user_flow_names(engine) or '(none)'}")
    print(f"  Tools: (none)")
    print()

    user_request = (
        "세계에서 가장 높은 산 Top 5를 조사해서 "
        "이름, 높이, 위치를 표로 정리해줘"
    )

    print(f"User: {user_request}\n")

    result, trace = await engine.run_traced(
        user_request,
        planning_mode="autonomous",
        dynamic_options=options,
    )

    # -- Artifacts --
    mermaid_path = ARTIFACT_DIR / "bare_agent_run.md"
    mermaid_path.write_text(engine.compare_mermaid(trace), encoding="utf-8")

    dynamic_info = engine.last_dynamic_generation or {}
    if "generated" in dynamic_info:
        for idx, gen in enumerate(dynamic_info["generated"], 1):
            if gen.get("generated_code"):
                name = gen.get("dynamic_flow", f"flow_{idx}")
                path = ARTIFACT_DIR / f"{name}.py"
                path.write_text(gen["generated_code"], encoding="utf-8")
                print(f"  Generated flow {idx}: {name}")
    elif dynamic_info.get("generated_code"):
        name = dynamic_info.get("dynamic_flow", "dynamic_flow")
        path = ARTIFACT_DIR / f"{name}.py"
        path.write_text(dynamic_info["generated_code"], encoding="utf-8")
        print(f"  Generated flow: {name}")

    final_flows = _user_flow_names(engine)
    print(f"\n  Flows after run: {final_flows or '(none)'}")
    print(f"  Mermaid: {mermaid_path}")
    print(f"\nResult:")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
