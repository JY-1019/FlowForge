"""Static tools + criteria + memory demo for FlowForge.

Run:

    python examples/static_tools_criteria_memory_demo.py

This is the static counterpart to ``dynamic_skill_mcp_demo.py``.  It declares
all tools explicitly in annotations, uses typed input/output models, retries a
step with ``pass_criteria`` feedback, writes/read shared run data through
``ctx.shared_data``, and demonstrates that session memory from the first run
is available to the second ``ctx.call_llm()`` call.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from flowforge import FlowForge, flow, global_config, step, task
from flowforge.types import FunctionTool, LLMConfig


class StaticRequest(BaseModel):
    topic: str
    minimum_score: int = Field(default=90, ge=0, le=100)


class StaticDraft(BaseModel):
    topic: str
    score: int
    tool_value: str
    attempt: int
    used_feedback: bool


class StaticMemoryNote(BaseModel):
    note: str
    saw_previous_runs: bool = False


class StaticReport(BaseModel):
    topic: str
    score: int
    tool_value: str
    shared_tool_value: str
    draft_attempt: int
    used_feedback: bool
    llm_note: str
    saw_previous_runs: bool


def explicit_score_tool(topic: str) -> dict[str, Any]:
    return {
        "topic": topic,
        "tool_value": f"metric:{topic}",
        "score_boost": 35,
    }


EXPLICIT_SCORE_TOOL = FunctionTool(
    func=explicit_score_tool,
    name="explicit_score_tool",
    description="Return deterministic scoring context for a topic.",
)


def _llm_config_from_env() -> LLMConfig:
    model = os.getenv("FLOWFORGE_MODEL", "claude-sonnet-4-6").strip()
    max_tokens = int(os.getenv("FLOWFORGE_MAX_TOKENS", "2048"))
    return LLMConfig.for_claude(
        model=model,
        max_tokens=max_tokens,
        temperature=0.1,
    )


@global_config(
    prompt="Static QA agent with explicit tools and session memory.",
    llm_config=_llm_config_from_env(),
    tools=[EXPLICIT_SCORE_TOOL],
)
class StaticToolsCriteriaMemoryAgent:
    @flow(
        name="static_quality",
        prompt="Run a static quality workflow with explicit tool use.",
        input_schema=StaticRequest,
        output_schema=StaticReport,
        tools=[EXPLICIT_SCORE_TOOL],
    )
    class StaticQualityFlow:
        @task(
            name="quality_task",
            prompt="Create a scored draft, judge it, then produce a report.",
            input_schema=StaticRequest,
            output_schema=StaticReport,
            tools=[EXPLICIT_SCORE_TOOL],
        )
        class QualityTask:
            @step(
                order=1,
                prompt="Create a draft score from the explicit scoring tool.",
                input_schema=StaticRequest,
                output_schema=StaticDraft,
                tools=[EXPLICIT_SCORE_TOOL],
                pass_criteria=(
                    "The score must be at least 90 and tool_value must be "
                    "non-empty."
                ),
                pass_criteria_max_retries=1,
            )
            async def draft(ctx):
                tool_result = await ctx.call_tool(
                    "explicit_score_tool",
                    topic=ctx.input.topic,
                )
                ctx.shared_data["tool_value"] = tool_result["tool_value"]

                attempt = len(ctx.pass_criteria_feedback) + 1
                score = (
                    ctx.input.minimum_score + 5
                    if ctx.pass_criteria_feedback
                    else ctx.input.minimum_score - 30
                )
                return {
                    "topic": ctx.input.topic,
                    "score": score,
                    "tool_value": tool_result["tool_value"],
                    "attempt": attempt,
                    "used_feedback": bool(ctx.pass_criteria_feedback),
                }

            @step(
                order=2,
                prompt="Write a memory-aware note about the accepted draft.",
                input_schema=StaticDraft,
                output_schema=StaticMemoryNote,
            )
            async def memory_note(ctx):
                return await ctx.call_llm(
                    "Write a compact note for {topic} after static validation. "
                    "Set saw_previous_runs=true only if the system prompt "
                    "contains a Previous Runs section."
                )

            @step(
                order=3,
                prompt="Assemble the final static report from typed inputs.",
                input_schema=StaticMemoryNote,
                output_schema=StaticReport,
            )
            async def finalise(ctx):
                draft = ctx.previous_results.get("draft")
                return {
                    "topic": draft.topic,
                    "score": draft.score,
                    "tool_value": draft.tool_value,
                    "shared_tool_value": ctx.shared_data.get("tool_value", ""),
                    "draft_attempt": draft.attempt,
                    "used_feedback": draft.used_feedback,
                    "llm_note": ctx.input.note,
                    "saw_previous_runs": ctx.input.saw_previous_runs,
                }


def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-topic", default="alpha")
    parser.add_argument("--second-topic", default="beta")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is required for this static LLM demo.\n"
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    engine = FlowForge.compile(StaticToolsCriteriaMemoryAgent)

    first = await engine.run(StaticRequest(topic=args.first_topic))
    second = await engine.run(StaticRequest(topic=args.second_topic))

    print("=" * 72)
    print("Static tools + criteria + memory demo")
    print("=" * 72)
    print("First run result:")
    print(json.dumps(_dump_model(first), indent=2, ensure_ascii=False))
    print()
    print("Second run result:")
    print(json.dumps(_dump_model(second), indent=2, ensure_ascii=False))
    print()
    print("Session memory prompt block:")
    print(engine.memory.to_prompt_block())


if __name__ == "__main__":
    asyncio.run(main())
