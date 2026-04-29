"""Static example covering explicit tools, pass criteria, I/O, and memory."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
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


def explicit_score_tool(topic: str) -> dict:
    """Return deterministic scoring context for the static example."""
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


@global_config(
    prompt="Static QA agent with explicit tools and memory.",
    llm_config=LLMConfig(model="test"),
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
                if ctx.pass_criteria_feedback:
                    score = ctx.input.minimum_score + 5
                else:
                    score = ctx.input.minimum_score - 30

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
                    "Write a compact note for {topic} after static validation."
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


@pytest.mark.asyncio
async def test_static_tools_criteria_input_output_and_memory():
    judge_prompts: list[str] = []
    llm_calls: list[dict] = []

    async def fake_call_llm_api(**kwargs):
        if kwargs["system_prompt"].startswith("You are an output quality evaluator"):
            judge_prompts.append(kwargs["user_prompt"])
            if '"score": 60' in kwargs["user_prompt"]:
                return '{"pass": false, "feedback": "score below threshold"}'
            return '{"pass": true, "feedback": ""}'

        llm_calls.append(kwargs)
        saw_previous_runs = "Previous Runs" in kwargs["system_prompt"]
        return {
            "note": (
                "previous run was visible"
                if saw_previous_runs
                else "first run has no session memory yet"
            ),
            "saw_previous_runs": saw_previous_runs,
        }

    with patch(
        "flowforge.execution.llm.call_llm_api",
        new_callable=AsyncMock,
    ) as mock_api:
        mock_api.side_effect = fake_call_llm_api
        engine = FlowForge.compile(StaticToolsCriteriaMemoryAgent)

        first = await engine.run(StaticRequest(topic="alpha"))
        second = await engine.run(StaticRequest(topic="beta"))

    assert isinstance(first, StaticReport)
    assert first.topic == "alpha"
    assert first.score == 95
    assert first.tool_value == "metric:alpha"
    assert first.shared_tool_value == "metric:alpha"
    assert first.draft_attempt == 2
    assert first.used_feedback is True
    assert first.saw_previous_runs is False

    assert isinstance(second, StaticReport)
    assert second.topic == "beta"
    assert second.score == 95
    assert second.shared_tool_value == "metric:beta"
    assert second.draft_attempt == 2
    assert second.used_feedback is True
    assert second.saw_previous_runs is True

    assert len(judge_prompts) == 4
    assert "The score must be at least 90" in judge_prompts[0]
    assert any("Previous Feedback" in prompt for prompt in judge_prompts)
    assert len(llm_calls) == 2
    assert llm_calls[0]["output_schema"] is StaticMemoryNote
    assert "Previous Runs" not in llm_calls[0]["system_prompt"]
    assert "Previous Runs" in llm_calls[1]["system_prompt"]
    assert "alpha" in llm_calls[1]["system_prompt"]
