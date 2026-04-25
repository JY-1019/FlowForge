"""Tests for planner prompt assembly and gap metadata."""

from unittest.mock import AsyncMock, patch

import pytest

from flowforge import FlowForge, flow, global_config, step, task
from flowforge.doc.models import FlowDoc
from flowforge.planner.llm_planner import LLMPlanner
from flowforge.planner.prompt_builder import PromptBuilder
from flowforge.types import LLMConfig


@global_config(prompt="planner test agent")
class PlannerAgent:
    @flow(name="alpha", prompt="alpha prompt fallback")
    class AlphaFlow:
        @task(name="a", prompt="a")
        class ATask:
            @step(order=1, prompt="a step")
            async def a_step(ctx):
                return {"result": "alpha"}

    @flow(name="beta", prompt="beta prompt fallback")
    class BetaFlow:
        @task(name="b", prompt="b")
        class BTask:
            @step(order=1, prompt="b step")
            async def b_step(ctx):
                return {"result": "beta"}


class TestPromptBuilder:

    def test_includes_preconditions_and_prompt_fallback(self):
        engine = FlowForge.compile(PlannerAgent)
        docs = {
            "global.alpha": FlowDoc(
                summary="Alpha summary",
                capabilities=["reporting"],
                preconditions=["papers payload must already exist"],
            ),
        }

        prompt = PromptBuilder().build(
            "build a paper report",
            engine.dag,
            docs,
            global_prompt="planner system prompt",
        )

        assert "alpha  — Alpha summary" in prompt
        assert "pre: papers payload must already exist" in prompt
        assert "beta  — beta prompt fallback" in prompt
        assert "requirements[]" in prompt


class TestLLMPlanner:

    @pytest.mark.asyncio
    async def test_gap_metadata_round_trips_from_llm(self):
        engine = FlowForge.compile(PlannerAgent)
        planner = LLMPlanner(mode="autonomous")

        with patch(
            "flowforge.planner.llm_planner.call_with_tool",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.return_value = {
                "routes": ["beta", "alpha"],
                "rationale": "beta then alpha",
                "gap_detected": True,
                "reason": "missing fetch flow",
                "suggested_flow_name": "fetch_trending_papers",
                "suggested_flow_prompt": "Fetch trending papers and return JSON payload.",
                "requirements": [
                    {
                        "id": "fetch",
                        "description": "Fetch trending papers",
                        "covered": False,
                        "needs_flow": True,
                        "suggested_flow_name": "fetch_trending_papers",
                        "suggested_flow_prompt": "Fetch trending papers and return JSON payload.",
                    }
                ],
            }

            plan = await planner.plan(
                "make a report",
                engine.dag,
                {},
                LLMConfig(model="test"),
            )

        assert plan.metadata["routes"] == ["beta", "alpha"]
        assert plan.metadata["gap_detected"] is True
        assert plan.metadata["reason"] == "missing fetch flow"
        assert plan.metadata["suggested_flow_name"] == "fetch_trending_papers"
        assert plan.metadata["requirements"][0]["id"] == "fetch"
        assert "global.alpha" in plan.node_ids
        assert "global.beta" in plan.node_ids
