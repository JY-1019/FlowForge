"""LLM-based execution planner."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from flowforge.llm.caller import call_with_tool
from flowforge.planner.base import AbstractPlanner, ExecutionPlan
from flowforge.planner.prompt_builder import PromptBuilder
from flowforge.planner.path_selector import DeterministicSelector
from flowforge.errors import PlannerError

if TYPE_CHECKING:
    from flowforge.schema.dag import FlowForgeDAG
    from flowforge.doc.models import AnyDoc
    from flowforge.types import LLMConfig


_PLAN_TOOL = {
    "name": "plan_execution",
    "description": "Return the ordered list of node IDs to execute",
    "input_schema": {
        "type": "object",
        "properties": {
            "node_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of DAG node IDs to execute",
            },
            "rationale": {
                "type": "string",
                "description": "Brief explanation of why this path was chosen",
            },
        },
        "required": ["node_ids"],
    },
}


class LLMPlanner(AbstractPlanner):
    """Uses an LLM to select the execution path from the DAG."""

    def __init__(self, mode: str = "deterministic") -> None:
        self._mode = mode
        self._prompt_builder = PromptBuilder()
        self._deterministic = DeterministicSelector()

    async def plan(
        self,
        user_request: Any,
        dag: FlowForgeDAG,
        docs: dict[str, AnyDoc],
        llm_config: LLMConfig,
    ) -> ExecutionPlan:
        if self._mode == "deterministic":
            node_ids = self._deterministic.select(dag, docs)
            return ExecutionPlan(node_ids=node_ids, mode="deterministic")

        # For autonomous/hybrid, call the LLM
        prompt = self._prompt_builder.build(user_request, dag, docs)
        try:
            node_ids, rationale = await self._call_llm(prompt, llm_config)
        except Exception as e:
            # Fallback to deterministic
            node_ids = self._deterministic.select(dag, docs)
            rationale = f"LLM planning failed ({e}), using deterministic order"

        return ExecutionPlan(
            node_ids=node_ids,
            mode=self._mode,
            rationale=rationale,
        )

    async def _call_llm(
        self,
        prompt: str,
        llm_config: LLMConfig,
    ) -> tuple[list[str], str]:
        """Call the configured LLM provider and return (node_ids, rationale).

        Supports Anthropic, OpenAI, and Google Gemini via the shared
        :func:`flowforge.llm.caller.call_with_tool` helper.
        """
        try:
            data = await call_with_tool(
                prompt=prompt,
                tool_schema=_PLAN_TOOL,
                llm_config=llm_config,
                max_tokens=1024,
            )
        except Exception as e:
            raise PlannerError(
                f"LLM planning call failed (provider={llm_config.provider}): {e}"
            ) from e

        return data.get("node_ids", []), data.get("rationale", "")
