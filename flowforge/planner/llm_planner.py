"""LLM-based execution planner.

Flow-only planning
------------------
The planner operates at the **FLOW level only**.  Tasks and steps within a
flow are determined by their ``order`` and ``branch`` configuration — the
planner does not need to reason about them.

The LLM returns dot-separated **route paths** (e.g. ``"ml_platform.training"``),
which are expanded to full node-ID sets via ``dag.resolve_route()``.  This
automatically includes every ancestor (so the runner can reach the target) and
every descendant (tasks, steps).
"""
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

_PLAN_TOOL = {
    "name": "plan_execution",
    "description": (
        "Select the flow routes to execute. Return dot-separated route paths "
        "(e.g. 'ml_platform.training'). Do NOT include the 'global.' prefix."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "routes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Dot-separated flow route paths to execute. "
                    "Example: ['ml_platform.training', 'devops.ci_pipeline']"
                ),
            },
            "rationale": {
                "type": "string",
                "description": "Brief explanation of why these routes were chosen",
            },
        },
        "required": ["routes"],
    },
}


class LLMPlanner(AbstractPlanner):
    """Uses an LLM to select the execution path from the DAG.

    The planner works at the **flow level only**: it asks the LLM to pick
    route paths, then expands them via ``dag.resolve_route()`` to include
    all ancestors and descendants (tasks, steps).
    """

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
            routes, rationale = await self._call_llm(prompt, llm_config)
            # Expand route paths → full node-ID sets via resolve_route.
            node_ids = self._resolve_routes(routes, dag)
        except Exception as e:
            # Fallback to deterministic
            node_ids = self._deterministic.select(dag, docs)
            rationale = f"LLM planning failed ({e}), using deterministic order"

        return ExecutionPlan(
            node_ids=node_ids,
            mode=self._mode,
            rationale=rationale,
        )

    def _resolve_routes(
        self,
        routes: list[str],
        dag: FlowForgeDAG,
    ) -> list[str]:
        """Expand route paths to full node-ID sets.

        Each route (e.g. ``"ml_platform.training"``) is resolved via
        ``dag.resolve_route()`` which includes all ancestors up to
        ``"global"`` and all descendants (tasks, steps).

        Invalid routes are logged and skipped.
        """
        all_ids: set[str] = set()
        for route in routes:
            # Strip "global." prefix if the LLM included it
            route = route.removeprefix("global.")
            try:
                all_ids |= dag.resolve_route(route)
            except ValueError as e:
                logger.warning("planner returned invalid route %r: %s", route, e)
        return sorted(all_ids)

    async def _call_llm(
        self,
        prompt: str,
        llm_config: LLMConfig,
    ) -> tuple[list[str], str]:
        """Call the configured LLM provider and return (routes, rationale).

        Supports Anthropic, OpenAI, and Google Gemini via the shared
        :func:`flowforge.llm.caller.call_with_tool` helper.
        """
        try:
            data = await call_with_tool(
                prompt=prompt,
                tool_schema=_PLAN_TOOL,
                llm_config=llm_config,
                max_tokens=512,
            )
        except Exception as e:
            raise PlannerError(
                f"LLM planning call failed (provider={llm_config.provider}): {e}"
            ) from e

        return data.get("routes", []), data.get("rationale", "")
