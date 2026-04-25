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
        "Select the flow routes to execute in order. Return dot-separated "
        "route paths (e.g. 'ml_platform.training'). Do NOT include the "
        "'global.' prefix. If the request still needs a missing flow, "
        "report the gap metadata too."
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
            "gap_detected": {
                "type": "boolean",
                "description": (
                    "True when some required capability is still missing from "
                    "the current DAG even after selecting runnable routes"
                ),
            },
            "reason": {
                "type": "string",
                "description": "Why the gap exists or why a new flow is needed",
            },
            "suggested_flow_name": {
                "type": "string",
                "description": "snake_case name for the missing flow",
            },
            "suggested_flow_prompt": {
                "type": "string",
                "description": "1-2 sentence prompt describing the missing flow",
            },
            "downstream_flow_route": {
                "type": "string",
                "description": (
                    "When gap_detected is true: the dot-separated route path of "
                    "the existing flow that should consume the missing flow's "
                    "output (e.g. 'paper_report_pipeline'). The missing flow "
                    "will be chained immediately BEFORE this flow. Leave empty "
                    "if no such downstream flow exists."
                ),
            },
            "requirements": {
                "type": "array",
                "description": (
                    "User request decomposed into ordered requirements. Each "
                    "requirement may map to an existing route or describe a "
                    "missing flow/tool capability."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "covered": {"type": "boolean"},
                        "matched_route": {"type": "string"},
                        "needs_flow": {"type": "boolean"},
                        "suggested_flow_name": {"type": "string"},
                        "suggested_flow_prompt": {"type": "string"},
                        "downstream_flow_route": {"type": "string"},
                        "needs_tool": {"type": "boolean"},
                        "suggested_tool_name": {"type": "string"},
                        "dependency_hints": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["description", "covered"],
                },
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
            return ExecutionPlan(
                node_ids=node_ids,
                mode="deterministic",
                metadata={"routes": []},
            )

        # For autonomous/hybrid, call the LLM
        global_node = dag.get_node("global")
        global_prompt = getattr(getattr(global_node, "meta", None), "prompt", "")
        prompt = self._prompt_builder.build(
            user_request,
            dag,
            docs,
            global_prompt=global_prompt,
        )
        try:
            data = await self._call_llm(prompt, llm_config)
            routes = data.get("routes", [])
            rationale = data.get("rationale", "")
            # Expand route paths → full node-ID sets via resolve_route.
            node_ids = self._resolve_routes(routes, dag)
        except Exception as e:
            # Fallback to deterministic
            node_ids = self._deterministic.select(dag, docs)
            rationale = f"LLM planning failed ({e}), using deterministic order"
            data = {
                "routes": [],
                "gap_detected": False,
                "reason": "",
                "suggested_flow_name": "",
                "suggested_flow_prompt": "",
                "downstream_flow_route": "",
                "requirements": [],
            }

        return ExecutionPlan(
            node_ids=node_ids,
            mode=self._mode,
            rationale=rationale,
            metadata={
                "routes": data.get("routes", []),
                "gap_detected": bool(data.get("gap_detected", False)),
                "reason": data.get("reason", ""),
                "suggested_flow_name": data.get("suggested_flow_name", ""),
                "suggested_flow_prompt": data.get("suggested_flow_prompt", ""),
                "downstream_flow_route": data.get("downstream_flow_route", ""),
                "requirements": data.get("requirements", []),
            },
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
        ordered = [
            node.id
            for node in dag.topological_order()
            if node.id in all_ids
        ]
        return ordered

    async def _call_llm(
        self,
        prompt: str,
        llm_config: LLMConfig,
    ) -> dict[str, Any]:
        """Call the configured LLM provider and return planner JSON data.

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

        return data
