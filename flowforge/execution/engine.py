"""ExecutionEngine — drives DAG execution and owns the RunTracer lifecycle."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from flowforge.execution.context import GlobalContext
from flowforge.execution.memory import SessionMemory
from flowforge.execution.runner import FlowRunner
from flowforge.schema.dag import FlowForgeDAG, NodeType
from flowforge.annotations.metadata import GlobalMeta
from flowforge.viz.run_trace import RunTracer, RunTrace

if TYPE_CHECKING:
    from flowforge.tools.registry import ToolRegistry
    from flowforge.doc.models import AnyDoc
    from flowforge.types import DynamicRunOptions

_logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Drives execution of the compiled DAG given user input.

    After each call to `run()` or `run_traced()`, the trace of that execution
    is available via `self.last_trace`.

    The engine maintains a ``SessionMemory`` that persists across ``run()``
    calls.  Each run automatically records a compact summary so the LLM can
    reference earlier results.  Clear it with ``engine.memory.clear()``.
    """

    def __init__(
        self,
        dag: FlowForgeDAG,
        global_meta: GlobalMeta,
        docs: dict[str, AnyDoc] | None = None,
        tool_registry: ToolRegistry | None = None,
        compiled_agent: Any = None,
        dynamic_options: DynamicRunOptions | None = None,
    ) -> None:
        self._dag          = dag
        self._global_meta  = global_meta
        self._docs         = docs if docs is not None else {}
        self._tool_registry = tool_registry
        self._compiled_agent = compiled_agent
        self._dynamic_options = dynamic_options
        self._flow_runner  = FlowRunner()
        self.last_trace: RunTrace | None = None
        self.last_dynamic_generation: dict[str, Any] | None = getattr(
            compiled_agent, "_last_dynamic_generation", None,
        )
        self.memory = SessionMemory()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        input_data: Any,
        planning_mode: str = "deterministic",
        route: str | list[str] | None = None,
        resume_from: Any = None,
        dynamic_options: DynamicRunOptions | dict[str, Any] | None = None,
    ) -> Any:
        """Execute the pipeline. Trace is stored in self.last_trace.

        Args:
            input_data:    Input for the first flow.
            planning_mode: "deterministic" | "autonomous" | "hybrid".
            route:         Execute only the specified path(s).
                           e.g. ``"web_analysis"`` or
                           ``"web_analysis.scrape_and_analyze"`` or
                           ``["web_analysis", "notification"]``.
                           Overrides planning_mode when set.
            resume_from:   A ``Checkpoint`` from a previous failed run.
        """
        result, trace = await self._execute(
            input_data, trace=True, planning_mode=planning_mode, route=route,
            resume_from=resume_from, dynamic_options=dynamic_options,
        )
        self.last_trace = trace
        return result

    async def run_traced(
        self,
        input_data: Any,
        planning_mode: str = "deterministic",
        route: str | list[str] | None = None,
        resume_from: Any = None,
        dynamic_options: DynamicRunOptions | dict[str, Any] | None = None,
    ) -> tuple[Any, RunTrace]:
        """Execute the pipeline and explicitly return (result, RunTrace)."""
        result, trace = await self._execute(
            input_data, trace=True, planning_mode=planning_mode, route=route,
            resume_from=resume_from, dynamic_options=dynamic_options,
        )
        self.last_trace = trace
        return result, trace

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _execute(
        self,
        input_data: Any,
        trace: bool = False,
        planning_mode: str = "deterministic",
        route: str | list[str] | None = None,
        resume_from: Any = None,
        dynamic_options: DynamicRunOptions | dict[str, Any] | None = None,
    ) -> tuple[Any, RunTrace]:
        from flowforge.tools.registry import ToolRegistry as _ToolRegistry
        from flowforge.types import DynamicRunOptions as _DynamicRunOptions

        tracer  = RunTracer(run_input=input_data) if trace else None
        tool_reg = self._tool_registry or _ToolRegistry()
        self._set_last_dynamic_generation(None)
        if isinstance(dynamic_options, _DynamicRunOptions):
            run_dynamic_options = dynamic_options
        elif dynamic_options is not None:
            run_dynamic_options = _DynamicRunOptions.model_validate(dynamic_options)
        elif self._dynamic_options is not None:
            run_dynamic_options = self._dynamic_options
        else:
            run_dynamic_options = _DynamicRunOptions()

        global_ctx = GlobalContext(
            llm_config=self._global_meta.llm_config,
            global_prompt=self._global_meta.prompt,
            tool_registry=tool_reg,
            tracer=tracer,
            global_tools=self._global_meta.tools,
            session_memory=self.memory,
            dynamic_options=run_dynamic_options,
        )
        global_ctx.all_docs = self._docs

        # Inject compiled agent reference for the dynamic flow generator.
        if self._compiled_agent is not None:
            global_ctx.shared_data["_compiled_agent"] = self._compiled_agent

        # ── Resume: restore checkpoint for skip logic ────────────────────
        if resume_from is not None:
            global_ctx.checkpoint = resume_from

        # ── Route: user-specified execution path ──────────────────────────
        planned_node_ids: set[str] | None = None
        planned_root_flow_ids: list[str] | None = None
        if route is not None:
            routes = [route] if isinstance(route, str) else route
            planned_node_ids, planned_root_flow_ids = self._resolve_route_selection(
                routes,
                allow_internal=True,
            )

        # ── AI Planning (autonomous / hybrid) ──────────────────────────────
        elif planning_mode != "deterministic" and self._docs:
            try:
                plan = await self._plan_with_llm(input_data, planning_mode)
                planned_node_ids, planned_root_flow_ids, user_flow_ids = (
                    self._extract_plan_selection(plan)
                )

                _logger.info(
                    "planner selected %d nodes (%d user flows) for mode=%s, "
                    "rationale: %s, flows: %s",
                    len(planned_node_ids), len(user_flow_ids), planning_mode,
                    plan.rationale, user_flow_ids,
                )

                # ── Dynamic flow: support full gaps and partial gaps ─────
                gap_detected = bool(plan.metadata.get("gap_detected", False))
                if (
                    (gap_detected or not user_flow_ids)
                    and self._global_meta.dynamic_flow
                    and run_dynamic_options.enabled
                    and self._compiled_agent is not None
                ):
                    _logger.info(
                        "planner reported %s — triggering dynamic flow "
                        "generation",
                        "partial gap" if gap_detected else "no matching user flows",
                    )
                    dynamic_results: list[dict[str, Any]] = []
                    dynamic_payloads = self._build_dynamic_inputs(input_data, plan)
                    if not dynamic_payloads:
                        _logger.warning(
                            "no valid dynamic payloads could be built — "
                            "falling back to deterministic execution"
                        )
                    for dynamic_input in dynamic_payloads:
                        dynamic_result = await self._execute_dynamic_generator(
                            global_ctx,
                            dynamic_input,
                        )
                        dynamic_results.append(dynamic_result)

                        if not dynamic_result.get("success", False):
                            current_output = dynamic_result
                            self._set_last_dynamic_generation(dynamic_result)
                            self.memory.record_run(
                                input_data=input_data,
                                output_data=current_output,
                                route=route,
                                planning_mode=planning_mode,
                            )
                            run_trace = (
                                tracer.finish_run(current_output) if tracer else RunTrace()
                            )
                            return current_output, run_trace

                    last_dynamic = (
                        dynamic_results[0]
                        if len(dynamic_results) == 1
                        else {"success": True, "generated": dynamic_results}
                    )
                    self._set_last_dynamic_generation(last_dynamic)

                    replanned = await self._plan_with_llm(input_data, planning_mode)
                    planned_node_ids, planned_root_flow_ids, user_flow_ids = (
                        self._extract_plan_selection(replanned)
                    )
                    _logger.info(
                        "replanned after dynamic injection: %d nodes, roots=%s",
                        len(planned_node_ids), planned_root_flow_ids,
                    )

            except Exception as e:
                _logger.warning(
                    "planning failed (%s), falling back to deterministic order", e
                )

        global_ctx.planned_node_ids = planned_node_ids
        global_ctx.planned_root_flow_ids = planned_root_flow_ids

        current_output = input_data

        try:
            current_output = await self._run_root_flows(global_ctx, input_data)
        except Exception as e:
            if tracer:
                tracer.finish_run(current_output, error=str(e))
            raise

        # Record this run in session memory for cross-run context.
        self.memory.record_run(
            input_data=input_data,
            output_data=current_output,
            route=route,
            planning_mode=planning_mode,
        )

        run_trace = tracer.finish_run(current_output) if tracer else RunTrace()
        return current_output, run_trace

    async def _plan_with_llm(
        self,
        input_data: Any,
        planning_mode: str,
    ) -> Any:
        from flowforge.planner.llm_planner import LLMPlanner

        planner = LLMPlanner(mode=planning_mode)
        return await planner.plan(
            input_data,
            self._dag,
            self._docs,
            self._global_meta.llm_config,
        )

    def _extract_plan_selection(
        self,
        plan: Any,
    ) -> tuple[set[str], list[str], list[str]]:
        planned_node_ids = set(plan.node_ids)

        all_flow_ids = sorted(
            nid for nid in planned_node_ids
            if nid.startswith("global.")
            and nid != "global"
            and self._dag.get_node(nid) is not None
            and self._dag.get_node(nid).type == NodeType.FLOW
        )
        user_flow_ids = [
            fid for fid in all_flow_ids
            if not self._dag.get_node(fid).name.startswith("_")
        ]
        internal_flow_ids = set(all_flow_ids) - set(user_flow_ids)

        if internal_flow_ids:
            internal_subtree: set[str] = set()
            for flow_id in internal_flow_ids:
                internal_subtree |= {
                    n.id for n in self._dag.get_all_nodes()
                    if n.id.startswith(flow_id)
                }
            planned_node_ids -= internal_subtree

        routes = plan.metadata.get("routes", []) if getattr(plan, "metadata", None) else []
        ordered_root_flow_ids = self._ordered_root_flow_ids_from_routes(
            routes,
            allow_internal=False,
        )
        if not ordered_root_flow_ids:
            ordered_root_flow_ids = [
                node.id
                for node in self._dag.get_children("global")
                if node.type == NodeType.FLOW
                and node.id in planned_node_ids
                and not node.name.startswith("_")
            ]

        return planned_node_ids, ordered_root_flow_ids, user_flow_ids

    def _resolve_route_selection(
        self,
        routes: list[str],
        *,
        allow_internal: bool,
    ) -> tuple[set[str], list[str]]:
        planned_node_ids: set[str] = set()
        for route in routes:
            planned_node_ids |= self._dag.resolve_route(route)

        ordered_root_flow_ids = self._ordered_root_flow_ids_from_routes(
            routes,
            allow_internal=allow_internal,
        )
        return planned_node_ids, ordered_root_flow_ids

    def _ordered_root_flow_ids_from_routes(
        self,
        routes: list[str],
        *,
        allow_internal: bool,
    ) -> list[str]:
        ordered_root_flow_ids: list[str] = []
        seen: set[str] = set()

        for route in routes:
            root_name = route.removeprefix("global.").split(".", 1)[0]
            root_id = f"global.{root_name}"
            node = self._dag.get_node(root_id)
            if node is None or node.type != NodeType.FLOW:
                continue
            if not allow_internal and node.name.startswith("_"):
                continue
            if root_id in seen:
                continue
            seen.add(root_id)
            ordered_root_flow_ids.append(root_id)

        return ordered_root_flow_ids

    def _get_root_flows(
        self,
        global_ctx: GlobalContext,
    ) -> list[Any]:
        root_flows = [
            node for node in self._dag.get_children("global")
            if node.type == NodeType.FLOW
        ]

        planned_node_ids = global_ctx.planned_node_ids
        if planned_node_ids is not None:
            root_flows = [node for node in root_flows if node.id in planned_node_ids]

        ordered_root_flow_ids = global_ctx.planned_root_flow_ids or []
        if ordered_root_flow_ids:
            by_id = {node.id: node for node in root_flows}
            ordered = [
                by_id[root_id]
                for root_id in ordered_root_flow_ids
                if root_id in by_id
            ]
            remaining = [
                node for node in root_flows
                if node.id not in {flow.id for flow in ordered}
            ]
            root_flows = ordered + remaining

        return root_flows

    async def _run_root_flows(
        self,
        global_ctx: GlobalContext,
        input_data: Any,
    ) -> Any:
        current_output = input_data
        for flow_node in self._get_root_flows(global_ctx):
            current_output = await self._flow_runner.run(
                flow_node.meta,
                global_ctx,
                current_output,
                parent_node_id="global",
            )
        return current_output

    def _build_dynamic_input(
        self,
        input_data: Any,
        plan: Any,
    ) -> Any | None:
        """Build a single dynamic-generation payload from plan metadata.

        Returns ``None`` when ``gap_detected`` is True but no
        ``suggested_flow_name`` is provided — the caller should treat this
        as an unrecoverable planning error rather than passing raw
        ``input_data`` to the dynamic generator.
        """
        metadata = getattr(plan, "metadata", {}) or {}
        if (
            metadata.get("gap_detected")
            and metadata.get("suggested_flow_name")
            and metadata.get("suggested_flow_prompt")
        ):
            return {
                "user_query": str(input_data),
                "gap_analysis": {
                    "covered": False,
                    "reason": metadata.get("reason", ""),
                    "suggested_flow_name": metadata.get("suggested_flow_name", ""),
                    "suggested_flow_prompt": metadata.get("suggested_flow_prompt", ""),
                },
                "downstream_flow_route": metadata.get("downstream_flow_route", ""),
            }
        # gap_detected=True but missing flow name/prompt → cannot generate.
        if metadata.get("gap_detected"):
            _logger.warning(
                "gap_detected=True but suggested_flow_name or "
                "suggested_flow_prompt is missing — skipping dynamic generation"
            )
            return None
        return input_data

    def _build_dynamic_inputs(
        self,
        input_data: Any,
        plan: Any,
    ) -> list[Any]:
        """Build one dynamic-generation payload per missing requirement."""
        metadata = getattr(plan, "metadata", {}) or {}
        requirements = metadata.get("requirements") or []
        payloads: list[Any] = []

        for index, requirement in enumerate(requirements):
            if not isinstance(requirement, dict):
                continue
            if requirement.get("covered", False):
                continue
            if not requirement.get("needs_flow", True):
                continue
            flow_name = requirement.get("suggested_flow_name")
            flow_prompt = requirement.get("suggested_flow_prompt")
            if not flow_name or not flow_prompt:
                continue
            payloads.append({
                "user_query": str(input_data),
                "requirement": requirement,
                "requirement_index": index,
                "gap_analysis": {
                    "covered": False,
                    "reason": requirement.get("description", ""),
                    "suggested_flow_name": flow_name,
                    "suggested_flow_prompt": flow_prompt,
                },
                "downstream_flow_route": requirement.get(
                    "downstream_flow_route",
                    metadata.get("downstream_flow_route", ""),
                ),
            })

        if payloads:
            return payloads
        fallback = self._build_dynamic_input(input_data, plan)
        if fallback is None:
            return []
        return [fallback]

    async def _execute_dynamic_generator(
        self,
        global_ctx: GlobalContext,
        dynamic_input: Any,
    ) -> dict[str, Any]:
        node = self._dag.get_node("global._dynamic_generator")
        if node is None or node.type != NodeType.FLOW:
            return {
                "success": False,
                "reason": "Internal _dynamic_generator flow is not available.",
            }

        result = await self._flow_runner.run(
            node.meta,
            global_ctx,
            dynamic_input,
            parent_node_id="global",
        )
        if isinstance(result, dict):
            return result
        return {
            "success": False,
            "reason": "Dynamic generator returned a non-dict result.",
            "result": result,
        }

    def _set_last_dynamic_generation(
        self,
        data: dict[str, Any] | None,
    ) -> None:
        self.last_dynamic_generation = data
        if self._compiled_agent is not None:
            self._compiled_agent._last_dynamic_generation = data
