"""ExecutionEngine — drives DAG execution and owns the RunTracer lifecycle."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from flowforge.execution.context import GlobalContext
from flowforge.execution.runner import FlowRunner
from flowforge.schema.dag import FlowForgeDAG, NodeType
from flowforge.annotations.metadata import GlobalMeta
from flowforge.viz.run_trace import RunTracer, RunTrace

if TYPE_CHECKING:
    from flowforge.tools.registry import ToolRegistry
    from flowforge.doc.models import AnyDoc


class ExecutionEngine:
    """Drives execution of the compiled DAG given user input.

    After each call to `run()` or `run_traced()`, the trace of that execution
    is available via `self.last_trace`.
    """

    def __init__(
        self,
        dag: FlowForgeDAG,
        global_meta: GlobalMeta,
        docs: dict[str, AnyDoc] | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._dag          = dag
        self._global_meta  = global_meta
        self._docs         = docs or {}
        self._tool_registry = tool_registry
        self._flow_runner  = FlowRunner()
        self.last_trace: RunTrace | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        input_data: Any,
        planning_mode: str = "deterministic",
        route: str | list[str] | None = None,
        resume_from: Any = None,
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
            resume_from=resume_from,
        )
        self.last_trace = trace
        return result

    async def run_traced(
        self,
        input_data: Any,
        planning_mode: str = "deterministic",
        route: str | list[str] | None = None,
        resume_from: Any = None,
    ) -> tuple[Any, RunTrace]:
        """Execute the pipeline and explicitly return (result, RunTrace)."""
        result, trace = await self._execute(
            input_data, trace=True, planning_mode=planning_mode, route=route,
            resume_from=resume_from,
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
    ) -> tuple[Any, RunTrace]:
        from flowforge.tools.registry import ToolRegistry as _ToolRegistry

        tracer  = RunTracer(run_input=input_data) if trace else None
        tool_reg = self._tool_registry or _ToolRegistry()

        # ── Route: user-specified execution path ──────────────────────────
        planned_node_ids: set[str] | None = None
        if route is not None:
            routes = [route] if isinstance(route, str) else route
            planned_node_ids = set()
            for r in routes:
                planned_node_ids |= self._dag.resolve_route(r)

        # ── AI Planning (autonomous / hybrid) ──────────────────────────────
        elif planning_mode != "deterministic" and self._docs:
            from flowforge.planner.llm_planner import LLMPlanner
            import logging as _logging
            _logger = _logging.getLogger(__name__)
            planner = LLMPlanner(mode=planning_mode)
            try:
                plan = await planner.plan(
                    input_data, self._dag, self._docs,
                    self._global_meta.llm_config,
                )
                planned_node_ids = set(plan.node_ids)
                # Log selected routes (flow-level paths) for observability
                flow_ids = sorted(
                    nid for nid in planned_node_ids
                    if nid.startswith("global.") and nid != "global"
                    and self._dag.get_node(nid) is not None
                    and self._dag.get_node(nid).type == NodeType.FLOW
                )
                _logger.info(
                    "planner selected %d nodes (%d flows) for mode=%s, "
                    "rationale: %s, flows: %s",
                    len(planned_node_ids), len(flow_ids), planning_mode,
                    plan.rationale, flow_ids,
                )
            except Exception as e:
                _logging.getLogger(__name__).warning(
                    "planning failed (%s), falling back to deterministic order", e
                )

        global_ctx = GlobalContext(
            llm_config=self._global_meta.llm_config,
            global_prompt=self._global_meta.prompt,
            tool_registry=tool_reg,
            tracer=tracer,
            global_tools=self._global_meta.tools,
        )
        global_ctx.all_docs = self._docs
        global_ctx.planned_node_ids = planned_node_ids

        # ── Resume: restore checkpoint for skip logic ────────────────────
        if resume_from is not None:
            global_ctx.checkpoint = resume_from

        root_flows = [
            n for n in self._dag.get_children("global")
            if n.type == NodeType.FLOW
        ]

        # Filter root-level flows according to the plan.
        if planned_node_ids is not None:
            root_flows = [n for n in root_flows if n.id in planned_node_ids]

        current_output = input_data
        error_msg: str | None = None

        try:
            for flow_node in root_flows:
                current_output = await self._flow_runner.run(
                    flow_node.meta,
                    global_ctx,
                    current_output,
                    parent_node_id="global",
                )
        except Exception as e:
            error_msg = str(e)
            # Re-raise after finalising the trace
            run_trace = tracer.finish_run(current_output, error=error_msg) if tracer else RunTrace()
            raise

        run_trace = tracer.finish_run(current_output) if tracer else RunTrace()
        return current_output, run_trace
