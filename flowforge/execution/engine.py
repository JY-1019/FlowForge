"""ExecutionEngine — drives DAG execution and owns the RunTracer lifecycle."""
from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

from flowforge.execution.context import GlobalContext
from flowforge.execution.memory import SessionMemory
from flowforge.execution.parallel import run_parallel
from flowforge.execution.runner import FlowRunner, _group_by_order
from flowforge.observability import node_span
from flowforge.errors import PlannerError
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

        # ── Harness Ledger: activate when any root flow carries @ledger ──
        self._setup_ledger(global_ctx, tracer, input_data)

        # Inject compiled agent reference for the dynamic flow generator.
        if self._compiled_agent is not None:
            global_ctx.shared_data["_compiled_agent"] = self._compiled_agent

        if (
            self._global_meta.dynamic_flow
            and run_dynamic_options.enabled
            and self._compiled_agent is not None
        ):
            self._install_dynamic_runtime_repairs(
                global_ctx=global_ctx,
                input_data=input_data,
            )
            await self._repair_stale_dynamic_flows(
                global_ctx=global_ctx,
                input_data=input_data,
            )

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

            # ── Dynamic flow: support full gaps and partial gaps ─────────
            gap_detected = bool(plan.metadata.get("gap_detected", False))
            if (
                (gap_detected or not user_flow_ids)
                and self._global_meta.dynamic_flow
                and run_dynamic_options.enabled
                and self._compiled_agent is not None
            ):
                _logger.info(
                    "planner reported %s — triggering dynamic flow generation",
                    "partial gap" if gap_detected else "no matching user flows",
                )
                dynamic_results: list[dict[str, Any]] = []
                dynamic_payloads = self._build_dynamic_inputs(
                    input_data, plan, run_dynamic_options,
                )
                if not dynamic_payloads:
                    raise PlannerError(
                        "planner requested dynamic generation, but no valid "
                        "dynamic payloads could be built"
                    )
                for dynamic_input in dynamic_payloads:
                    try:
                        dynamic_result = await self._execute_dynamic_generator(
                            global_ctx,
                            dynamic_input,
                        )
                    except Exception as dynamic_exc:
                        dynamic_result = {
                            "success": False,
                            "injected": False,
                            "reason": str(dynamic_exc),
                            "error": str(dynamic_exc),
                        }
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

        global_ctx.planned_node_ids = planned_node_ids
        global_ctx.planned_root_flow_ids = planned_root_flow_ids

        current_output = input_data

        try:
            current_output = await self._run_root_flows(global_ctx, input_data)
        except Exception as e:
            recovered = await self._try_dynamic_recovery_after_failure(
                global_ctx=global_ctx,
                input_data=input_data,
                error=e,
                planning_mode=planning_mode,
                route=route,
                run_dynamic_options=run_dynamic_options,
                planned_root_flow_ids=planned_root_flow_ids,
            )
            if recovered:
                current_output = await self._run_root_flows(global_ctx, input_data)
            else:
                if tracer:
                    tracer.finish_run(current_output, error=str(e))
                self._finish_ledger(global_ctx, current_output, error=str(e))
                raise

        # Record this run in session memory for cross-run context.
        self.memory.record_run(
            input_data=input_data,
            output_data=current_output,
            route=route,
            planning_mode=planning_mode,
        )

        self._finish_ledger(global_ctx, current_output, error=None)
        run_trace = tracer.finish_run(current_output) if tracer else RunTrace()
        return current_output, run_trace

    # ------------------------------------------------------------------
    # Harness Ledger wiring
    # ------------------------------------------------------------------

    def _setup_ledger(self, global_ctx: GlobalContext, tracer: Any, input_data: Any) -> None:
        config = next(
            (f.ledger for f in self._global_meta.flows if getattr(f, "ledger", None)),
            None,
        )
        if config is None:
            return
        from flowforge.ledger.core import Ledger
        from flowforge.viz.run_trace import _safe_repr

        led = Ledger(config, llm_config=self._global_meta.llm_config)
        run_id = tracer.trace.run_id if tracer else "run"
        led.start_run(run_id, _safe_repr(input_data))
        global_ctx.ledger = led

    def _finish_ledger(self, global_ctx: GlobalContext, output: Any, error: str | None) -> None:
        led = getattr(global_ctx, "ledger", None)
        if led is None:
            return
        from flowforge.viz.run_trace import _safe_repr

        try:
            led.finish_run(_safe_repr(output), error)
        finally:
            led.close()
            global_ctx.ledger = None

    async def _try_dynamic_recovery_after_failure(
        self,
        *,
        global_ctx: GlobalContext,
        input_data: Any,
        error: Exception,
        planning_mode: str,
        route: str | list[str] | None,
        run_dynamic_options: DynamicRunOptions,
        planned_root_flow_ids: list[str] | None,
    ) -> bool:
        """Generate a missing upstream flow when planner under-reported a gap."""
        if (
            route is not None
            or planning_mode == "deterministic"
            or not self._global_meta.dynamic_flow
            or not run_dynamic_options.enabled
            or self._compiled_agent is None
            or not planned_root_flow_ids
        ):
            return False
        if global_ctx.shared_data.get("_dynamic_runtime_failure_recovered", False):
            return False

        downstream_route = planned_root_flow_ids[0].removeprefix("global.")
        dynamic_input = self._build_runtime_failure_dynamic_input(
            input_data=input_data,
            error=error,
            downstream_route=downstream_route,
        )
        _logger.warning(
            "execution failed after planning; attempting dynamic upstream "
            "generation before %s: %s",
            downstream_route, error,
        )
        try:
            dynamic_result = await self._execute_dynamic_generator(
                global_ctx,
                dynamic_input,
            )
        except Exception as dynamic_exc:
            dynamic_result = {
                "success": False,
                "injected": False,
                "reason": str(dynamic_exc),
                "error": str(dynamic_exc),
            }
        self._set_last_dynamic_generation(dynamic_result)
        if not dynamic_result.get("success", False):
            return False

        replanned = await self._plan_with_llm(input_data, planning_mode)
        planned_node_ids, planned_root_flow_ids, _user_flow_ids = (
            self._extract_plan_selection(replanned)
        )
        global_ctx.planned_node_ids = planned_node_ids
        global_ctx.planned_root_flow_ids = planned_root_flow_ids
        global_ctx.shared_data["_dynamic_runtime_failure_recovered"] = True
        return True

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
        root_flows = self._order_dynamic_bridge_flows(root_flows)

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
            root_flows = self._order_dynamic_bridge_flows(root_flows)

        return root_flows

    @staticmethod
    def _order_dynamic_bridge_flows(root_flows: list[Any]) -> list[Any]:
        """Honor generated-flow bridge metadata during bridge execution."""

        ordered = list(root_flows)
        for node in list(ordered):
            downstream = getattr(
                node.meta,
                "__flowforge_dynamic_downstream_flow_route__",
                "",
            )
            if not downstream:
                continue
            target = downstream.removeprefix("global.").split(".", 1)[0]
            if not target:
                continue
            target_id = f"global.{target}"
            node_index = next(
                (i for i, candidate in enumerate(ordered) if candidate.id == node.id),
                None,
            )
            target_index = next(
                (i for i, candidate in enumerate(ordered) if candidate.id == target_id),
                None,
            )
            if node_index is None or target_index is None or node_index < target_index:
                continue
            bridge = ordered.pop(node_index)
            target_index = next(
                i for i, candidate in enumerate(ordered) if candidate.id == target_id
            )
            ordered.insert(target_index, bridge)
        return ordered

    def _install_dynamic_runtime_repairs(
        self,
        *,
        global_ctx: GlobalContext,
        input_data: Any,
    ) -> None:
        """Attach repair hooks to persisted generated flows loaded at compile time."""
        try:
            from flowforge.dynamic.generator import DynamicFlowGenerator
            from flowforge.dynamic.meta_flow import _install_runtime_repair
        except Exception as exc:
            _logger.warning("dynamic runtime repair unavailable: %s", exc)
            return

        generator = DynamicFlowGenerator(
            llm_config=self._global_meta.llm_config,
            dag=self._dag,
            docs=self._docs,
            tool_configs=self._global_meta.tools,
            dynamic_options=global_ctx.dynamic_options,
        )

        for flow_meta in list(self._global_meta.flows):
            if getattr(flow_meta, "runtime_repair", None) is not None:
                continue
            source_code = getattr(
                flow_meta,
                "__flowforge_dynamic_source_code__",
                "",
            )
            if not source_code:
                continue

            downstream_route = getattr(
                flow_meta,
                "__flowforge_dynamic_downstream_flow_route__",
                "",
            )
            downstream_contract, downstream_name = (
                generator.resolve_downstream_contract(downstream_route)
            )
            project_context = generator.build_code_generation_context(
                flow_name=flow_meta.name,
                flow_prompt=flow_meta.prompt,
                user_query=input_data,
                downstream_contract=downstream_contract,
                downstream_flow_name=downstream_name,
            )
            _install_runtime_repair(
                flow_meta=flow_meta,
                agent=self._compiled_agent,
                generator=generator,
                global_ctx=global_ctx,
                initial_code=source_code,
                user_query=input_data,
                downstream_contract=downstream_contract,
                downstream_flow_name=downstream_name,
                project_context=project_context,
                downstream_flow_route=downstream_route,
            )

    async def _repair_stale_dynamic_flows(
        self,
        *,
        global_ctx: GlobalContext,
        input_data: Any,
    ) -> None:
        """Preflight persisted generated code and repair stale bad patterns."""
        try:
            from flowforge.dynamic.generator import DynamicFlowGenerator
        except Exception as exc:
            _logger.warning("dynamic preflight repair unavailable: %s", exc)
            return

        generator = DynamicFlowGenerator(
            llm_config=self._global_meta.llm_config,
            dag=self._dag,
            docs=self._docs,
            tool_configs=self._global_meta.tools,
            dynamic_options=global_ctx.dynamic_options,
        )

        for flow_meta in list(self._global_meta.flows):
            source_code = getattr(
                flow_meta,
                "__flowforge_dynamic_source_code__",
                "",
            )
            if not source_code:
                continue
            quality_error = generator.check_generated_code_quality(
                source_code,
                input_data,
            )
            if quality_error is None:
                continue
            repair = getattr(flow_meta, "runtime_repair", None)
            if not callable(repair):
                continue
            _logger.warning(
                "preflight repair for generated flow %s: %s",
                flow_meta.name, quality_error,
            )
            repaired = False
            try:
                repaired = bool(await repair(
                    "Preflight generated-code quality check failed: "
                    f"{quality_error}"
                ))
            except Exception as exc:
                _logger.warning(
                    "preflight repair failed for generated flow %s: %s",
                    flow_meta.name, exc,
                )
            if not repaired:
                self._drop_dynamic_flow(flow_meta.name)

    def _drop_dynamic_flow(self, flow_name: str) -> None:
        """Remove an unusable generated flow so planning can regenerate it."""
        node_id = f"global.{flow_name}"
        self._dag.remove_subtree(node_id)
        self._global_meta.flows = [
            flow for flow in self._global_meta.flows
            if flow.name != flow_name
        ]
        _logger.warning(
            "dropped stale generated flow %s after failed preflight repair",
            flow_name,
        )

    async def _run_root_flows(
        self,
        global_ctx: GlobalContext,
        input_data: Any,
    ) -> Any:
        with node_span("flowforge.run", "run", name=self._global_meta.prompt or "run"):
            return await self._run_root_flows_impl(global_ctx, input_data)

    async def _run_root_flows_impl(
        self,
        global_ctx: GlobalContext,
        input_data: Any,
    ) -> Any:
        current_output = input_data
        groups = _group_by_order(
            self._get_root_flows(global_ctx),
            lambda flow_node: flow_node.meta.order,
        )
        for group in groups:
            unique_flows = [flow_node for flow_node in group if flow_node.meta.unique]
            run_group = [unique_flows[0]] if unique_flows else group

            if len(run_group) == 1:
                current_output = await self._flow_runner.run(
                    run_group[0].meta,
                    global_ctx,
                    current_output,
                    parent_node_id="global",
                )
            else:
                coros = [
                    self._flow_runner.run(
                        flow_node.meta,
                        global_ctx,
                        current_output,
                        parent_node_id="global",
                    )
                    for flow_node in run_group
                ]
                results = await run_parallel(coros)
                current_output = results[-1] if results else current_output
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
            flow_name = self._synthesise_flow_name(input_data, metadata)
            flow_prompt = self._synthesise_flow_prompt(input_data, metadata)
            _logger.warning(
                "gap_detected=True but suggested_flow_name or "
                "suggested_flow_prompt is missing — synthesizing dynamic "
                "flow payload %r",
                flow_name,
            )
            return {
                "user_query": str(input_data),
                "gap_analysis": {
                    "covered": False,
                    "reason": metadata.get("reason", ""),
                    "suggested_flow_name": flow_name,
                    "suggested_flow_prompt": flow_prompt,
                },
                "downstream_flow_route": metadata.get("downstream_flow_route", ""),
            }
        return input_data

    def _build_runtime_failure_dynamic_input(
        self,
        *,
        input_data: Any,
        error: Exception,
        downstream_route: str,
    ) -> dict[str, Any]:
        reason = (
            f"The selected downstream flow '{downstream_route}' could not "
            f"consume the original request directly. Runtime error: {error}"
        )
        flow_name = self._synthesise_flow_name(
            input_data,
            {"reason": f"{input_data} before {downstream_route}"},
        )
        flow_prompt = (
            "Create an upstream FlowForge flow that satisfies the data "
            f"collection/preparation part of this request and returns a "
            f"payload consumable by `{downstream_route}`. Original request: "
            f"{str(input_data)[:700]}. Downstream failure: {str(error)[:400]}"
        )
        return {
            "user_query": str(input_data),
            "gap_analysis": {
                "covered": False,
                "reason": reason,
                "suggested_flow_name": flow_name,
                "suggested_flow_prompt": flow_prompt,
            },
            "downstream_flow_route": downstream_route,
        }

    @staticmethod
    def _synthesise_flow_name(input_data: Any, metadata: dict[str, Any]) -> str:
        text = str(metadata.get("reason") or input_data or "dynamic flow")
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower())
        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "into",
            "must", "should", "using", "use", "need", "needs", "request",
            "project", "create", "generate", "flowforge",
        }
        useful = [word for word in words if word not in stop_words]
        stem = "_".join(useful[:5]) or "dynamic_generated"
        if not stem.startswith("dynamic_"):
            stem = f"dynamic_{stem}"
        return stem[:80].strip("_") or "dynamic_generated"

    @staticmethod
    def _synthesise_flow_prompt(input_data: Any, metadata: dict[str, Any]) -> str:
        reason = str(metadata.get("reason") or "").strip()
        request = str(input_data)
        if reason:
            return (
                f"Dynamically satisfy the missing capability: {reason}. "
                f"User request: {request[:600]}"
            )
        return f"Dynamically satisfy this user request: {request[:700]}"

    def _build_dynamic_inputs(
        self,
        input_data: Any,
        plan: Any,
        dynamic_options: DynamicRunOptions | None = None,
    ) -> list[Any]:
        """Build one dynamic-generation payload per missing requirement.

        Before building a payload, checks:
        1. Whether a flow with the same name already exists in the DAG
           (may have been injected earlier in this session).
        2. Whether the manifest already has a record for this flow name
           (generated in a previous process but not loaded — e.g.
           ``auto_load_generated=False``).

        Either condition skips generation for that requirement.
        """
        metadata = getattr(plan, "metadata", {}) or {}
        requirements = metadata.get("requirements") or []
        payloads: list[Any] = []

        opts = dynamic_options or self._dynamic_options

        # Collect existing flow names from DAG for dedup
        existing_flow_names: set[str] = set()
        for node in self._dag.get_all_nodes():
            if node.type == NodeType.FLOW:
                existing_flow_names.add(node.name)

        # Also check manifest on disk (covers auto_load_generated=False case)
        manifest_flow_names: set[str] = set()
        if opts is not None:
            try:
                from flowforge.dynamic.manifest import load_manifest
                manifest = load_manifest(opts)
                manifest_flow_names = {r.name for r in manifest.flows}
            except Exception:
                pass

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

            # Skip if a flow with this name already exists
            if flow_name in existing_flow_names:
                _logger.info(
                    "skipping dynamic generation for %r — already in DAG",
                    flow_name,
                )
                continue
            if flow_name in manifest_flow_names:
                _logger.info(
                    "skipping dynamic generation for %r — found in manifest "
                    "(consider auto_load_generated=True to reuse it)",
                    flow_name,
                )
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

        previous_planned = global_ctx.planned_node_ids
        previous_roots = global_ctx.planned_root_flow_ids
        try:
            internal_ids = {
                n.id for n in self._dag.get_all_nodes()
                if n.id == "global._dynamic_generator"
                or n.id.startswith("global._dynamic_generator.")
            }
            global_ctx.planned_node_ids = internal_ids
            global_ctx.planned_root_flow_ids = ["global._dynamic_generator"]
            result = await self._flow_runner.run(
                node.meta,
                global_ctx,
                dynamic_input,
                parent_node_id="global",
            )
        finally:
            global_ctx.planned_node_ids = previous_planned
            global_ctx.planned_root_flow_ids = previous_roots
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
