"""FlowForge — Annotation-Based AI Agent Framework.

Public API
----------
Decorators::

    @global_config  — top-level agent class
    @flow           — execution unit (may contain sub-flows and tasks;
                      optionally a branch dispatcher)
    @task           — execution unit (may contain steps or child tasks;
                      optionally a branch dispatcher)
    @step           — atomic execution step (optionally a branch dispatcher)

Entry point::

    FlowForge.compile(MyAgent) → CompiledAgent

Branch dispatching
------------------
There is **no** ``@branch`` decorator.  Branch behaviour is built into
``@step``, ``@task``, and ``@flow`` via optional ``condition``,
``branches``, and ``fallback`` parameters.  See each decorator's docstring
for usage examples.

Minimal example::

    from flowforge import global_config, flow, task, step, FlowForge
    from flowforge import BranchCondition
    from pydantic import BaseModel

    class Query(BaseModel):
        text: str
        source: str  # "web" | "db"

    async def web_handler(ctx): return f"web: {ctx.input.text}"
    async def db_handler(ctx):  return f"db: {ctx.input.text}"

    @global_config(prompt="demo agent")
    class MyAgent:
        @flow(name="search", prompt="search for information")
        class SearchFlow:
            @task(name="dispatch", prompt="dispatch to correct backend")
            class DispatchTask:
                @step(
                    order=1,
                    prompt="route query to the right source",
                    condition=BranchCondition(field="source", enum=["web", "db"]),
                    branches={"web": web_handler, "db": db_handler},
                    fallback=web_handler,
                )
                async def route(ctx): ...

    import asyncio
    engine = FlowForge.compile(MyAgent)
    result = asyncio.run(engine.run(Query(text="hello", source="web")))
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

__version__ = "0.1.0"
__author__ = "Jongyeon Keum"
__license__ = "MIT"

from flowforge.annotations.decorators import (
    global_config,
    flow,
    task,
    step,
)
from flowforge.annotations.decorators import _GLOBAL_ATTR
from flowforge.types import LLMConfig, BranchCondition, MCPServer, ToolConfig
from flowforge.dynamic.generator import DynamicFlowGenerator
from flowforge.errors import (
    FlowForgeError,
    OrderConflictError,
    IOBindingError,
    BranchOutputMismatchError,
    CycleDetectedError,
    CompileError,
    ExecutionError,
    ApprovalRequired,
)

if TYPE_CHECKING:
    from flowforge.schema.dag import FlowForgeDAG
    from flowforge.doc.models import AnyDoc
    from flowforge.viz.run_trace import RunTrace, Checkpoint


class AgentSession:
    """Per-user session — isolates memory and trace while sharing the DAG.

    Created via ``CompiledAgent.create_session()``.  Each session has its
    own ``SessionMemory`` and ``last_trace``, so concurrent users never
    interfere with each other.

    The heavy, immutable resources (DAG, docs, tool registry) are shared
    with the parent ``CompiledAgent`` — no duplication.

    Typical FastAPI usage::

        # app startup
        agent = FlowForge.compile(MyAgent)
        await agent.generate_docs(planning_only=True)

        # per request
        @app.post("/chat")
        async def chat(req: ChatRequest):
            session = get_or_create_session(req.user_id)  # your session store
            result = await session.run(req.input)
            return result
    """

    def __init__(
        self,
        dag: FlowForgeDAG,
        global_meta: Any,
        docs: dict[str, AnyDoc],
    ) -> None:
        from flowforge.execution.engine import ExecutionEngine
        from flowforge.tools.registry import ToolRegistry

        self._dag = dag
        self._global_meta = global_meta
        self._docs = docs
        self._engine = ExecutionEngine(
            dag=dag,
            global_meta=global_meta,
            docs=docs,
            tool_registry=ToolRegistry(),
        )

    @property
    def memory(self) -> Any:
        """Session memory — persists across ``run()`` calls within this session."""
        return self._engine.memory

    @property
    def last_trace(self) -> RunTrace | None:
        return self._engine.last_trace

    async def run(
        self,
        input_data: Any,
        planning_mode: str = "deterministic",
        route: str | list[str] | None = None,
        resume_from: Any = None,
    ) -> Any:
        """Execute the pipeline. See ``CompiledAgent.run()`` for details."""
        return await self._engine.run(
            input_data, planning_mode=planning_mode, route=route,
            resume_from=resume_from,
        )

    async def run_traced(
        self,
        input_data: Any,
        planning_mode: str = "deterministic",
        route: str | list[str] | None = None,
        resume_from: Any = None,
    ) -> tuple[Any, RunTrace]:
        """Execute the pipeline and return (result, RunTrace)."""
        return await self._engine.run_traced(
            input_data, planning_mode=planning_mode, route=route,
            resume_from=resume_from,
        )

    def compare_mermaid(self, trace: RunTrace | None = None) -> str:
        """Return comparison Mermaid diagrams (full DAG vs executed path)."""
        from flowforge.viz.renderer import render_mermaid
        from flowforge.viz.subtree import render_run_mermaid

        t = trace or self.last_trace
        if t is None:
            raise RuntimeError("No run trace available.")

        status = "OK" if t.succeeded else "FAILED"
        dur = f"{t.duration_ms:.0f} ms" if t.duration_ms else "-"
        n_exec = len(t.executed_node_ids)
        n_total = len(self._dag.get_all_nodes())

        lines = [
            "# FlowForge - DAG vs Executed Path",
            "", "## 1. Full DAG Structure", "",
            "```mermaid", render_mermaid(self._dag), "```", "",
            f"## 2. Executed Path - Run `{t.run_id}`",
            f"> Status: **{status}** | Duration: **{dur}** | "
            f"Executed: **{n_exec} / {n_total}** nodes", "",
            "```mermaid", render_run_mermaid(self._dag, t), "```",
        ]
        return "\n".join(lines)


class CompiledAgent:
    """The result of FlowForge.compile() — holds DAG, docs, and a default engine.

    For **single-user / CLI** usage, call ``run()`` / ``run_traced()`` directly
    on this object.

    For **multi-user / server** usage (FastAPI, etc.), call
    ``create_session()`` to get a per-user ``AgentSession`` with isolated
    memory and trace::

        agent = FlowForge.compile(MyAgent)
        await agent.generate_docs(planning_only=True)

        # per user
        session = agent.create_session()
        result = await session.run(user_input)
    """

    def __init__(
        self,
        dag: FlowForgeDAG,
        global_meta: Any,
        docs: dict[str, AnyDoc] | None = None,
    ) -> None:
        from flowforge.execution.engine import ExecutionEngine
        from flowforge.tools.registry import ToolRegistry

        self._dag = dag
        self._global_meta = global_meta
        self._docs: dict[str, AnyDoc] = docs or {}
        self._tool_registry = ToolRegistry()
        self._engine = ExecutionEngine(
            dag=dag,
            global_meta=global_meta,
            docs=self._docs,
            tool_registry=self._tool_registry,
            compiled_agent=self,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def dag(self) -> FlowForgeDAG:
        return self._dag

    @property
    def docs(self) -> dict[str, AnyDoc]:
        return self._docs

    @property
    def last_trace(self) -> RunTrace | None:
        """Trace of the most recent run() call. None before first run."""
        return self._engine.last_trace

    @property
    def memory(self) -> Any:
        """Session memory that persists across ``run()`` calls.

        Stores compact summaries of previous runs so the LLM can reference
        earlier results.  Call ``engine.memory.clear()`` to reset.
        """
        return self._engine.memory

    def create_session(self) -> AgentSession:
        """Create a per-user session with isolated memory and trace.

        The session shares the compiled DAG and docs (immutable, zero-copy)
        but has its own ``SessionMemory`` and ``last_trace``.

        Use this in multi-user server environments (FastAPI, etc.) to
        prevent state leakage between concurrent users.
        """
        return AgentSession(
            dag=self._dag,
            global_meta=self._global_meta,
            docs=self._docs,
        )

    def add_flow(self, flow_cls: type) -> str:
        """Dynamically add a ``@flow``-decorated class to the live DAG.

        The new flow is compiled incrementally and attached under the
        ``"global"`` root.  The execution engine is updated in place so
        subsequent ``run()`` calls can route to the new flow.

        Parameters
        ----------
        flow_cls:
            A class decorated with ``@flow``.

        Returns
        -------
        str
            The DAG node ID of the newly added flow.

        Raises
        ------
        CompileError
            If *flow_cls* is not decorated with ``@flow`` or if a flow with
            the same name already exists.
        """
        from flowforge.annotations.decorators import _FLOW_ATTR
        from flowforge.schema.compiler import add_flow_to_dag
        from flowforge.schema.resolver import resolve_execution_order

        if not hasattr(flow_cls, _FLOW_ATTR):
            raise CompileError(
                f"{flow_cls.__name__} is not decorated with @flow"
            )

        flow_meta = getattr(flow_cls, _FLOW_ATTR)

        # Also add to global_meta.flows so recompile() preserves it.
        self._global_meta.flows.append(flow_meta)

        node_id = add_flow_to_dag(self._dag, flow_meta)

        # Re-validate topological order (catches cycles introduced by
        # depends_on edges from the new flow).
        resolve_execution_order(self._dag)

        # Rebuild the engine so it picks up the expanded DAG.
        self._engine = self._rebuild_engine()

        return node_id

    def recompile(self) -> None:
        """Full recompile of the DAG from the current ``global_meta``.

        Useful after programmatically mutating ``global_meta.flows``.
        """
        from flowforge.schema.compiler import compile_dag
        from flowforge.schema.resolver import resolve_execution_order

        self._dag = compile_dag(self._global_meta)
        resolve_execution_order(self._dag)
        self._engine = self._rebuild_engine()

    def _rebuild_engine(self):
        from flowforge.execution.engine import ExecutionEngine
        from flowforge.tools.registry import ToolRegistry

        engine = ExecutionEngine(
            dag=self._dag,
            global_meta=self._global_meta,
            docs=self._docs,
            tool_registry=self._tool_registry or ToolRegistry(),
            compiled_agent=self,
        )
        # Preserve session memory across rebuilds.
        engine.memory = self._engine.memory
        return engine

    async def generate_docs(
        self,
        force: bool = False,
        planning_only: bool = False,
    ) -> dict[str, AnyDoc]:
        """Generate AI docs for DAG nodes.

        Parameters
        ----------
        force:
            Regenerate docs even if cached.
        planning_only:
            When ``True``, only generate docs for GLOBAL and FLOW nodes.
            This is sufficient for the autonomous/hybrid planner (which
            operates at the flow level) and dramatically reduces the
            number of LLM calls.
        """
        from flowforge.doc.generator import DocGenerator
        from flowforge.doc.cache import DocCache
        from flowforge.schema.dag import NodeType

        node_types = {NodeType.GLOBAL, NodeType.FLOW} if planning_only else None

        generator = DocGenerator(
            llm_config=self._global_meta.llm_config,
            cache=DocCache(),
            force_regenerate=force,
        )
        docs = await generator.generate_all(
            self._dag.get_all_nodes(),
            node_types=node_types,
        )
        self._docs.update(docs)
        self._engine._docs = self._docs
        return docs

    async def run(
        self,
        input_data: Any,
        planning_mode: str = "deterministic",
        route: str | list[str] | None = None,
        resume_from: Any = None,
    ) -> Any:
        """Execute the agent pipeline. Trace stored in self.last_trace.

        Args:
            input_data:    The input passed to the first flow.
            planning_mode: ``"deterministic"`` (default) — run all nodes in
                           compiled order.
                           ``"autonomous"`` — call the LLM planner first;
                           only the nodes it selects are executed.
                           ``"hybrid"`` — same as autonomous but allows minor
                           deviations from the compiled path.
            route:         Execute only the specified path(s).
                           ``"web_analysis"`` — run only the web_analysis flow.
                           ``"web_analysis.scrape_and_analyze"`` — specific task.
                           ``["web_analysis", "notification"]`` — multiple routes.
                           Overrides ``planning_mode`` when set.
            resume_from:   A :class:`Checkpoint` (from a previous run's
                           ``trace.checkpoint``) to resume from.  Already-
                           completed nodes are skipped and their cached
                           outputs are reused.

        Note: ``"autonomous"`` / ``"hybrid"`` require ``generate_docs()`` to
        have been called first so the planner has doc summaries to reason over.
        """
        return await self._engine.run(
            input_data, planning_mode=planning_mode, route=route,
            resume_from=resume_from,
        )

    async def run_traced(
        self,
        input_data: Any,
        planning_mode: str = "deterministic",
        route: str | list[str] | None = None,
        resume_from: Any = None,
    ) -> tuple[Any, RunTrace]:
        """Execute the agent pipeline and explicitly return (result, RunTrace).

        See :meth:`run` for ``planning_mode`` and ``route`` options.
        """
        return await self._engine.run_traced(
            input_data, planning_mode=planning_mode, route=route,
            resume_from=resume_from,
        )

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def visualize(self, output_path: str = "dag.svg", **kwargs: Any) -> None:
        """Render the full DAG to a file.

        Tries graphviz (SVG/PNG) first; automatically falls back to Mermaid
        (saved as ``<output_path>.md``) when graphviz is not installed.
        """
        try:
            from flowforge.viz.renderer import render_graphviz
            render_graphviz(self._dag, output_path, **kwargs)
        except ImportError:
            md_path = str(output_path) + ".md"
            from flowforge.viz.renderer import render_mermaid
            with open(md_path, "w") as f:
                f.write("```mermaid\n" + render_mermaid(self._dag) + "\n```\n")
            print(
                f"[flowforge] graphviz not available — Mermaid diagram saved to {md_path}\n"
                "Install graphviz for SVG/PNG: pip install flowforge[viz] + brew/apt install graphviz"
            )

    def mermaid(self) -> str:
        """Return a Mermaid diagram string of the full DAG."""
        from flowforge.viz.renderer import render_mermaid
        return render_mermaid(self._dag)

    def visualize_run(
        self,
        output_path: str = "run.svg",
        trace: RunTrace | None = None,
        fmt: str = "svg",
    ) -> str:
        """Render the executed subtree of the last (or given) run.

        Executed nodes are colored with execution order and timing.
        Skipped nodes are grayed out with a dashed border.

        Args:
            output_path: Destination file path.
            trace:       Explicit RunTrace; defaults to self.last_trace.
            fmt:         "svg" | "png" | "pdf".

        Returns:
            The rendered file path as a string.

        Raises:
            RuntimeError: If no trace is available yet.
        """
        t = trace or self.last_trace
        if t is None:
            raise RuntimeError(
                "No run trace available. Call engine.run() or engine.run_traced() first."
            )
        try:
            from flowforge.viz.subtree import render_run_graphviz
            return str(render_run_graphviz(self._dag, t, output_path, fmt=fmt))
        except ImportError:
            md_path = str(output_path) + ".md"
            mermaid_str = self.run_mermaid(t)
            with open(md_path, "w") as f:
                f.write("```mermaid\n" + mermaid_str + "\n```\n")
            print(
                f"[flowforge] graphviz not available — Mermaid run diagram saved to {md_path}\n"
                "Install graphviz for SVG/PNG: pip install flowforge[viz] + brew/apt install graphviz"
            )
            return md_path

    def run_mermaid(self, trace: RunTrace | None = None) -> str:
        """Return a Mermaid diagram string for the last (or given) run."""
        from flowforge.viz.subtree import render_run_mermaid

        t = trace or self.last_trace
        if t is None:
            raise RuntimeError(
                "No run trace available. Call engine.run() or engine.run_traced() first."
            )
        return render_run_mermaid(self._dag, t)

    def compare_mermaid(self, trace: RunTrace | None = None) -> str:
        """Return a Markdown string with two Mermaid diagrams side-by-side:

        1. **Full DAG** — every node in the compiled structure.
        2. **Executed path** — same nodes, but executed ones are colored with
           order + timing and skipped ones are gray-dashed.

        Paste the output into any Markdown viewer (VS Code, GitHub, mermaid.live)
        to compare the agent's route against the full available structure.

        Args:
            trace: Explicit RunTrace; defaults to self.last_trace.

        Raises:
            RuntimeError: If no trace is available.
        """
        from flowforge.viz.renderer import render_mermaid
        from flowforge.viz.subtree import render_run_mermaid

        t = trace or self.last_trace
        if t is None:
            raise RuntimeError(
                "No run trace available. Call engine.run() or engine.run_traced() first."
            )

        status  = "✓ OK" if t.succeeded else "✗ FAILED"
        dur     = f"{t.duration_ms:.0f} ms" if t.duration_ms else "—"
        n_exec  = len(t.executed_node_ids)
        n_total = len(self._dag.get_all_nodes())

        lines = [
            "# FlowForge — DAG vs Executed Path",
            "",
            "## 1. Full DAG Structure",
            "> Every node compiled from the annotations.",
            "",
            "```mermaid",
            render_mermaid(self._dag),
            "```",
            "",
            f"## 2. Executed Path — Run `{t.run_id}`",
            f"> Status: **{status}** · Duration: **{dur}** · "
            f"Executed: **{n_exec} / {n_total}** nodes",
            ">",
            "> - **Colored** nodes ran (dark = global/flow/task/step, red = error)",
            "> - **Bold arrows** (`==>`) are the executed edges",
            "> - **Gray dashed** nodes were compiled but skipped this run",
            "",
            "```mermaid",
            render_run_mermaid(self._dag, t),
            "```",
        ]
        return "\n".join(lines)

    def print_run_summary(self, trace: RunTrace | None = None) -> None:
        """Print a terminal table summarising the last (or given) run."""
        from flowforge.viz.subtree import print_run_summary

        t = trace or self.last_trace
        if t is None:
            raise RuntimeError("No run trace available.")
        print_run_summary(t)


class FlowForge:
    """Main entry point — compile an annotated agent class into a runnable pipeline."""

    @staticmethod
    def compile(agent_cls: type) -> CompiledAgent:
        """
        Compile the @global_config-decorated class into a DAG.

        When ``@global_config(dynamic_flow=True)`` is set, the built-in
        ``_dynamic_generator`` meta-flow is automatically injected into the
        DAG so the engine can generate new flows at runtime.

        Usage:
            engine = FlowForge.compile(MyAgent)
        """
        if not hasattr(agent_cls, _GLOBAL_ATTR):
            raise CompileError(
                f"{agent_cls.__name__} is not decorated with @global_config"
            )

        global_meta = getattr(agent_cls, _GLOBAL_ATTR)

        # Inject the built-in dynamic generator flow when dynamic_flow=True.
        #
        # If the user has already defined a flow named "_dynamic_generator"
        # in their agent, we use theirs instead of the built-in one.  This
        # lets advanced users fully customise the dynamic generation pipeline
        # while still using the framework's plumbing (gap detection, compile
        # loop, DAG injection).
        if global_meta.dynamic_flow:
            import logging as _logging
            _compile_logger = _logging.getLogger(__name__)

            existing_names = {f.name for f in global_meta.flows}
            if "_dynamic_generator" in existing_names:
                _compile_logger.info(
                    "user-defined _dynamic_generator flow found — "
                    "using it instead of the built-in meta-flow"
                )
            else:
                from flowforge.dynamic.meta_flow import DynamicGeneratorFlow
                from flowforge.annotations.decorators import _FLOW_ATTR

                meta_flow_meta = getattr(DynamicGeneratorFlow, _FLOW_ATTR)
                global_meta.flows.append(meta_flow_meta)

        from flowforge.schema.compiler import compile_dag
        from flowforge.schema.resolver import resolve_execution_order

        dag = compile_dag(global_meta)

        # Validate — raises on cycles
        resolve_execution_order(dag)

        return CompiledAgent(dag=dag, global_meta=global_meta)


__all__ = [
    # Decorators — branch dispatching is a parameter of @step / @task / @flow
    "global_config",
    "flow",
    "task",
    "step",
    # Types
    "LLMConfig",
    "BranchCondition",
    "MCPServer",
    "ToolConfig",
    # Main classes
    "FlowForge",
    "CompiledAgent",
    "AgentSession",
    # Dynamic flow generation
    "DynamicFlowGenerator",
    # Errors
    "FlowForgeError",
    "OrderConflictError",
    "IOBindingError",
    "BranchOutputMismatchError",
    "CycleDetectedError",
    "CompileError",
    "ExecutionError",
    "ApprovalRequired",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
