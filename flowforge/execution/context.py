"""Context hierarchy for FlowForge execution.

Contexts flow top-down through the execution stack:

    GlobalContext
        Holds agent-wide configuration (LLM settings, tool registry, docs,
        optional tracer).  One instance per ``engine.run()`` call.

    FlowContext   (child of GlobalContext)
        Created for each flow invocation.  Holds the flow-level prompt, input
        data, and a mutable ``flow_state`` dict for intra-flow communication.

    TaskContext   (child of FlowContext)
        Created for each task invocation.  Accumulates step results in an
        ``OrderedDict`` keyed by step ``order`` so later steps can inspect
        earlier outputs.

    StepContext   (child of TaskContext)
        Created for each step invocation.  Exposes the step-level prompt,
        validated input, accumulated previous results, and convenience
        properties for tool and LLM access.

        When the step is a *branch dispatcher* (``StepMeta.is_branch``),
        two additional fields are populated by the runner after the condition
        is resolved:

        ``condition_value``
            The raw value extracted from the input for the discriminator
            field.
        ``selected_branch``
            The key of the handler that was selected (e.g. ``"csv"`` or
            ``"__fallback__"``).  Recorded in the run trace.

Note — no BranchContext
~~~~~~~~~~~~~~~~~~~~~~~~
There is no separate ``BranchContext`` class.  Branch dispatching is a
runtime behaviour of ``@step``, ``@task``, and ``@flow`` nodes — not a
structural type.  ``StepContext`` is always the context object received by
step handlers, whether the step is branching or not.  This keeps handler
signatures uniform and simplifies user code.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.types import LLMConfig, ToolConfig
    from flowforge.tools.registry import ToolRegistry
    from flowforge.doc.models import AnyDoc
    from flowforge.viz.run_trace import RunTracer


class GlobalContext:
    """Top-level context holding shared, agent-wide configuration.

    One ``GlobalContext`` is created per ``engine.run()`` call and passed
    down to every FlowContext, TaskContext, and StepContext in that run.

    Attributes
    ----------
    llm_config:
        Default LLM configuration (model, temperature, …) used by nodes that
        do not declare their own.
    global_prompt:
        The ``@global_config`` prompt prepended to every LLM call.
    tool_registry:
        Registry of all globally available tools (MCP servers, function
        tools, HTTP adapters).
    env_vars:
        Arbitrary key/value pairs injected into the run (e.g. API keys,
        feature flags).  Available as ``ctx.global_ctx.env_vars``.
    all_docs:
        AI-generated documentation for every DAG node, keyed by node ID.
        Populated by ``CompiledAgent.generate_docs()``.
    tracer:
        Optional ``RunTracer`` that records every node start / finish / error
        for post-run visualisation.  ``None`` unless ``run_traced()`` is used.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        global_prompt: str,
        tool_registry: ToolRegistry,
        env_vars: dict[str, str] | None = None,
        tracer: RunTracer | None = None,
        global_tools: list[ToolConfig] | None = None,
    ) -> None:
        self.llm_config     = llm_config
        self.global_prompt  = global_prompt
        self.tool_registry  = tool_registry
        self.env_vars: dict[str, str] = env_vars or {}
        self.all_docs: dict[str, AnyDoc] = {}
        self.tracer: RunTracer | None = tracer
        # Set by the engine when planning_mode != "deterministic".
        # None means "run everything"; a set means "only run these node IDs".
        self.planned_node_ids: set[str] | None = None
        # Raw ToolConfig list from @global_config for hierarchical merging.
        self.global_tools: list[ToolConfig] = global_tools or []


class FlowContext:
    """Context for a single flow invocation.

    Created by ``FlowRunner`` at the start of each flow execution and passed
    down to ``TaskRunner`` instances.

    Attributes
    ----------
    global_ctx:
        Reference to the shared ``GlobalContext`` for this run.
    flow_name:
        The ``name`` field of the ``@flow`` decorator.
    flow_prompt:
        The ``prompt`` field of the ``@flow`` decorator.
    flow_input:
        The validated input data passed to this flow invocation.
    parent_flow_output:
        The output of the parent flow (when this flow is nested).
        ``None`` for top-level flows.
    flow_state:
        Mutable dictionary for intra-flow communication.  Steps and tasks
        may store intermediate state here, but prefer returning values via
        the normal output chain when possible.
    flow_doc:
        AI-generated documentation for this flow, or ``None`` if not yet
        generated.
    """

    def __init__(
        self,
        global_ctx: GlobalContext,
        flow_name: str,
        flow_prompt: str,
        flow_input: Any = None,
        parent_flow_output: Any = None,
        flow_tools: list[ToolConfig] | None = None,
    ) -> None:
        self.global_ctx          = global_ctx
        self.flow_name           = flow_name
        self.flow_prompt         = flow_prompt
        self.flow_input          = flow_input
        self.parent_flow_output  = parent_flow_output
        self.flow_state: dict[str, Any] = {}
        self.flow_doc = global_ctx.all_docs.get(f"global.{flow_name}")
        # Tools declared on this flow (not yet merged with parent).
        self.flow_tools: list[ToolConfig] = flow_tools or []


class TaskContext:
    """Context for a single task invocation.

    Created by ``TaskRunner`` and passed down to ``StepRunner`` instances.

    Attributes
    ----------
    flow_ctx:
        The ``FlowContext`` of the enclosing flow.
    task_name:
        The ``name`` field of the ``@task`` decorator.
    task_prompt:
        The ``prompt`` field of the ``@task`` decorator.
    task_input:
        The validated input data passed to this task.
    parent_task_output:
        The output of the parent task when this task is a child in a
        container-task hierarchy.  ``None`` for top-level tasks.
    step_results:
        Ordered mapping from ``order`` integer → step output.  Accumulated
        as each step completes.  Accessible from any step via
        ``ctx.previous_results``.
    """

    def __init__(
        self,
        flow_ctx: FlowContext,
        task_name: str,
        task_prompt: str,
        task_input: Any = None,
        parent_task_output: Any = None,
        task_tools: list[ToolConfig] | None = None,
    ) -> None:
        self.flow_ctx           = flow_ctx
        self.task_name          = task_name
        self.task_prompt        = task_prompt
        self.task_input         = task_input
        self.parent_task_output = parent_task_output
        # Keyed by step order; populated incrementally as steps finish.
        self.step_results: OrderedDict[int, Any] = OrderedDict()
        # Tools declared on this task (not yet merged with parents).
        self.task_tools: list[ToolConfig] = task_tools or []

    @property
    def global_ctx(self) -> GlobalContext:
        """Shortcut to the ``GlobalContext`` through the flow context."""
        return self.flow_ctx.global_ctx


class StepContext:
    """Context for a single step invocation.

    Passed as the sole argument to every ``@step``-decorated function and to
    every branch handler function.

    Attributes
    ----------
    task_ctx:
        The enclosing ``TaskContext``.
    step_prompt:
        The ``prompt`` field of the ``@step`` decorator.
    input:
        The validated input for this step.  For the first step in a task this
        is the task's input; for subsequent steps it is the output of the
        previous step.
    order:
        The ``order`` number declared on the ``@step`` decorator.
    previous_results:
        Snapshot of ``task_ctx.step_results`` at the moment this step starts.
        Allows a step to inspect the outputs of earlier steps without
        modifying the live accumulator.
    condition_value:
        Populated by the runner when this is a branching step.  Holds the
        raw value extracted from the input for the discriminator field.
        ``None`` for non-branching steps.
    selected_branch:
        Populated by the runner when this is a branching step.  Holds the
        key of the handler that was selected (e.g. ``"csv"`` or
        ``"__fallback__"``).  ``""`` (empty string) for non-branching steps.
    """

    def __init__(
        self,
        task_ctx: TaskContext,
        step_prompt: str,
        step_input: Any = None,
        order: int = 0,
        step_tools: list[ToolConfig] | None = None,
        output_schema: type | None = None,
    ) -> None:
        self.task_ctx      = task_ctx
        self.step_prompt   = step_prompt
        self.input         = step_input
        self.order         = order

        # Snapshot of accumulated step results at the time this step starts.
        self.previous_results: dict[int, Any] = dict(task_ctx.step_results)

        # Branch-dispatching fields — populated by StepRunner when is_branch.
        self.condition_value: Any = None
        self.selected_branch: str = ""

        # Tools declared on this step (not yet merged with parents).
        self.step_tools: list[ToolConfig] = step_tools or []

        # Output schema for structured LLM output (Pydantic BaseModel or None).
        self.output_schema: type | None = output_schema

    @property
    def flow_ctx(self) -> FlowContext:
        """Shortcut to the enclosing ``FlowContext``."""
        return self.task_ctx.flow_ctx

    @property
    def global_ctx(self) -> GlobalContext:
        """Shortcut to the shared ``GlobalContext``."""
        return self.task_ctx.global_ctx

    @property
    def tools(self) -> ToolRegistry:
        """Registry of all tools available to this step."""
        return self.global_ctx.tool_registry

    @property
    def llm_config(self) -> LLMConfig:
        """Default LLM configuration for this run."""
        return self.global_ctx.llm_config

    @property
    def merged_tools(self) -> list[ToolConfig]:
        """All tools available to this step, merged from global → flow → task → step.

        Tools are accumulated hierarchically:
        - Global tools (from ``@global_config``)
        - Flow tools (from the enclosing ``@flow``)
        - Task tools (from the enclosing ``@task``)
        - Step tools (from this ``@step``)

        Later levels can shadow earlier ones (same tool name = override).
        """
        from flowforge.types import MCPServer, FunctionTool, HTTPTool

        # Collect in order: global → flow → task → step
        all_tools: list[ToolConfig] = []

        # Global tools from GlobalMeta (stored in tool_registry's source configs)
        # We access them via the global_ctx since they were registered at compile.
        # But we also want the raw ToolConfig list for the LLM call.
        # global_ctx stores global tools already; flow/task/step carry theirs.
        all_tools.extend(self.global_ctx.global_tools)
        all_tools.extend(self.flow_ctx.flow_tools)
        all_tools.extend(self.task_ctx.task_tools)
        all_tools.extend(self.step_tools)
        return all_tools

    def _resolve_tool_configs(self, tool_names: list[str]) -> list[ToolConfig]:
        """Resolve tool names referenced via ``<tool_name>`` in a prompt.

        Searches ``merged_tools`` for configs whose name matches. If a name
        is not found, it is silently skipped (the LLM may still know about
        the tool from the global registry).
        """
        from flowforge.types import MCPServer, FunctionTool, HTTPTool

        merged = self.merged_tools
        result: list[ToolConfig] = []
        for name in tool_names:
            for tc in merged:
                tc_name = ""
                if isinstance(tc, MCPServer):
                    tc_name = tc.name
                elif isinstance(tc, FunctionTool):
                    tc_name = tc.name or (tc.func.__name__ if hasattr(tc.func, '__name__') else "")
                elif isinstance(tc, HTTPTool):
                    tc_name = tc.name
                if tc_name == name:
                    result.append(tc)
                    break
        return result

    async def call_llm(self, prompt: str) -> Any:
        """Call the LLM with a templated user prompt.

        The annotation ``prompt`` (``self.step_prompt``) is used as the
        **system prompt**.  The *prompt* argument to this method is the
        **user/task prompt** sent as the user message.

        Template syntax
        ~~~~~~~~~~~~~~~
        * ``{field_name}`` — replaced with the value of ``self.input.field_name``
          (Pydantic model attribute) or ``self.input["field_name"]`` (dict).
        * ``<tool_name>`` — marks that the named tool should be included in
          the LLM ``tools`` parameter for this call.  The ``<...>`` marker is
          removed from the final prompt text.

        Returns
        -------
        Any
            The LLM response content (text string, or parsed structured output).

        Raises
        ------
        ExecutionError
            On LLM call failure.
        """
        from flowforge.execution.llm import render_prompt, parse_tool_refs, call_llm_api

        # 1. Parse <tool_name> references and strip them from prompt text.
        clean_prompt, tool_names = parse_tool_refs(prompt)

        # 2. Template {var} with input fields.
        rendered = render_prompt(clean_prompt, self.input)

        # 3. Resolve tool configs for referenced tools.
        tool_configs = self._resolve_tool_configs(tool_names)

        # 4. Call LLM (pass output_schema for structured output).
        return await call_llm_api(
            system_prompt=self.step_prompt,
            user_prompt=rendered,
            llm_config=self.llm_config,
            tool_configs=tool_configs,
            tool_registry=self.tools,
            output_schema=self.output_schema,
        )
