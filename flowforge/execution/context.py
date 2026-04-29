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
    from flowforge.types import LLMConfig, ToolConfig, ToolReference, DynamicRunOptions
    from flowforge.tools.registry import ToolRegistry
    from flowforge.doc.models import AnyDoc
    from flowforge.viz.run_trace import RunTracer, Checkpoint
    from flowforge.execution.memory import SessionMemory


def _tool_ref_name(tool: "ToolReference") -> str:
    """Return the public name for a ToolConfig or string tool reference."""
    if isinstance(tool, str):
        return tool

    from flowforge.types import AgentSkill, ClaudeSkill, FunctionTool, HTTPTool, MCPServer

    if isinstance(tool, MCPServer):
        return tool.name
    if isinstance(tool, FunctionTool):
        return tool.name or (
            tool.func.__name__ if hasattr(tool.func, "__name__") else ""
        )
    if isinstance(tool, HTTPTool):
        return tool.name
    if isinstance(tool, ClaudeSkill):
        return tool.name or tool.skill_id
    if isinstance(tool, AgentSkill):
        return tool.name
    return ""


def _resolve_tool_references(
    raw_tools: list["ToolReference"],
) -> list["ToolConfig"]:
    """Resolve string tool references against concrete configs in scope.

    Generated flows often use ``tools=["web_fetch_url"]`` because they cannot
    reconstruct the original ``FunctionTool`` object.  The concrete config is
    still available from the parent/global tool chain; this helper de-dupes
    and replaces matching string references with that config.
    """
    concrete_by_name: dict[str, ToolConfig] = {}
    for tool in raw_tools:
        if isinstance(tool, str):
            continue
        name = _tool_ref_name(tool)
        if name:
            concrete_by_name[name] = tool

    merged: list[ToolConfig] = []
    seen: set[str] = set()
    for tool in raw_tools:
        resolved = concrete_by_name.get(tool) if isinstance(tool, str) else tool
        if resolved is None:
            continue
        name = _tool_ref_name(resolved)
        if (
            name
            and not isinstance(tool, str)
            and concrete_by_name.get(name) is not resolved
        ):
            continue
        key = name or f"anon:{id(resolved)}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(resolved)
    return merged


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
    shared_data:
        Mutable dictionary shared across **all** flows in a single run.
        Unlike ``FlowContext.flow_state`` (scoped to one flow), this dict
        is accessible from any flow, task, or step via
        ``ctx.global_ctx.shared_data``.  Use it to pass data between
        independent flows that don't have a direct input/output chain.
    session_memory:
        ``SessionMemory`` instance that persists across ``run()`` calls.
        Stores compact summaries of previous runs so the LLM can reference
        earlier context.  Managed by ``ExecutionEngine``.
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
        global_tools: list[ToolReference] | None = None,
        session_memory: SessionMemory | None = None,
        dynamic_options: DynamicRunOptions | None = None,
    ) -> None:
        self.llm_config     = llm_config
        self.global_prompt  = global_prompt
        self.tool_registry  = tool_registry
        self.env_vars: dict[str, str] = env_vars or {}
        self.all_docs: dict[str, AnyDoc] = {}
        self.shared_data: dict[str, Any] = {}
        self.tracer: RunTracer | None = tracer
        # Set by the engine when planning_mode != "deterministic".
        # None means "run everything"; a set means "only run these node IDs".
        self.planned_node_ids: set[str] | None = None
        # Ordered root-flow IDs selected by route/planner.
        # Used only at the top-level execution boundary; nested filtering
        # continues to use ``planned_node_ids``.
        self.planned_root_flow_ids: list[str] | None = None
        # Raw tool configs / string refs from @global_config for hierarchy.
        self.global_tools: list[ToolReference] = global_tools or []
        # Checkpoint for resume support — populated by engine when resume_from is set.
        self.checkpoint: Checkpoint | None = None
        # Session memory — persists across run() calls.
        self.session_memory: SessionMemory | None = session_memory
        # Dynamic generation options for this run.
        self.dynamic_options: DynamicRunOptions | None = dynamic_options


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
        flow_tools: list[ToolReference] | None = None,
    ) -> None:
        self.global_ctx          = global_ctx
        self.flow_name           = flow_name
        self.flow_prompt         = flow_prompt
        self.flow_input          = flow_input
        self.parent_flow_output  = parent_flow_output
        self.flow_state: dict[str, Any] = {}
        self.flow_doc = global_ctx.all_docs.get(f"global.{flow_name}")
        # Tools declared on this flow (not yet merged with parent).
        self.flow_tools: list[ToolReference] = flow_tools or []


class StepResults(OrderedDict):
    """Ordered step outputs keyed by order, with optional name aliases.

    The canonical keys remain integer ``order`` values.  Name aliases are a
    compatibility layer for generated/user code that asks for a prior result
    by step function name, e.g. ``ctx.step_results.get("search_papers")``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name_to_order: dict[str, int] = {}

    def set_result(self, order: int, step_name: str, value: Any) -> None:
        super().__setitem__(order, value)
        if step_name:
            self._name_to_order[step_name] = order

    def snapshot(self) -> "StepResults":
        copied = StepResults()
        for key, value in self.items():
            super(StepResults, copied).__setitem__(key, value)
        copied._name_to_order = dict(self._name_to_order)
        return copied

    def _resolve_key(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._name_to_order.get(key, key)
        return key

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(self._resolve_key(key))

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._resolve_key(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return super().get(self._resolve_key(key), default)


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
        task_tools: list[ToolReference] | None = None,
    ) -> None:
        from flowforge.execution.memory import StepHistory

        self.flow_ctx           = flow_ctx
        self.task_name          = task_name
        self.task_prompt        = task_prompt
        self.task_input         = task_input
        self.parent_task_output = parent_task_output
        # Canonically keyed by step order; also supports step-name lookups.
        self.step_results: StepResults = StepResults()
        # Tools declared on this task (not yet merged with parents).
        self.task_tools: list[ToolReference] = task_tools or []
        # Step-level LLM conversation history within this task.
        self.step_history = StepHistory()

    @property
    def global_ctx(self) -> GlobalContext:
        """Shortcut to the ``GlobalContext`` through the flow context."""
        return self.flow_ctx.global_ctx

    @property
    def shared_data(self) -> dict[str, Any]:
        """Mutable dict shared across all flows in this run.

        Shortcut for ``self.global_ctx.shared_data``.
        """
        return self.global_ctx.shared_data


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
        step_tools: list[ToolReference] | None = None,
        output_schema: type | None = None,
    ) -> None:
        self.task_ctx      = task_ctx
        self.step_prompt   = step_prompt
        self.input         = step_input
        self.order         = order

        # Snapshot of accumulated step results at the time this step starts.
        self.previous_results: StepResults = task_ctx.step_results.snapshot()

        # Branch-dispatching fields — populated by StepRunner when is_branch.
        self.condition_value: Any = None
        self.selected_branch: str = ""

        # Tools declared on this step (not yet merged with parents).
        self.step_tools: list[ToolReference] = step_tools or []

        # Output schema for structured LLM output (Pydantic BaseModel or None).
        self.output_schema: type | None = output_schema

        # Feedback from previous pass_criteria failures (populated by StepRunner
        # on retry).  Step functions can read this to adjust their behavior.
        self._pass_criteria_feedback: list[str] = []

    @property
    def pass_criteria_feedback(self) -> list[str]:
        """List of feedback strings from previous pass_criteria failures.

        Empty on the first attempt.  On retries, contains one entry per
        previous failed attempt describing what the LLM judge found wrong.
        Step functions that use ``ctx.call_llm()`` can include this feedback
        in the prompt to guide the LLM toward a better response.
        """
        return self._pass_criteria_feedback

    @property
    def step_results(self) -> OrderedDict[int, Any]:
        """Backward-compatible alias for prior step outputs in this task.

        Some user-authored or dynamically generated step code refers to
        ``ctx.step_results`` directly. ``TaskContext`` is still the source of
        truth, but exposing it here keeps step handlers ergonomic and avoids
        brittle failures in generated flows.
        """
        return self.task_ctx.step_results

    @property
    def flow_ctx(self) -> FlowContext:
        """Shortcut to the enclosing ``FlowContext``."""
        return self.task_ctx.flow_ctx

    @property
    def task_input(self) -> Any:
        """Original input to the enclosing task.

        ``ctx.input`` is the previous step's output (or the task input for
        the first step).  ``ctx.task_input`` is always the task's original
        input, regardless of step position.  Use this to read fields that
        are set once at task entry (e.g. ``project_dir``, ``target_url``)
        from any step.
        """
        return self.task_ctx.task_input

    @property
    def flow_input(self) -> Any:
        """Original input to the enclosing flow."""
        return self.task_ctx.flow_ctx.flow_input

    @property
    def global_ctx(self) -> GlobalContext:
        """Shortcut to the shared ``GlobalContext``."""
        return self.task_ctx.global_ctx

    @property
    def shared_data(self) -> dict[str, Any]:
        """Mutable dict shared across all flows in this run.

        Shortcut for ``self.global_ctx.shared_data``.  Use this to exchange
        data between independent flows that don't share an input/output chain.
        """
        return self.global_ctx.shared_data

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
        # Collect in order: global → flow → task → step
        all_tools: list[ToolReference] = []

        # Global tools from GlobalMeta (stored in tool_registry's source configs)
        # We access them via the global_ctx since they were registered at compile.
        # But we also want the raw ToolConfig list for the LLM call.
        # global_ctx stores global tools already; flow/task/step carry theirs.
        all_tools.extend(self.global_ctx.global_tools)
        all_tools.extend(self.flow_ctx.flow_tools)
        all_tools.extend(self.task_ctx.task_tools)
        all_tools.extend(self.step_tools)
        return _resolve_tool_references(all_tools)

    def _resolve_tool_configs(self, tool_names: list[str]) -> list[ToolConfig]:
        """Resolve tool names referenced via ``<tool_name>`` in a prompt.

        Each unresolved name is logged at WARNING level so that hallucinated
        tool/skill markers (e.g. an Agent Skill name the LLM invented from
        the flow name) do not silently degrade the runtime call into a
        toolless prompt.
        """
        import logging as _logging

        result: list[ToolConfig] = []
        unresolved: list[str] = []
        for name in tool_names:
            matched = False
            for tc in self.merged_tools:
                tc_name = _tool_ref_name(tc)
                if tc_name == name:
                    result.append(tc)
                    matched = True
                    break
            if not matched:
                unresolved.append(name)

        if unresolved:
            available = sorted(
                {_tool_ref_name(tc) for tc in self.merged_tools}
                - {""}
            )
            _logging.getLogger(__name__).warning(
                "call_llm: unresolved tool/skill markers %s — dropped from "
                "the request. Registered names: %s",
                unresolved,
                available or "(none)",
            )
        return result

    async def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Call a tool by name from the merged tool hierarchy.

        Searches the merged tool chain (global -> flow -> task -> step) for a
        ``FunctionTool`` matching *tool_name* and invokes its underlying
        function with the provided keyword arguments.

        Parameters
        ----------
        tool_name:
            The name of the tool to call (e.g. ``"pptx_create"``).
        **kwargs:
            Arguments passed through to the tool function.

        Returns
        -------
        Any
            The tool function's return value.

        Raises
        ------
        ValueError
            If the tool is not found.
        """
        import inspect
        from flowforge.types import FunctionTool as _FT
        from flowforge.execution.tool_result import wrap_tool_result

        for tc in self.merged_tools:
            if isinstance(tc, _FT) and tc.name == tool_name:
                call_kwargs = dict(kwargs)
                try:
                    sig = inspect.signature(tc.func)
                    if "ctx" in sig.parameters and "ctx" not in call_kwargs:
                        call_kwargs["ctx"] = self
                except (TypeError, ValueError):
                    pass
                if inspect.iscoroutinefunction(tc.func):
                    raw = await tc.func(**call_kwargs)
                else:
                    raw = tc.func(**call_kwargs)
                return wrap_tool_result(raw)
        available = [
            tc.name for tc in self.merged_tools if isinstance(tc, _FT)
        ]
        raise ValueError(
            f"Tool {tool_name!r} not found. Available FunctionTools: {available}"
        )

    async def call_skill(
        self,
        skill_name: str,
        prompt: str,
        *,
        timeout: int = 300,
        model: str = "",
        max_tokens: int = 4096,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Call a Claude Code skill (slash command) from within a step.

        This method invokes the ``claude`` CLI with a skill prompt and
        returns the result.  It enables FlowForge steps to leverage
        Claude Code's built-in skills (e.g. ``/commit``, ``/review-pr``,
        code generation, etc.) or any custom skills installed in the
        user's environment.

        Parameters
        ----------
        skill_name:
            The skill name to invoke (e.g. ``"commit"``, ``"review-pr"``,
            ``"pdf"``).  This is equivalent to typing ``/skill_name`` in
            Claude Code.
        prompt:
            The prompt or arguments to pass to the skill.
        timeout:
            Maximum execution time in seconds (default: 300).
        model:
            Model override (e.g. ``"sonnet"``).  Uses the default if empty.
        max_tokens:
            Maximum tokens for the response.
        cwd:
            Working directory for the CLI invocation.  Defaults to the
            dynamic options ``project_root`` if available, otherwise ``cwd()``.

        Returns
        -------
        dict
            ``{"ok": True, "result": "...", "skill": "..."}`` on success,
            ``{"ok": False, "error": "...", "skill": "..."}`` on failure.

        Examples
        --------
        ```python
        @step(order=1, prompt="코드 리뷰")
        async def review(ctx):
            result = await ctx.call_skill("review-pr", "PR #42를 리뷰해줘")
            return result
        ```
        """
        import asyncio
        import shutil

        # Find claude CLI
        claude_bin = shutil.which("claude")
        if not claude_bin:
            return {
                "ok": False,
                "error": (
                    "Claude CLI ('claude') not found in PATH. "
                    "Install Claude Code first: https://docs.anthropic.com/en/docs/claude-code"
                ),
                "skill": skill_name,
            }

        # Build the full prompt with skill invocation
        full_prompt = f"/{skill_name} {prompt}"

        cmd = [claude_bin, "--print"]
        if model:
            cmd.extend(["--model", model])
        cmd.append(full_prompt)

        # Determine working directory
        run_cwd = cwd
        if not run_cwd:
            dyn_opts = getattr(self.global_ctx, "dynamic_options", None)
            if dyn_opts and dyn_opts.project_root:
                run_cwd = dyn_opts.project_root

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=run_cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )

            output = stdout.decode("utf-8", errors="replace").strip()
            err_output = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode == 0:
                return {
                    "ok": True,
                    "result": output,
                    "skill": skill_name,
                }
            else:
                return {
                    "ok": False,
                    "error": err_output or f"claude exited with code {proc.returncode}",
                    "result": output,
                    "skill": skill_name,
                }
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "error": f"Skill '{skill_name}' timed out after {timeout}s",
                "skill": skill_name,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to invoke skill: {e}",
                "skill": skill_name,
            }

    async def call_llm(
        self,
        prompt: str,
        *,
        stream: bool = False,
        max_tool_rounds: int | None = None,
    ) -> Any:
        """Call the LLM with a templated user prompt.

        The annotation ``prompt`` (``self.step_prompt``) is used as the
        **system prompt**.  The *prompt* argument to this method is the
        **user/task prompt** sent as the user message.

        Memory injection
        ~~~~~~~~~~~~~~~~
        Two levels of memory are automatically injected into the system prompt:

        * **Session memory** — summaries of previous ``run()`` calls (if any).
        * **Step history** — summaries of earlier steps in this task (if any).

        This gives the LLM context about what happened before without
        replaying full conversations.

        Template syntax
        ~~~~~~~~~~~~~~~
        * ``{field_name}`` — replaced with the value of ``self.input.field_name``
          (Pydantic model attribute) or ``self.input["field_name"]`` (dict).
        * ``<tool_name>`` — marks that the named tool should be included in
          the LLM ``tools`` parameter for this call.  The ``<...>`` marker is
          removed from the final prompt text.

        Parameters
        ----------
        prompt:
            The user/task prompt (supports ``{var}`` and ``<tool>`` syntax).
        stream:
            When ``True``, return an **async generator** that yields ``str``
            chunks as the model produces them.  Streaming mode does not
            support tool-use loops or structured output.

        Returns
        -------
        Any
            The LLM response content (text string, parsed structured output,
            or async generator when ``stream=True``).

        Raises
        ------
        ExecutionError
            On LLM call failure.
        """
        from flowforge.execution.llm import (
            call_llm_api,
            find_tool_refs,
            is_html_tag_name,
            parse_tool_refs,
            render_prompt,
        )

        available_tool_names = sorted(
            {_tool_ref_name(tc) for tc in self.merged_tools} - {""}
        )

        # 1. Parse <tool_name> references and strip only registered tools from
        # prompt text.  Unknown angle-bracket text is preserved so code prompts
        # containing literal HTML like <div> or <button> remain intact.
        clean_prompt, tool_names = parse_tool_refs(
            prompt,
            known_names=available_tool_names,
        )
        unknown_tool_like = [
            name
            for name in find_tool_refs(prompt)
            if name not in available_tool_names and not is_html_tag_name(name)
        ]
        if unknown_tool_like:
            self._resolve_tool_configs(unknown_tool_like)

        # 2. Template {var} with input fields.
        rendered = render_prompt(clean_prompt, self.input)

        # 3. Resolve tool configs for referenced tools.
        tool_configs = self._resolve_tool_configs(tool_names)

        # 4. Build system prompt with memory context.
        system_prompt = self._build_system_prompt()

        # 5. Streaming mode — return async generator directly.
        if stream:
            from flowforge.execution.llm import stream_llm_api
            return stream_llm_api(
                system_prompt=system_prompt,
                user_prompt=rendered,
                llm_config=self.llm_config,
                tool_configs=tool_configs,
                tool_registry=self.tools,
            )

        # 6. Normal mode — call LLM (pass output_schema for structured output).
        result = await call_llm_api(
            system_prompt=system_prompt,
            user_prompt=rendered,
            llm_config=self.llm_config,
            tool_configs=tool_configs,
            tool_registry=self.tools,
            output_schema=self.output_schema,
            max_tool_rounds=max_tool_rounds,
        )

        # 7. Record this step's interaction in the task's step history.
        step_name = ""
        if hasattr(self, "_step_func_name"):
            step_name = self._step_func_name
        self.task_ctx.step_history.record_step(
            step_name=step_name,
            order=self.order,
            prompt=rendered,
            response=result,
        )

        return result

    def _build_system_prompt(self) -> str:
        """Build the full system prompt with memory blocks appended."""
        parts = [self.step_prompt]

        # Inject session memory (previous runs).
        session_mem = self.global_ctx.session_memory
        if session_mem:
            block = session_mem.to_prompt_block()
            if block:
                parts.append(block)

        # Inject step history (earlier steps in this task).
        step_hist = self.task_ctx.step_history
        if step_hist:
            block = step_hist.to_prompt_block()
            if block:
                parts.append(block)

        return "\n\n".join(parts)
