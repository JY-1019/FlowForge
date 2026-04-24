"""FlowForge decorator implementations.

Public decorators
-----------------
``@global_config``  — top-level agent class
``@flow``           — execution unit that groups tasks / sub-flows
``@task``           — execution unit that owns steps or child tasks
``@step``           — atomic execution step (optionally a branch dispatcher)

Branch dispatching
------------------
Branch behaviour is built into ``@step``, ``@task``, and ``@flow`` via
optional ``condition`` / ``branches`` / ``fallback`` parameters — there is
**no** separate ``@branch`` decorator.

When ``condition`` is supplied to any of these decorators the node becomes a
*branch dispatcher*:

* ``@step(condition=..., branches={...})``
    Evaluates ``condition.field`` on the step input and calls the matching
    handler function.  Handlers receive a ``StepContext`` and must share the
    same return type.

* ``@task(condition=..., branches={...})``
    Evaluates ``condition.field`` on the task input and delegates execution
    to one of the ``@task``-decorated classes listed in ``branches``.

* ``@flow(condition=..., branches={...})``
    Evaluates ``condition.field`` on the flow input and delegates execution
    to one of the ``@flow``-decorated classes listed in ``branches``.

Decorator processing order
--------------------------
Python processes inner class decorators *before* outer ones, so by the time
``@task`` runs all ``@step`` functions inside the class body already carry
their ``__flowforge_step_meta__`` attribute.  The same applies for
``@flow`` → ``@task`` → ``@step`` nesting.

Example
-------
::

    @flow(
        name="search",
        prompt="route to the right search backend",
        condition=BranchCondition(field="source", enum=["web", "db"]),
        branches={"web": WebSearchFlow, "db": DBSearchFlow},
        fallback=WebSearchFlow,
    )
    class SearchFlow: ...           # body ignored when condition is set

    @task(name="parse", prompt="parse the document")
    class ParseTask:
        @step(order=1, prompt="detect format")
        async def detect(ctx): ...

        @step(
            order=2,
            prompt="route by format",
            condition=BranchCondition(field="fmt", enum=["csv", "json"]),
            branches={"csv": csv_handler, "json": json_handler},
            fallback=csv_handler,
        )
        async def route(ctx): ...

        @step(order=3, prompt="normalise result")
        async def normalise(ctx): ...
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, TYPE_CHECKING

from flowforge.annotations.metadata import (
    StepMeta,
    TaskMeta,
    FlowMeta,
    GlobalMeta,
)
from flowforge.annotations.validators import (
    validate_order_uniqueness,
    validate_io_chain,
    validate_step_branch_handlers,
)
from flowforge.errors import CompileError

if TYPE_CHECKING:
    from flowforge.types import BranchCondition, LLMConfig, ToolConfig

# ---------------------------------------------------------------------------
# Internal sentinel attribute names
#
# These string constants are the attribute names used to attach FlowForge
# metadata to decorated functions and classes.  They are intentionally
# obscure to avoid collisions with user-defined attributes.
# ---------------------------------------------------------------------------

_STEP_ATTR   = "__flowforge_step_meta__"
_TASK_ATTR   = "__flowforge_task_meta__"
_FLOW_ATTR   = "__flowforge_flow_meta__"
_GLOBAL_ATTR = "__flowforge_global_meta__"


# ---------------------------------------------------------------------------
# @step
# ---------------------------------------------------------------------------

def step(
    *,
    order: int,
    prompt: str,
    input_schema: type | None = None,
    output_schema: type | None = None,
    tool_mode: bool = False,
    timeout_seconds: int = 60,
    unique: bool = False,
    approval: bool = False,
    tools: list[ToolConfig] | None = None,
    condition: BranchCondition | None = None,
    branches: dict[str, Callable[..., Any]] | None = None,
    fallback: Callable[..., Any] | None = None,
    pass_criteria: str | None = None,
    pass_criteria_max_retries: int = 3,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark an async function as a step within a leaf task.

    Steps are the atomic execution units of FlowForge.  Inside a leaf task
    they are executed in ascending ``order`` — the output of step *n* becomes
    the input of step *n+1*.

    Branch dispatching
    ~~~~~~~~~~~~~~~~~~
    Passing ``condition`` (and optionally ``branches`` / ``fallback``) turns
    the step into a *branch dispatcher*.  Instead of running ``func``
    directly, the runtime:

    1. Reads ``condition.field`` from the step input.
    2. Looks up the resolved value in ``branches``.
    3. Calls the matching handler (or ``fallback``, or ``func`` itself as last
       resort) with the current ``StepContext``.

    All handlers in ``branches`` **must** share the same return type
    annotation; a ``BranchOutputMismatchError`` is raised at decoration time
    if they differ.

    Parameters
    ----------
    order:
        Execution position within the parent task.  Must be unique across
        all ``@step`` nodes in the same leaf task.
    prompt:
        Natural-language instruction for the LLM when this step runs.
    input_schema:
        Optional Pydantic model for input validation / coercion.
    output_schema:
        Optional Pydantic model for output validation / coercion.
    tool_mode:
        When ``True`` the step is exposed to the AI planner as an LLM tool
        and may be called dynamically out of order.
    timeout_seconds:
        Wall-clock execution limit.  Raises ``asyncio.TimeoutError`` if
        exceeded.
    unique:
        When ``True`` this step is the **sole** representative of its
        ``order`` group within the parent task.  All sibling steps that
        share the same ``order`` are skipped; only this step executes.
        At most one step per order group may carry ``unique=True`` —
        declaring two steps with the same ``order`` and ``unique=True``
        raises ``OrderConflictError`` at decoration time.
    condition:
        ``BranchCondition`` that specifies the discriminator field and its
        allowed values.  Required to enable branch dispatching.
    branches:
        ``{value: handler}`` mapping.  ``handler`` must be an async callable
        that accepts a ``StepContext`` and returns a value.
    fallback:
        Handler called when no key in ``branches`` matches the resolved
        condition value.  Falls back to the decorated ``func`` if omitted.
    pass_criteria:
        Natural-language criteria the step output must satisfy.  When set,
        an LLM judge evaluates the output after execution.  If the output
        fails the criteria, the step is re-run with feedback about what was
        wrong (up to ``pass_criteria_max_retries`` attempts).  After all
        retries are exhausted, the last result is used regardless.
    pass_criteria_max_retries:
        Maximum number of retry attempts when ``pass_criteria`` is set.
        Defaults to 3.

    Returns
    -------
    Callable
        The original ``func`` unchanged (metadata is attached as an
        attribute, not via a wrapper).
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        meta = StepMeta(
            order=order,
            prompt=prompt,
            func=func,
            input_schema=input_schema,
            output_schema=output_schema,
            tool_mode=tool_mode,
            timeout_seconds=timeout_seconds,
            unique=unique,
            approval=approval,
            tools=tools or [],
            condition=condition,
            branches=branches or {},
            fallback=fallback,
            pass_criteria=pass_criteria,
            pass_criteria_max_retries=pass_criteria_max_retries,
        )

        # Validate branch handler return-type consistency at decoration time
        # so errors surface immediately rather than at first runtime call.
        if meta.is_branch:
            validate_step_branch_handlers(meta)

        setattr(func, _STEP_ATTR, meta)
        return func

    return decorator


# ---------------------------------------------------------------------------
# @task
# ---------------------------------------------------------------------------

def task(
    *,
    name: str,
    prompt: str,
    input_schema: type | None = None,
    output_schema: type | None = None,
    order: int | None = None,
    unique: bool = False,
    tools: list[ToolConfig] | None = None,
    on_error: str = "raise",
    max_loops: int = 1,
    loop_condition: Callable[..., bool] | None = None,
    condition: BranchCondition | None = None,
    branches: dict[str, type] | None = None,
    fallback: type | None = None,
) -> Callable[[type], type]:
    """Mark a class as a task, collecting its steps and / or child tasks.

    Tasks can be one of three kinds:

    * **Leaf task** — the class body contains ``@step``-decorated functions
      that are executed in ``order`` sequence.
    * **Container task** — the class body contains ``@task``-decorated inner
      classes that are executed sequentially; the container task itself does
      not hold ``@step`` nodes directly.
    * **Branch task** — ``condition`` is supplied; the class body is ignored
      at runtime.  Instead the task evaluates ``condition.field`` on its input
      and delegates to one of the ``@task``-decorated classes in ``branches``.

    Parameters
    ----------
    name:
        Unique identifier used in DAG node IDs.
    prompt:
        Natural-language description of this task.
    input_schema / output_schema:
        Optional Pydantic models for I/O boundary validation.
    order:
        Execution position of this task within its parent flow.  Tasks
        sharing the same ``order`` inside the same parent flow run
        **in parallel** — each receives the same input and the output of the
        last task in the group is forwarded to the next order group.
        ``None`` (default) preserves insertion-order sequential execution
        (backward-compatible default).
    unique:
        When ``True`` this task is the **sole** representative of its
        ``order`` group within the parent flow.  All sibling tasks with the
        same ``order`` are skipped.  At most one task per order group per
        parent may carry ``unique=True``.
    condition:
        ``BranchCondition`` that identifies the discriminator field on the
        task input.  Turns this task into a branch dispatcher.
    branches:
        ``{value: TaskClass}`` mapping.  Each value must be a class decorated
        with ``@task``.  At runtime the task delegates to the class whose key
        matches ``condition.field`` in the input.
    fallback:
        A ``@task``-decorated class used when no key in ``branches`` matches.

    Raises
    ------
    CompileError
        If any value in ``branches`` (or ``fallback``) is not decorated with
        ``@task``.
    OrderConflictError
        If two ``@step`` nodes with the same ``order`` inside a leaf task
        both have ``unique=True``.
    IOBindingError
        If consecutive ``@step`` nodes have incompatible I/O schemas.
    """
    def decorator(cls: type) -> type:
        if condition is not None:
            # ---------------------------------------------------------------
            # Branch task: resolve TaskMeta from the provided branch classes.
            # The class body is intentionally NOT scanned — a branch task is a
            # pure dispatcher and does not execute its own steps.
            # ---------------------------------------------------------------
            branch_metas: dict[str, TaskMeta] = {}
            for key, branch_cls in (branches or {}).items():
                if not hasattr(branch_cls, _TASK_ATTR):
                    raise CompileError(
                        f"@task '{name}': branches['{key}'] must be decorated "
                        f"with @task, got {branch_cls!r}"
                    )
                branch_metas[key] = getattr(branch_cls, _TASK_ATTR)

            fallback_meta: TaskMeta | None = None
            if fallback is not None:
                if not hasattr(fallback, _TASK_ATTR):
                    raise CompileError(
                        f"@task '{name}': fallback must be decorated with @task, "
                        f"got {fallback!r}"
                    )
                fallback_meta = getattr(fallback, _TASK_ATTR)

            meta = TaskMeta(
                name=name,
                prompt=prompt,
                cls=cls,
                input_schema=input_schema,
                output_schema=output_schema,
                order=order,
                unique=unique,
                tools=tools or [],
                on_error=on_error,
                max_loops=max_loops,
                loop_condition=loop_condition,
                condition=condition,
                branches=branch_metas,
                fallback=fallback_meta,
            )
        else:
            # ---------------------------------------------------------------
            # Normal task (leaf or container): scan the class body for @step
            # functions and nested @task classes.
            # ---------------------------------------------------------------
            steps: list[StepMeta] = []
            child_tasks: list[TaskMeta] = []

            for attr_name, attr_value in cls.__dict__.items():
                # Skip dunder attributes (Python internals).
                if attr_name.startswith("__"):
                    continue

                if hasattr(attr_value, _STEP_ATTR):
                    # Regular or branching @step
                    steps.append(getattr(attr_value, _STEP_ATTR))
                elif inspect.isclass(attr_value) and hasattr(attr_value, _TASK_ATTR):
                    # Nested @task (container task pattern)
                    child_tasks.append(getattr(attr_value, _TASK_ATTR))

            meta = TaskMeta(
                name=name,
                prompt=prompt,
                cls=cls,
                input_schema=input_schema,
                output_schema=output_schema,
                steps=steps,
                child_tasks=child_tasks,
                order=order,
                unique=unique,
                tools=tools or [],
                on_error=on_error,
                max_loops=max_loops,
                loop_condition=loop_condition,
            )

            # Run compile-time validators immediately so errors are reported
            # at the point where the decorator is applied, not at run time.
            validate_order_uniqueness(meta)
            validate_io_chain(meta)

        setattr(cls, _TASK_ATTR, meta)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# @flow
# ---------------------------------------------------------------------------

def flow(
    *,
    name: str,
    prompt: str,
    input_schema: type | None = None,
    output_schema: type | None = None,
    depends_on: list[str] | None = None,
    parallel: bool = False,
    max_retries: int = 3,
    order: int | None = None,
    unique: bool = False,
    tools: list[ToolConfig] | None = None,
    condition: BranchCondition | None = None,
    branches: dict[str, type] | None = None,
    fallback: type | None = None,
) -> Callable[[type], type]:
    """Mark a class as a flow, collecting its child flows and tasks.

    A flow is the top-level organisational unit in FlowForge.  Flows can:

    * Contain **child flows** (nested sub-flows) and/or **tasks**.
    * Declare **dependencies** on other flows via ``depends_on``, which adds
      directed edges in the DAG.
    * Run their child flows in **parallel** (``anyio.create_task_group``).
    * Act as **branch dispatchers** when ``condition`` is supplied.

    Branch dispatching
    ~~~~~~~~~~~~~~~~~~
    When ``condition`` is given the flow class body is ignored at runtime.
    The flow instead evaluates ``condition.field`` on its input and
    delegates *entirely* to one of the ``@flow``-decorated classes listed in
    ``branches``.  The selected branch flow is executed with the same input
    as the branching flow.

    Parameters
    ----------
    name:
        Unique identifier used in DAG node IDs and ``depends_on`` references.
    prompt:
        Natural-language description of this flow's purpose.
    input_schema / output_schema:
        Optional Pydantic models for I/O boundary validation.
    depends_on:
        List of flow names (or full DAG IDs) that must finish before this
        flow starts.  Used to build ``depends_on`` edges in the DAG.
    parallel:
        When ``True`` all immediate child flows execute concurrently.
    max_retries:
        Number of retry attempts on ``ExecutionError`` before propagating.
    order:
        Execution position of this flow within its parent (another flow or
        ``@global_config``).  Flows sharing the same ``order`` inside the
        same parent run **in parallel** — each receives the same input and
        the output of the last flow in the group is forwarded to the next
        order group.  ``None`` (default) preserves insertion-order sequential
        execution (backward-compatible default).
    unique:
        When ``True`` this flow is the **sole** representative of its
        ``order`` group within its parent.  All sibling flows with the same
        ``order`` are skipped.  At most one flow per order group per parent
        may carry ``unique=True``.
    condition:
        ``BranchCondition`` that identifies the discriminator field on the
        flow input.  Turns this flow into a branch dispatcher.
    branches:
        ``{value: FlowClass}`` mapping.  Each value must be a class decorated
        with ``@flow``.
    fallback:
        A ``@flow``-decorated class used when no key in ``branches`` matches.

    Raises
    ------
    CompileError
        If any value in ``branches`` (or ``fallback``) is not decorated with
        ``@flow``.
    """
    def decorator(cls: type) -> type:
        if condition is not None:
            # ---------------------------------------------------------------
            # Branch flow: resolve FlowMeta from the provided branch classes.
            # The class body is NOT scanned — a branch flow is a pure
            # dispatcher and does not execute its own child flows or tasks.
            # ---------------------------------------------------------------
            branch_metas: dict[str, FlowMeta] = {}
            for key, branch_cls in (branches or {}).items():
                if not hasattr(branch_cls, _FLOW_ATTR):
                    raise CompileError(
                        f"@flow '{name}': branches['{key}'] must be decorated "
                        f"with @flow, got {branch_cls!r}"
                    )
                branch_metas[key] = getattr(branch_cls, _FLOW_ATTR)

            fallback_meta: FlowMeta | None = None
            if fallback is not None:
                if not hasattr(fallback, _FLOW_ATTR):
                    raise CompileError(
                        f"@flow '{name}': fallback must be decorated with @flow, "
                        f"got {fallback!r}"
                    )
                fallback_meta = getattr(fallback, _FLOW_ATTR)

            meta = FlowMeta(
                name=name,
                prompt=prompt,
                cls=cls,
                input_schema=input_schema,
                output_schema=output_schema,
                depends_on=depends_on or [],
                parallel=parallel,
                max_retries=max_retries,
                order=order,
                unique=unique,
                tools=tools or [],
                condition=condition,
                branches=branch_metas,
                fallback=fallback_meta,
            )
        else:
            # ---------------------------------------------------------------
            # Normal flow: scan the class body for nested @flow and @task
            # classes.
            # ---------------------------------------------------------------
            child_flows: list[FlowMeta] = []
            tasks: list[TaskMeta] = []

            for attr_name, attr_value in cls.__dict__.items():
                if attr_name.startswith("__"):
                    continue

                if inspect.isclass(attr_value) and hasattr(attr_value, _FLOW_ATTR):
                    child_flows.append(getattr(attr_value, _FLOW_ATTR))
                elif inspect.isclass(attr_value) and hasattr(attr_value, _TASK_ATTR):
                    tasks.append(getattr(attr_value, _TASK_ATTR))

            meta = FlowMeta(
                name=name,
                prompt=prompt,
                cls=cls,
                input_schema=input_schema,
                output_schema=output_schema,
                depends_on=depends_on or [],
                parallel=parallel,
                max_retries=max_retries,
                child_flows=child_flows,
                tasks=tasks,
                order=order,
                unique=unique,
                tools=tools or [],
            )

        setattr(cls, _FLOW_ATTR, meta)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# @global_config
# ---------------------------------------------------------------------------

def global_config(
    *,
    prompt: str,
    llm_config: LLMConfig | None = None,
    tools: list[ToolConfig] | None = None,
    dynamic_flow: bool = False,
) -> Callable[[type], type]:
    """Mark the top-level agent class, collecting all root-level flows.

    There must be exactly **one** ``@global_config``-decorated class per
    agent.  It is the entry point for ``FlowForge.compile()``.

    Parameters
    ----------
    prompt:
        Global system prompt prepended to every LLM call in this agent.
    llm_config:
        Default LLM configuration (model name, temperature, …).  Uses
        framework defaults when omitted.
    tools:
        List of globally registered tools (MCP servers, HTTP adapters, …).
        All steps and tasks in the agent can access these.
    dynamic_flow:
        When ``True``, enables runtime flow generation.  If no existing
        flow matches a user query in autonomous mode, the built-in
        ``DynamicFlowGenerator`` will analyse the gap, generate new
        FlowForge decorator code via LLM, compile it, and inject the
        new flow into the live DAG for execution.

    Notes
    -----
    ``@global_config`` scans the class body for ``@flow``-decorated inner
    classes.  Only direct children are collected; grandchild flows are
    discovered transitively through their parent ``@flow`` metadata.
    """
    from flowforge.types import LLMConfig as _LLMConfig

    def decorator(cls: type) -> type:
        flows: list[FlowMeta] = []

        for attr_name, attr_value in cls.__dict__.items():
            if attr_name.startswith("__"):
                continue
            if inspect.isclass(attr_value) and hasattr(attr_value, _FLOW_ATTR):
                flows.append(getattr(attr_value, _FLOW_ATTR))

        # Fall back to framework defaults when llm_config is not provided.
        _llm_config = llm_config if llm_config is not None else _LLMConfig()

        meta = GlobalMeta(
            prompt=prompt,
            cls=cls,
            llm_config=_llm_config,
            tools=tools or [],
            flows=flows,
            dynamic_flow=dynamic_flow,
        )
        setattr(cls, _GLOBAL_ATTR, meta)
        return cls

    return decorator
