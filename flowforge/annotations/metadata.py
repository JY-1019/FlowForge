"""Metadata dataclasses for FlowForge annotations.

Each dataclass mirrors one FlowForge decorator:

    @global_config  →  GlobalMeta
    @flow           →  FlowMeta
    @task           →  TaskMeta
    @step           →  StepMeta

Branch behaviour is NOT a separate annotation type.  Instead, ``@step``,
``@task``, and ``@flow`` all accept optional ``condition`` / ``branches`` /
``fallback`` parameters.  When those parameters are supplied the node acts as
a *branch dispatcher*: at runtime it evaluates ``condition.field`` on its
input, looks up the matching handler / child node in ``branches``, and
forwards execution there.

Design
------
- ``StepMeta.is_branch``  → True when ``condition`` is set on a ``@step``
- ``TaskMeta.is_branch``  → True when ``condition`` is set on a ``@task``
- ``FlowMeta.is_branch``  → True when ``condition`` is set on a ``@flow``

Leaf vs. container distinction (Tasks)
---------------------------------------
A *leaf* task owns ``@step`` nodes directly.
A *container* task owns child ``@task`` classes.
A *branch* task is a pure dispatcher: it selects one of the tasks listed in
``branches`` and delegates execution to it; its own ``steps`` / ``child_tasks``
lists are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.types import BranchCondition, LLMConfig, ToolReference

# Type alias for loop condition callable: (output) -> bool
# Returns True when the output is acceptable (stop looping).
LoopCondition = Any  # Callable[[Any], bool] — uses Any to avoid import issues


@dataclass
class StepMeta:
    """Metadata attached to a ``@step``-decorated function.

    Parameters
    ----------
    order:
        Execution position within the parent leaf task.  Multiple steps may
        share the same ``order`` number — they execute **in parallel** and
        each receives the same input.  The output of the last step in a
        parallel group is forwarded to the next order group.
    prompt:
        Natural-language instruction passed to the LLM when this step runs.
    func:
        The async function decorated with ``@step``.
    input_schema:
        Optional Pydantic model for the step's expected input.  When ``None``
        the output of the previous order group is passed through unchanged.
    output_schema:
        Optional Pydantic model for the step's return value.  When set the
        return value is coerced / validated before being passed downstream.
    tool_mode:
        When ``True`` the step is registered as an LLM tool-use endpoint and
        is invoked dynamically by the AI planner rather than in fixed order.
    timeout_seconds:
        Maximum wall-clock seconds this step may run before raising a timeout.
    unique:
        When ``True`` this step is the **exclusive** representative of its
        ``order`` group: all other steps with the same ``order`` inside the
        same task are skipped, and only this step runs.

        Rules
        -----
        * At most **one** step per order group may have ``unique=True``.
          Declaring two steps with the same ``order`` and ``unique=True``
          raises ``OrderConflictError`` at decoration time.
        * Scoped to the **same parent task** — ``unique=True`` on a step in
          task A does not affect steps with the same ``order`` in task B.

    Branch parameters (all optional)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``condition`` is supplied the step becomes a *branch dispatcher*:
    instead of running ``func`` directly it evaluates ``condition.field`` on
    the step input, resolves a handler from ``branches``, and calls it.

    condition:
        A ``BranchCondition`` that names the discriminator field and its
        allowed enum values.
    branches:
        Mapping from condition-value string → async handler callable.  The
        handler receives a ``StepContext`` and must return the same type as
        every other handler (enforced at decoration time).
    fallback:
        Handler called when the resolved condition value is not found in
        ``branches``.  When omitted and no branch matches, ``func`` itself
        is called as the last resort.
    """

    order: int
    prompt: str
    func: Callable[..., Any]
    input_schema: type | None = None
    output_schema: type | None = None
    tool_mode: bool = False
    timeout_seconds: int = 60

    # When True, this step is the sole runner in its order group (others skip).
    unique: bool = False

    # When True, execution pauses before this step and raises ApprovalRequired.
    # The caller must approve and resume using the checkpoint mechanism.
    approval: bool = False

    # Natural-language criteria the step output must satisfy.  When set, an LLM
    # judge evaluates the output after each execution.  If the judge says the
    # output fails the criteria, the step is re-run with feedback about what was
    # wrong (up to ``pass_criteria_max_retries`` times).
    pass_criteria: str | None = None
    pass_criteria_max_retries: int = 3

    # Tools available to this step (merged with parent task/flow/global tools).
    tools: list[ToolReference] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Branch dispatcher fields (optional — only set when condition is given)
    # ------------------------------------------------------------------

    # The BranchCondition that controls which handler is selected at runtime.
    condition: BranchCondition | None = field(default=None)

    # key → async callable that handles the matched condition value.
    branches: dict[str, Callable[..., Any]] = field(default_factory=dict)

    # Called when no key in `branches` matches the condition value.
    fallback: Callable[..., Any] | None = None

    @property
    def is_branch(self) -> bool:
        """``True`` when this step acts as a branch dispatcher.

        A step is a branch dispatcher when a ``condition`` has been supplied.
        Non-branch steps simply execute ``func`` in order.
        """
        return self.condition is not None


@dataclass
class TaskMeta:
    """Metadata attached to a ``@task``-decorated class.

    A task is one of three *kinds*:

    1. **Leaf task** (``is_leaf == True``) — owns ``@step`` nodes directly.
       Steps with the same ``order`` execute in parallel; different ``order``
       values execute sequentially.
    2. **Container task** (``is_leaf == False``, ``is_branch == False``) —
       owns child ``@task`` classes, executed according to their ``order``
       values.
    3. **Branch task** (``is_branch == True``) — pure dispatcher; evaluates
       ``condition.field`` on its input and delegates to one of the task
       instances listed in ``branches``.

    Parameters
    ----------
    name:
        Unique identifier used in DAG node IDs and ``depends_on`` references.
    prompt:
        Natural-language description passed to the LLM context when this task
        executes.
    cls:
        The class object decorated with ``@task``.
    input_schema / output_schema:
        Optional Pydantic models for I/O validation at the task boundary.
    steps:
        ``StepMeta`` nodes belonging to this leaf task.  Steps sharing the
        same ``order`` run in parallel with the same input.
    child_tasks:
        Child ``TaskMeta`` instances (container tasks only).  Tasks sharing
        the same ``order`` run in parallel.
    order:
        Execution position of this task within its parent flow.  Tasks
        sharing the same ``order`` inside the same parent flow run
        **in parallel** and each receives the same input.  When ``None``
        (default) the task is assigned a unique sequential order at runtime,
        preserving insertion-order execution (backward-compatible default).
    unique:
        When ``True`` this task is the **exclusive** representative of its
        ``order`` group within the parent flow: all sibling tasks with the
        same ``order`` are skipped.  At most one task per order group per
        parent may carry ``unique=True``; a duplicate raises
        ``OrderConflictError`` at decoration time.
        Scoped to the same parent flow — does not affect tasks with the
        same ``order`` in different parent flows.

    Branch parameters (all optional)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    condition:
        ``BranchCondition`` that identifies the discriminator field.
    branches:
        Mapping from condition-value string → ``TaskMeta`` of the task to
        execute for that value.
    fallback:
        ``TaskMeta`` used when no key in ``branches`` matches.
    """

    name: str
    prompt: str
    cls: type
    input_schema: type | None = None
    output_schema: type | None = None

    # Steps owned by this task (populated only for leaf tasks).
    steps: list[StepMeta] = field(default_factory=list)

    # Child tasks (populated only for container tasks).
    child_tasks: list[TaskMeta] = field(default_factory=list)

    # Execution position within the parent flow (None = auto-sequential).
    order: int | None = None

    # When True, this task is the sole runner in its order group (others skip).
    unique: bool = False

    # Tools available to this task and all its child steps/tasks.
    tools: list[ToolReference] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    # Controls what happens when a step inside this task raises an error.
    #
    #  "raise"           (default) — propagate the error immediately.
    #  "skip_remaining"  — stop executing further steps in this task and
    #                      return the last successful step output.  If the
    #                      *first* step fails the error is still propagated.
    on_error: str = "raise"

    # ------------------------------------------------------------------
    # Loop fields (optional)
    # ------------------------------------------------------------------

    # Maximum number of times to re-run this task's step chain when the
    # loop_condition returns False.  1 = no loop (default), 3 = up to 3 attempts.
    max_loops: int = 1

    # Callable (output) -> bool.  Called after each execution.
    # Returns True → accept result, stop looping.
    # Returns False → discard result, re-run from step 1.
    # None → no loop regardless of max_loops.
    loop_condition: LoopCondition | None = None

    # ------------------------------------------------------------------
    # Branch dispatcher fields (optional)
    # ------------------------------------------------------------------

    # The BranchCondition that controls which branch task is executed.
    condition: BranchCondition | None = field(default=None)

    # key → TaskMeta of the task that handles the matched condition value.
    branches: dict[str, TaskMeta] = field(default_factory=dict)

    # TaskMeta used when no key matches the condition value.
    fallback: TaskMeta | None = None

    @property
    def is_leaf(self) -> bool:
        """``True`` when this task directly contains ``@step`` nodes.

        A task is a leaf when it has no child tasks **and** is not a branch
        dispatcher.  Only leaf tasks run their ``steps`` list.
        """
        return len(self.child_tasks) == 0 and not self.is_branch

    @property
    def is_branch(self) -> bool:
        """``True`` when this task is a branch dispatcher.

        A branch task selects one child task at runtime based on
        ``condition.field`` and delegates all execution to it.
        """
        return self.condition is not None


@dataclass
class FlowMeta:
    """Metadata attached to a ``@flow``-decorated class.

    A flow is a top-level execution unit that may contain:

    * **Child flows** — sub-flows executed (by their ``order``) before or
      alongside the flow's own tasks.
    * **Tasks** — executed according to their ``order`` values after all
      child flows at the same order have finished.
    * **Branch targets** — when ``condition`` is set the flow selects one of
      the ``FlowMeta`` instances in ``branches`` and delegates entirely to it.

    Same-order execution
    --------------------
    Child flows (and tasks) that share the same ``order`` value run
    **in parallel** — each receives the same input as the group, and the
    output of the **last** member is forwarded to the next order group.

    When ``order=None`` (default) the node is assigned a unique sequential
    order at runtime, preserving insertion-order execution.

    Parameters
    ----------
    name:
        Unique identifier used in DAG node IDs and ``depends_on`` references.
    prompt:
        Natural-language description of this flow's purpose.
    cls:
        The class object decorated with ``@flow``.
    input_schema / output_schema:
        Optional Pydantic models for I/O validation at the flow boundary.
    depends_on:
        List of flow names that must complete before this flow starts.
        Used to build dependency edges in the DAG.
    parallel:
        Legacy flag.  When ``True`` all *immediate child flows* run
        concurrently regardless of their individual ``order`` values.
        Prefer explicit same-``order`` sibling flows for fine-grained
        parallelism control.
    max_retries:
        Number of times the flow will be retried on ``ExecutionError`` before
        propagating the exception.
    order:
        Execution position of this flow within its parent (another flow or
        ``@global_config``).  Flows sharing the same ``order`` inside the same
        parent run **in parallel** with the same input.
        ``None`` (default) → auto-sequential (backward-compatible).
    unique:
        When ``True`` this flow is the **exclusive** representative of its
        ``order`` group within its parent: all sibling flows at the same
        ``order`` are skipped.  At most one flow per order group per parent
        may carry ``unique=True``.
        Scoped to the same parent — does not affect flows with the same
        ``order`` in different parents.

    Branch parameters (all optional)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    condition:
        ``BranchCondition`` that identifies the discriminator field on the
        flow input.
    branches:
        Mapping from condition-value string → ``FlowMeta`` of the flow to
        execute for that value.
    fallback:
        ``FlowMeta`` used when no key in ``branches`` matches.
    """

    name: str
    prompt: str
    cls: type
    input_schema: type | None = None
    output_schema: type | None = None
    depends_on: list[str] = field(default_factory=list)
    parallel: bool = False
    max_retries: int = 3

    # Child flows nested inside this flow.
    child_flows: list[FlowMeta] = field(default_factory=list)

    # Tasks owned by this flow.
    tasks: list[TaskMeta] = field(default_factory=list)

    # Execution position within the parent (None = auto-sequential).
    order: int | None = None

    # When True, this flow is the sole runner in its order group (others skip).
    unique: bool = False

    # Tools available to this flow and all its child tasks/steps/flows.
    tools: list[ToolReference] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Branch dispatcher fields (optional)
    # ------------------------------------------------------------------

    # The BranchCondition that controls which branch flow is executed.
    condition: BranchCondition | None = field(default=None)

    # key → FlowMeta of the flow that handles the matched condition value.
    branches: dict[str, FlowMeta] = field(default_factory=dict)

    # FlowMeta used when no key matches the condition value.
    fallback: FlowMeta | None = None

    # Optional runtime self-repair hook for dynamically generated flows.
    # Signature: ``async def repair(error: str) -> bool``.  Returns True
    # when the generated code has been regenerated and the meta mutated in
    # place (so the next retry runs the fixed version).  FlowRunner calls
    # it on ``ExecutionError`` before each retry attempt.
    runtime_repair: Callable[..., Any] | None = None

    @property
    def is_branch(self) -> bool:
        """``True`` when this flow is a branch dispatcher.

        A branch flow selects one child flow at runtime based on
        ``condition.field`` and delegates all execution to it.  Its own
        ``child_flows`` and ``tasks`` are ignored.
        """
        return self.condition is not None


@dataclass
class GlobalMeta:
    """Metadata attached to a ``@global_config``-decorated class.

    ``GlobalMeta`` is the root of the entire agent definition.  There is
    exactly one ``GlobalMeta`` per compiled agent.

    Parameters
    ----------
    prompt:
        Global system prompt prepended to every LLM call made by this agent.
    cls:
        The class object decorated with ``@global_config``.
    llm_config:
        Default LLM settings (model, temperature, etc.) used by all nodes
        unless overridden at a lower level.
    tools:
        List of globally registered tools (MCP servers, HTTP adapters, …).
        Tools registered here are accessible from every step/task/flow.
    flows:
        Top-level ``FlowMeta`` instances discovered inside the decorated
        class body.
    """

    prompt: str
    cls: type
    llm_config: LLMConfig = field(default=None)  # type: ignore[assignment]
    tools: list[ToolReference] = field(default_factory=list)
    flows: list[FlowMeta] = field(default_factory=list)

    # When True, the agent can dynamically generate new flows at runtime
    # when no existing flow matches the user's query.  The built-in
    # DynamicFlowGenerator analyses the gap, generates FlowForge decorator
    # code via LLM, compiles it, and injects the new flow into the live DAG.
    dynamic_flow: bool = False

    # When True, inject the framework's builtin tool pack (web, json,
    # files, document tools) into the global tool list at compile time.
    # This makes builtin tools available to all steps via ctx.call_tool().
    include_builtin_tools: bool = False
