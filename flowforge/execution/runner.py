"""Individual runners for Step, Task, and Flow nodes.

Architecture
------------
Each runner is responsible for a single node type and follows the same
four-phase lifecycle:

1. **Input validation** — coerce the incoming data into the node's
   ``input_schema`` (if declared).
2. **Context creation** — build the appropriate ``*Context`` object and
   record a ``start_node`` trace event.
3. **Execution** — run the step function / dispatch to child tasks/flows.
4. **Output validation & storage** — coerce the result into
   ``output_schema``, store in the parent context, record
   ``finish_node`` / ``error_node``.

Branch dispatching
------------------
Branch behaviour is handled *inline* by the normal runners rather than in a
dedicated ``BranchRunner``:

* ``StepRunner``  — checks ``meta.is_branch`` after creating the context.
  When ``True`` it resolves the condition value from the step input and calls
  the matching handler (or the fallback, or ``meta.func`` as last resort).
  Regardless of which callable is invoked the output goes through the same
  validation and storage path.

* ``TaskRunner``  — checks ``meta.is_branch`` before choosing execution mode.
  When ``True`` it delegates to ``_run_branch_task()``, which resolves the
  condition and recursively runs the matching branch ``TaskMeta``.

* ``FlowRunner``  — checks ``meta.is_branch`` inside ``_run_once()``.  When
  ``True`` it delegates to ``_run_branch_flow()``, which resolves the
  condition and recursively runs the matching branch ``FlowMeta``.

DAG node IDs
------------
Each runner constructs its own DAG node ID by appending its name / order to
the ``parent_node_id`` parameter — mirroring exactly the scheme used by
``schema/compiler.py``.  This ensures that trace events reference the correct
node in the compiled DAG.

Retry / timeout
---------------
``FlowRunner`` implements exponential back-off retry around ``_run_once()``.
``StepRunner`` wraps each step in ``run_with_timeout()`` using the step's
``timeout_seconds``.
"""
from __future__ import annotations

import asyncio
import inspect
import json as _json
import logging
import re
from typing import Any

from flowforge.annotations.metadata import (
    StepMeta,
    TaskMeta,
    FlowMeta,
)
from flowforge.execution.context import (
    GlobalContext,
    FlowContext,
    TaskContext,
    StepContext,
)
from flowforge.execution.parallel import run_parallel, run_with_timeout
from flowforge.errors import ExecutionError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order-grouping helper
# ---------------------------------------------------------------------------

def _group_by_order(items: list[Any], get_order: Any) -> list[list[Any]]:
    """Group *items* by their ``order`` value for parallel/sequential execution.

    Rules
    -----
    * Items with an **explicit** integer ``order`` that share the same value
      form one group and will run **in parallel**.
    * Items whose ``order`` is ``None`` each form their own singleton group
      and run **sequentially** after all explicit-order groups, in the order
      they appear in *items* (insertion-order, backward-compatible).

    The returned list of groups is sorted by effective order:
    explicit-order groups first (ascending), then ``None``-order groups in
    insertion order.

    Parameters
    ----------
    items:
        The list of ``StepMeta`` / ``TaskMeta`` / ``FlowMeta`` objects to
        group.
    get_order:
        A callable ``(item) -> int | None`` that extracts the order value.

    Returns
    -------
    list[list[Any]]
        Ordered list of groups.  Each group is a non-empty list of items.
    """
    explicit: dict[int, list[Any]] = {}   # order value → list of items
    none_groups: list[list[Any]] = []      # each None-order item = own group

    for item in items:
        o = get_order(item)
        if o is None:
            none_groups.append([item])
        else:
            explicit.setdefault(o, []).append(item)

    # Sort explicit groups by their order value, then append None-order groups.
    sorted_explicit = [explicit[k] for k in sorted(explicit)]
    return sorted_explicit + none_groups


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_validated(schema: type, result: Any) -> Any:
    """Coerce *result* into *schema* (a Pydantic ``BaseModel`` subclass).

    Handles common input forms:

    * A plain ``dict`` → ``schema.model_validate(result)``
    * A Pydantic model instance → round-trips through ``model_dump()``
    * A ``str`` → attempt JSON parse first (LLM responses are often JSON
      embedded in text), then fall back to ``model_validate``
    * Anything else → passed directly to ``model_validate``

    Parameters
    ----------
    schema:
        Target Pydantic model class.
    result:
        The value to coerce.

    Returns
    -------
    Any
        An instance of *schema*.
    """
    if isinstance(result, dict):
        return schema.model_validate(result)
    if hasattr(result, "model_dump"):
        return schema.model_validate(result.model_dump())
    if isinstance(result, str):
        parsed = _try_parse_json(result)
        if parsed is not None:
            return schema.model_validate(parsed)
    return schema.model_validate(result)


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Try to extract and parse a JSON object from *text*.

    LLM responses often wrap JSON in markdown fences or surrounding prose.
    This function handles:

    1. Pure JSON string (``{ ... }``)
    2. Markdown-fenced JSON (````json\\n{ ... }\\n` ```)
    3. JSON object embedded in surrounding text

    Returns ``None`` if no valid JSON object can be extracted.
    """
    import json

    stripped = text.strip()

    # Fast path: entire string is valid JSON.
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Try markdown code-fence extraction: ```json ... ``` or ``` ... ```
    fence_match = re.search(
        r"```(?:json)?\s*\n?\s*(\{.*?\})\s*\n?\s*```",
        stripped,
        re.DOTALL,
    )
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: find first { ... } block in text.
    brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _get_condition_value(input_data: Any, condition: Any) -> Any:
    """Extract the discriminator field value from *input_data*.

    Supports both ``dict`` inputs and Pydantic model instances (or any object
    with attribute access).

    Parameters
    ----------
    input_data:
        The step / task / flow input containing the discriminator field.
    condition:
        A ``BranchCondition`` instance with a ``field`` attribute naming the
        discriminator field.

    Returns
    -------
    Any
        The resolved field value, or ``None`` when the field is absent or
        when *input_data* is ``None``.
    """
    if input_data is None or condition is None:
        return None
    field = condition.field
    if isinstance(input_data, dict):
        return input_data.get(field)
    return getattr(input_data, field, None)


def _parse_judge_response(response: Any) -> dict[str, Any]:
    """Parse the LLM judge's JSON response into {"pass": bool, "feedback": str}.

    Handles plain dicts, JSON strings, and markdown-fenced JSON.
    """
    if isinstance(response, dict):
        return {"pass": bool(response.get("pass", False)),
                "feedback": response.get("feedback", "")}

    text = str(response)
    parsed = _try_parse_json(text)
    if parsed and "pass" in parsed:
        return {"pass": bool(parsed["pass"]),
                "feedback": parsed.get("feedback", "")}

    # Fallback: if the response contains "true" somewhere, assume pass.
    lower = text.lower()
    if '"pass": true' in lower or '"pass":true' in lower:
        return {"pass": True, "feedback": ""}
    return {"pass": False, "feedback": text[:200]}


# ---------------------------------------------------------------------------
# StepRunner
# ---------------------------------------------------------------------------

class StepRunner:
    """Executes a single ``@step`` node.

    Supports two modes:

    * **Normal step** — calls ``meta.func(ctx)`` directly.
    * **Branching step** (``meta.is_branch == True``) — resolves the
      condition, selects the matching handler (or fallback or ``meta.func``),
      and calls it with the same ``StepContext``.  The ``ctx.selected_branch``
      and ``ctx.condition_value`` fields are populated before the handler call
      so that handlers can inspect them if needed.
    """

    async def run(
        self,
        meta: StepMeta,
        task_ctx: TaskContext,
        step_input: Any,
        parent_node_id: str = "",
    ) -> Any:
        """Execute the step and return its output.

        Parameters
        ----------
        meta:
            The ``StepMeta`` for this step.
        task_ctx:
            The enclosing task context.
        step_input:
            Input data (the output of the previous step, or the task input
            for the first step).
        parent_node_id:
            DAG ID of the parent task node — used to construct this step's
            own node ID and trace events.

        Returns
        -------
        Any
            The step's return value, optionally coerced into
            ``meta.output_schema``.

        Raises
        ------
        ExecutionError
            On input/output validation failure or any uncaught exception
            raised by the step function.
        """
        node_id = f"{parent_node_id}.{meta.func.__name__}[{meta.order}]"
        tracer  = task_ctx.global_ctx.tracer

        # ------------------------------------------------------------------
        # Checkpoint: skip if already completed in a previous run
        # ------------------------------------------------------------------
        cp = task_ctx.global_ctx.checkpoint
        if cp is not None and node_id in cp.completed_node_ids:
            cached = cp.node_outputs.get(node_id)
            task_ctx.step_results.set_result(meta.order, meta.func.__name__, cached)
            if tracer:
                tracer.start_node(node_id, "step", meta.func.__name__, step_input)
                tracer.finish_node(node_id, cached)
            logger.debug("step skip (checkpoint)  node_id=%s", node_id)
            return cached

        # ------------------------------------------------------------------
        # Approval gate: pause execution if this step requires human approval.
        # When resuming from a checkpoint the approval is implicitly granted
        # (the caller already saw the input and chose to resume).
        # ------------------------------------------------------------------
        if meta.approval and cp is None:
            from flowforge.errors import ApprovalRequired
            # Build a checkpoint from the current tracer state so the caller
            # can resume after approving.
            checkpoint = None
            if tracer:
                tracer.start_node(node_id, "step", meta.func.__name__, step_input)
                # Finish the trace up to this point so checkpoint is usable.
                checkpoint = tracer.trace.checkpoint
            raise ApprovalRequired(
                node_id=node_id,
                step_input=step_input,
                checkpoint=checkpoint,
            )

        # ------------------------------------------------------------------
        # Phase 1: input validation
        # ------------------------------------------------------------------
        if meta.input_schema is not None and step_input is not None:
            try:
                step_input = _to_validated(meta.input_schema, step_input)
            except Exception as e:
                err = (
                    f"input validation failed against "
                    f"{meta.input_schema.__name__}: {e}"
                )
                if tracer:
                    tracer.start_node(node_id, "step", meta.func.__name__, step_input)
                    tracer.error_node(node_id, err)
                raise ExecutionError(meta.func.__name__, err) from e

        # ------------------------------------------------------------------
        # Phase 2: context creation
        # ------------------------------------------------------------------
        ctx = StepContext(
            task_ctx=task_ctx,
            step_prompt=meta.prompt,
            step_input=step_input,
            order=meta.order,
            step_tools=meta.tools,
            output_schema=meta.output_schema,
        )
        # Attach function name so step history can identify this step.
        ctx._step_func_name = meta.func.__name__

        if tracer:
            tracer.start_node(node_id, "step", meta.func.__name__, step_input)

        logger.debug(
            "step start  task=%s order=%d func=%s branch=%s",
            task_ctx.task_name, meta.order, meta.func.__name__, meta.is_branch,
        )

        # ------------------------------------------------------------------
        # Phase 3: execution — normal or branching
        # ------------------------------------------------------------------
        try:
            if meta.is_branch:
                result = await self._run_branch(meta, ctx, step_input)
            else:
                coro = meta.func(ctx)
                if not inspect.isawaitable(coro):
                    raise ExecutionError(
                        meta.func.__name__, "step function must be async"
                    )
                result = await run_with_timeout(coro, meta.timeout_seconds)

            # ------------------------------------------------------------------
            # Phase 3.5: pass_criteria — LLM judge + retry loop
            # ------------------------------------------------------------------
            if meta.pass_criteria and not meta.is_branch:
                result = await self._apply_pass_criteria(
                    meta, ctx, step_input, result, node_id,
                )

            # ------------------------------------------------------------------
            # Phase 4: output validation & storage
            # ------------------------------------------------------------------
            if meta.output_schema is not None and result is not None:
                result = _to_validated(meta.output_schema, result)

            task_ctx.step_results.set_result(meta.order, meta.func.__name__, result)

            if tracer:
                tracer.finish_node(
                    node_id,
                    result,
                    # Pass selected_branch so the trace shows which path ran.
                    selected_branch=ctx.selected_branch or None,
                )

            logger.debug(
                "step finish task=%s order=%d selected_branch=%r",
                task_ctx.task_name, meta.order, ctx.selected_branch or None,
            )
            return result

        except ExecutionError as e:
            # Enrich with context if not already set.
            if e.step_input is None:
                e.step_input = step_input
            if node_id not in e.trace_path:
                e.trace_path.insert(0, node_id)
            if tracer:
                tracer.error_node(node_id, str(e))
            raise
        except Exception as e:
            wrapped = ExecutionError(
                meta.func.__name__, str(e),
                step_input=step_input,
                partial_output=task_ctx.step_results.get(meta.order - 1),
            )
            if tracer:
                tracer.error_node(node_id, str(e))
            raise wrapped from e

    async def _apply_pass_criteria(
        self,
        meta: StepMeta,
        ctx: StepContext,
        step_input: Any,
        result: Any,
        node_id: str,
    ) -> Any:
        """Evaluate step output against pass_criteria using an LLM judge.

        If the output fails, re-run the step with feedback about what was
        wrong.  Retries up to ``meta.pass_criteria_max_retries`` times.
        After all retries are exhausted, the last result is returned as-is.
        """
        from flowforge.execution.llm import call_llm_api

        max_retries = meta.pass_criteria_max_retries
        feedbacks: list[str] = []

        for attempt in range(max_retries + 1):  # 0 = initial check, 1..N = retries
            # Ask the LLM judge to evaluate the result.
            result_repr = result
            if hasattr(result, "model_dump"):
                result_repr = result.model_dump()
            elif not isinstance(result, (dict, list, str, int, float, bool)):
                result_repr = str(result)

            judge_prompt = (
                f"You are a strict quality judge.  Evaluate whether the "
                f"following output satisfies the pass criteria.\n\n"
                f"## Pass Criteria\n{meta.pass_criteria}\n\n"
                f"## Step Output\n{_json.dumps(result_repr, ensure_ascii=False, default=str)}\n\n"
            )
            if feedbacks:
                history = "\n".join(
                    f"- Attempt {i+1}: {fb}" for i, fb in enumerate(feedbacks)
                )
                judge_prompt += f"## Previous Feedback\n{history}\n\n"

            judge_prompt += (
                "Respond in JSON: "
                '{"pass": true/false, "feedback": "reason if failed"}'
            )

            try:
                judge_resp = await call_llm_api(
                    system_prompt="You are an output quality evaluator. Always respond in valid JSON.",
                    user_prompt=judge_prompt,
                    llm_config=ctx.llm_config,
                )

                verdict = _parse_judge_response(judge_resp)
            except Exception as e:
                logger.warning(
                    "pass_criteria judge failed node_id=%s attempt=%d: %s",
                    node_id, attempt, e,
                )
                # If judge itself fails, accept the result.
                break

            if verdict["pass"]:
                logger.debug(
                    "pass_criteria PASSED node_id=%s attempt=%d",
                    node_id, attempt,
                )
                break

            feedback = verdict.get("feedback", "criteria not met")
            feedbacks.append(feedback)
            logger.info(
                "pass_criteria FAILED node_id=%s attempt=%d/%d feedback=%r",
                node_id, attempt, max_retries, feedback,
            )

            # If we've exhausted retries, accept the last result.
            if attempt >= max_retries:
                logger.warning(
                    "pass_criteria exhausted retries node_id=%s, using last result",
                    node_id,
                )
                break

            # Re-run the step with feedback injected into context.
            retry_ctx = StepContext(
                task_ctx=ctx.task_ctx,
                step_prompt=meta.prompt,
                step_input=step_input,
                order=meta.order,
                step_tools=meta.tools,
                output_schema=meta.output_schema,
            )
            # Store feedback history so the step function can access it.
            retry_ctx._pass_criteria_feedback = feedbacks.copy()

            coro = meta.func(retry_ctx)
            if inspect.isawaitable(coro):
                result = await run_with_timeout(coro, meta.timeout_seconds)

        return result

    async def _run_branch(
        self,
        meta: StepMeta,
        ctx: StepContext,
        step_input: Any,
    ) -> Any:
        """Dispatch to the appropriate branch handler.

        Resolves ``condition.field`` from *step_input*, looks up the matching
        handler in ``meta.branches``, and calls it with *ctx*.

        Falls back to ``meta.fallback`` when no key matches, or to
        ``meta.func`` itself when neither a matching key nor a fallback is
        configured.

        Parameters
        ----------
        meta:
            The branching ``StepMeta``.
        ctx:
            The ``StepContext`` for this invocation (``condition_value`` and
            ``selected_branch`` are set here before the handler is called).
        step_input:
            Raw step input used to extract the condition value.

        Returns
        -------
        Any
            The return value of the selected handler.
        """
        # Resolve the discriminator field value from the input.
        value = _get_condition_value(step_input, meta.condition)
        ctx.condition_value = value

        # Select the appropriate handler.
        handler = None
        if value is not None and str(value) in meta.branches:
            handler = meta.branches[str(value)]
            ctx.selected_branch = str(value)
            logger.debug(
                "step branch match  func=%s key=%r",
                meta.func.__name__, ctx.selected_branch,
            )
        elif meta.fallback is not None:
            handler = meta.fallback
            ctx.selected_branch = "__fallback__"
            logger.debug(
                "step branch fallback  func=%s cond=%r",
                meta.func.__name__, value,
            )
        else:
            # No match and no fallback — call the decorated function itself.
            logger.debug(
                "step branch no-match, using func  func=%s cond=%r",
                meta.func.__name__, value,
            )

        target = handler if handler is not None else meta.func
        coro   = target(ctx)
        return await run_with_timeout(
            coro if inspect.isawaitable(coro) else _wrap_sync(coro),
            meta.timeout_seconds,
        )


def _wrap_sync(value: Any) -> Any:
    """Return an awaitable that immediately yields *value*.

    Used when a handler returns a non-coroutine (technically unexpected but
    handled gracefully).
    """
    async def _coro() -> Any:
        return value
    return _coro()


# ---------------------------------------------------------------------------
# TaskRunner
# ---------------------------------------------------------------------------

class TaskRunner:
    """Executes a single ``@task`` node.

    Delegates to one of three internal methods depending on the task kind:

    * ``_run_leaf_task``      — runs the step chain.
    * ``_run_container_task`` — runs child tasks sequentially.
    * ``_run_branch_task``    — resolves the condition and runs one branch task.
    """

    def __init__(self) -> None:
        self._step_runner = StepRunner()

    async def run(
        self,
        meta: TaskMeta,
        flow_ctx: FlowContext,
        task_input: Any,
        parent_node_id: str = "",
    ) -> Any:
        """Execute the task and return its output.

        Parameters
        ----------
        meta:
            The ``TaskMeta`` for this task.
        flow_ctx:
            The enclosing flow context.
        task_input:
            Input data for the task.
        parent_node_id:
            DAG ID of the parent node — used to build this task's node ID.

        Returns
        -------
        Any
            The task's final output (last step's output, last child task's
            output, or the selected branch task's output).

        Raises
        ------
        ExecutionError
            On runtime failures or when no branch matches and no fallback is
            configured.
        """
        node_id  = f"{parent_node_id}.{meta.name}"
        tracer   = flow_ctx.global_ctx.tracer
        max_loops = meta.max_loops if meta.loop_condition is not None else 1

        # Checkpoint: skip if already completed in a previous run.
        cp = flow_ctx.global_ctx.checkpoint
        if cp is not None and node_id in cp.completed_node_ids:
            cached = cp.node_outputs.get(node_id)
            if tracer:
                tracer.start_node(node_id, "task", meta.name, task_input)
                tracer.finish_node(node_id, cached)
            logger.debug("task skip (checkpoint)  node_id=%s", node_id)
            return cached

        if tracer:
            tracer.start_node(node_id, "task", meta.name, task_input)
        logger.info(
            "task start  name=%s leaf=%s branch=%s loops=%d",
            meta.name, meta.is_leaf, meta.is_branch, max_loops,
        )

        try:
            result = None
            for loop_idx in range(max_loops):
                # Create a fresh TaskContext per loop iteration so step_results
                # are reset and steps re-run from scratch.
                task_ctx = TaskContext(
                    flow_ctx=flow_ctx,
                    task_name=meta.name,
                    task_prompt=meta.prompt,
                    task_input=task_input,
                    task_tools=meta.tools,
                )

                if meta.is_branch:
                    result = await self._run_branch_task(meta, task_ctx, task_input, node_id)
                elif meta.is_leaf:
                    result = await self._run_leaf_task(meta, task_ctx, task_input, node_id)
                else:
                    result = await self._run_container_task(meta, task_ctx, task_input, node_id)

                # Check loop condition (only when loop_condition is set).
                if meta.loop_condition is None or max_loops <= 1:
                    break  # No loop — accept first result.

                accepted = meta.loop_condition(result)
                if accepted:
                    logger.debug(
                        "task loop accepted  name=%s loop=%d/%d",
                        meta.name, loop_idx + 1, max_loops,
                    )
                    break
                else:
                    logger.info(
                        "task loop retry  name=%s loop=%d/%d",
                        meta.name, loop_idx + 1, max_loops,
                    )
                    # On last attempt, return whatever we got.

            if tracer:
                tracer.finish_node(node_id, result)
            logger.info("task finish name=%s", meta.name)
            return result

        except ExecutionError as e:
            if node_id not in e.trace_path:
                e.trace_path.insert(0, node_id)
            if tracer:
                tracer.error_node(node_id, str(e))
            raise

    async def _run_leaf_task(
        self,
        meta: TaskMeta,
        task_ctx: TaskContext,
        task_input: Any,
        node_id: str,
    ) -> Any:
        """Run steps grouped by order, with parallel execution within each group.

        Execution rules per order group
        --------------------------------
        * **Single step** — runs normally.
        * **Multiple steps, none unique** — all run **in parallel** (same
          input); the output of the *last* step in the group (by list
          position) is forwarded to the next group.
        * **One step with ``unique=True``** — only that step runs; all
          other steps in the same order group are skipped.

        ``None``-order steps are treated as unique sequential groups and
        run after all explicit-order groups.

        Error policy (``meta.on_error``)
        ---------------------------------
        * ``"raise"`` (default) — propagate the error immediately; the
          enclosing flow/retry loop handles it.
        * ``"skip_remaining"`` — catch the error, stop executing further
          step groups, and return the last successful output.  If the
          **first** step group fails (no prior output exists), the error
          is still propagated.

        Returns the output of the last step in the final order group.
        """
        groups        = _group_by_order(meta.steps, lambda s: s.order)
        current_input = task_input
        skip_on_error = meta.on_error == "skip_remaining"

        for group_idx, group in enumerate(groups):
            # Determine which step(s) to actually run.
            unique_steps = [s for s in group if s.unique]
            run_group    = [unique_steps[0]] if unique_steps else group

            try:
                if len(run_group) == 1:
                    # Single step (or unique winner) — run directly.
                    current_input = await self._step_runner.run(
                        run_group[0], task_ctx, current_input, parent_node_id=node_id
                    )
                else:
                    # Multiple steps at the same order → run in parallel.
                    # Every step in the group receives the *same* current_input.
                    coros = [
                        self._step_runner.run(
                            s, task_ctx, current_input, parent_node_id=node_id
                        )
                        for s in run_group
                    ]
                    results       = await run_parallel(coros)
                    current_input = results[-1] if results else current_input
            except (ExecutionError, Exception) as e:
                if skip_on_error and group_idx > 0:
                    # We have a prior successful output — return it instead of
                    # crashing the entire task/flow.
                    step_names = [s.func.__name__ for s in run_group]
                    logger.warning(
                        "step error (on_error=skip_remaining)  "
                        "task=%s  step=%s  skipping remaining steps  error=%s",
                        meta.name, step_names, e,
                    )
                    break
                raise

        return current_input

    async def _run_container_task(
        self,
        meta: TaskMeta,
        task_ctx: TaskContext,
        task_input: Any,
        node_id: str,
    ) -> Any:
        """Run child tasks grouped by order, with parallel execution within each group.

        Execution rules per order group (mirrors the step-level rules):

        * **Single task** — runs normally; its output feeds the next group.
        * **Multiple tasks, none unique** — all run **in parallel** with the
          same input; the output of the *last* task in the group is forwarded.
        * **One task with ``unique=True``** — only that task runs; all other
          tasks in the same order group are skipped.

        Child tasks with ``order=None`` each form their own singleton group
        and execute sequentially after any explicit-order groups, preserving
        insertion-order (backward-compatible default).

        Returns the output of the last child task in the last order group.
        """
        planned       = task_ctx.global_ctx.planned_node_ids
        parent_id     = node_id  # e.g. "global.deep_research.report"
        candidate     = [
            t for t in meta.child_tasks
            if planned is None or f"{parent_id}.{t.name}" in planned
        ]
        groups        = _group_by_order(candidate, lambda t: t.order)
        current_input = task_input

        for group in groups:
            unique_tasks = [t for t in group if t.unique]
            run_group    = [unique_tasks[0]] if unique_tasks else group

            if len(run_group) == 1:
                current_input = await self.run(
                    run_group[0],
                    task_ctx.flow_ctx,
                    current_input,
                    parent_node_id=node_id,
                )
            else:
                # Parallel child tasks: same input to every task in the group.
                coros = [
                    self.run(t, task_ctx.flow_ctx, current_input, parent_node_id=node_id)
                    for t in run_group
                ]
                results       = await run_parallel(coros)
                current_input = results[-1] if results else current_input

        return current_input

    async def _run_branch_task(
        self,
        meta: TaskMeta,
        task_ctx: TaskContext,
        task_input: Any,
        node_id: str,
    ) -> Any:
        """Resolve the condition and delegate to the matching branch task.

        The selected branch task is executed as a full child task with its
        own node ID derived from *node_id*.

        Raises
        ------
        ExecutionError
            When no branch key matches and no fallback is configured.
        """
        value = _get_condition_value(task_input, meta.condition)

        # Select the target TaskMeta.
        target_meta: TaskMeta | None = None
        selected: str = ""

        if value is not None and str(value) in meta.branches:
            target_meta = meta.branches[str(value)]
            selected    = str(value)
        elif meta.fallback is not None:
            target_meta = meta.fallback
            selected    = "__fallback__"

        if target_meta is None:
            raise ExecutionError(
                meta.name,
                f"no branch matched condition value {value!r} and no fallback "
                f"is configured.  Available keys: {list(meta.branches.keys())}",
            )

        logger.debug(
            "task branch selected=%r cond_value=%r task=%s",
            selected, value, meta.name,
        )

        # Delegate execution to the selected branch task.
        return await self.run(
            target_meta,
            task_ctx.flow_ctx,
            task_input,
            parent_node_id=node_id,
        )


# ---------------------------------------------------------------------------
# FlowRunner
# ---------------------------------------------------------------------------

class FlowRunner:
    """Executes a single ``@flow`` node, with retry logic.

    Wraps ``_run_once()`` in an exponential back-off retry loop up to
    ``meta.max_retries`` additional attempts.

    Branch dispatching
    ~~~~~~~~~~~~~~~~~~
    When ``meta.is_branch`` is ``True``, ``_run_once()`` short-circuits and
    calls ``_run_branch_flow()`` instead of executing the flow's own child
    flows and tasks.
    """

    def __init__(self) -> None:
        self._task_runner = TaskRunner()

    async def run(
        self,
        meta: FlowMeta,
        global_ctx: GlobalContext,
        flow_input: Any,
        parent_flow_output: Any = None,
        max_retries: int | None = None,
        parent_node_id: str = "global",
    ) -> Any:
        """Execute the flow (with retry) and return its output.

        Parameters
        ----------
        meta:
            The ``FlowMeta`` for this flow.
        global_ctx:
            Shared agent-run context.
        flow_input:
            Input data for this flow.
        parent_flow_output:
            Output of the parent flow (passed down for nested flows).
        max_retries:
            Override ``meta.max_retries`` when provided.
        parent_node_id:
            DAG ID prefix — ``"global"`` for root flows.

        Returns
        -------
        Any
            The flow's final output.

        Raises
        ------
        ExecutionError
            After all retry attempts are exhausted.
        """
        node_id      = f"{parent_node_id}.{meta.name}"

        # Checkpoint: skip if already completed in a previous run.
        cp = global_ctx.checkpoint
        if cp is not None and node_id in cp.completed_node_ids:
            cached = cp.node_outputs.get(node_id)
            tracer = global_ctx.tracer
            if tracer:
                tracer.start_node(node_id, "flow", meta.name, flow_input)
                tracer.finish_node(node_id, cached)
            logger.debug("flow skip (checkpoint)  node_id=%s", node_id)
            return cached

        _max_retries = max_retries if max_retries is not None else meta.max_retries
        last_exc: Exception | None = None

        for attempt in range(_max_retries + 1):
            try:
                return await self._run_once(
                    meta, global_ctx, flow_input, parent_flow_output, node_id
                )
            except ExecutionError as e:
                last_exc = e
                if attempt < _max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "flow retry  name=%s attempt=%d/%d wait=%ds error=%s",
                        meta.name, attempt + 1, _max_retries, wait, e,
                    )
                    await asyncio.sleep(wait)

        # Preserve trace_path and step_input from the underlying error.
        final_err = ExecutionError(
            meta.name,
            f"failed after {_max_retries + 1} attempt(s)",
        )
        if isinstance(last_exc, ExecutionError):
            final_err.step_input = last_exc.step_input
            final_err.partial_output = last_exc.partial_output
            final_err.trace_path = last_exc.trace_path
            if meta.name not in final_err.trace_path:
                final_err.trace_path.insert(0, meta.name)
        raise final_err from last_exc

    async def _run_once(
        self,
        meta: FlowMeta,
        global_ctx: GlobalContext,
        flow_input: Any,
        parent_flow_output: Any,
        node_id: str,
    ) -> Any:
        """Execute the flow body exactly once (no retry).

        Handles branch dispatching, parallel child flows, sequential child
        flows, and the flow's own task chain.
        """
        tracer = global_ctx.tracer

        flow_ctx = FlowContext(
            global_ctx=global_ctx,
            flow_name=meta.name,
            flow_prompt=meta.prompt,
            flow_input=flow_input,
            parent_flow_output=parent_flow_output,
            flow_tools=meta.tools,
        )

        if tracer:
            tracer.start_node(node_id, "flow", meta.name, flow_input)
        logger.info(
            "flow start  name=%s parallel=%s branch=%s",
            meta.name, meta.parallel, meta.is_branch,
        )

        try:
            if meta.is_branch:
                # Branch dispatcher: select one child flow and run it.
                current_output = await self._run_branch_flow(
                    meta, global_ctx, flow_input, node_id
                )
            else:
                current_output = flow_input

                planned = global_ctx.planned_node_ids

                if meta.child_flows:
                    if meta.parallel:
                        # Legacy ``parallel=True`` flag: run ALL child flows
                        # concurrently regardless of their individual order
                        # values.  Prefer explicit same-order siblings for
                        # fine-grained control.
                        candidate_flows = [
                            cf for cf in meta.child_flows
                            if planned is None or f"{node_id}.{cf.name}" in planned
                        ]
                        coros = [
                            self.run(cf, global_ctx, flow_input, parent_node_id=node_id)
                            for cf in candidate_flows
                        ]
                        results        = await run_parallel(coros)
                        current_output = results[-1] if results else flow_input
                    else:
                        # Order-based grouping: same-order child flows run in
                        # parallel; the output of the last flow in each group
                        # feeds the next group.
                        candidate_flows = [
                            cf for cf in meta.child_flows
                            if planned is None or f"{node_id}.{cf.name}" in planned
                        ]
                        cf_groups = _group_by_order(
                            candidate_flows, lambda f: f.order
                        )
                        for group in cf_groups:
                            unique_flows = [f for f in group if f.unique]
                            run_group    = [unique_flows[0]] if unique_flows else group

                            if len(run_group) == 1:
                                current_output = await self.run(
                                    run_group[0], global_ctx, current_output,
                                    parent_node_id=node_id,
                                )
                            else:
                                # Parallel child flows: every flow in the
                                # group receives the same current_output.
                                coros = [
                                    self.run(
                                        f, global_ctx, current_output,
                                        parent_node_id=node_id,
                                    )
                                    for f in run_group
                                ]
                                results        = await run_parallel(coros)
                                current_output = results[-1] if results else current_output

                # Run the flow's own tasks, also grouped by order, after all
                # child flows complete.
                if meta.tasks:
                    candidate_tasks = [
                        t for t in meta.tasks
                        if planned is None or f"{node_id}.{t.name}" in planned
                    ]
                    task_groups = _group_by_order(candidate_tasks, lambda t: t.order)
                    for group in task_groups:
                        unique_tasks = [t for t in group if t.unique]
                        run_group    = [unique_tasks[0]] if unique_tasks else group

                        if len(run_group) == 1:
                            current_output = await self._task_runner.run(
                                run_group[0], flow_ctx, current_output,
                                parent_node_id=node_id,
                            )
                        else:
                            coros = [
                                self._task_runner.run(
                                    t, flow_ctx, current_output,
                                    parent_node_id=node_id,
                                )
                                for t in run_group
                            ]
                            results        = await run_parallel(coros)
                            current_output = results[-1] if results else current_output

            if tracer:
                tracer.finish_node(node_id, current_output)
            logger.info("flow finish name=%s", meta.name)
            return current_output

        except ExecutionError as e:
            if node_id not in e.trace_path:
                e.trace_path.insert(0, node_id)
            if tracer:
                tracer.error_node(node_id, str(e))
            raise

    async def _run_branch_flow(
        self,
        meta: FlowMeta,
        global_ctx: GlobalContext,
        flow_input: Any,
        node_id: str,
    ) -> Any:
        """Resolve the condition and delegate to the matching branch flow.

        The selected branch flow is executed as a child flow with its own
        node ID derived from *node_id*.

        Raises
        ------
        ExecutionError
            When no branch key matches and no fallback is configured.
        """
        value = _get_condition_value(flow_input, meta.condition)

        # Select the target FlowMeta.
        target_meta: FlowMeta | None = None
        selected: str = ""

        if value is not None and str(value) in meta.branches:
            target_meta = meta.branches[str(value)]
            selected    = str(value)
        elif meta.fallback is not None:
            target_meta = meta.fallback
            selected    = "__fallback__"

        if target_meta is None:
            raise ExecutionError(
                meta.name,
                f"no branch matched condition value {value!r} and no fallback "
                f"is configured.  Available keys: {list(meta.branches.keys())}",
            )

        logger.debug(
            "flow branch selected=%r cond_value=%r flow=%s",
            selected, value, meta.name,
        )

        # Delegate execution to the selected branch flow.
        return await self.run(
            target_meta,
            global_ctx,
            flow_input,
            parent_node_id=node_id,
        )
