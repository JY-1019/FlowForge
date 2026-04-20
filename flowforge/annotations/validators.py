"""Compile-time validators for FlowForge annotations.

All validators run at *decoration* time (when ``@task`` or ``@step`` is
applied), so errors are reported immediately — not deferred until the first
``engine.run()`` call.

Validators
----------
``validate_order_uniqueness``
    Ensures that within each ``order`` group of a leaf task, at most one
    ``@step`` carries ``unique=True``.  Multiple ``@step`` nodes sharing
    the same ``order`` are now **allowed** — they execute in parallel.
    Only having two or more ``unique=True`` steps at the same order is
    an error (it would be ambiguous which one should run exclusively).

``validate_io_chain``
    Ensures that *between-group* transitions have compatible I/O schemas:
    the ``output_schema`` of the last step in order group *N* must equal
    the ``input_schema`` of the first step in order group *N+1* when both
    are explicitly declared.  Within-group pairs (steps sharing the same
    ``order``) are **skipped** because they all receive the same input
    rather than chaining to each other.

``validate_step_branch_handlers``
    Ensures all handler callables listed in a branching ``@step``'s
    ``branches`` dict share the same return-type annotation, so the
    next step in the chain always receives a consistent type.
"""
from __future__ import annotations

from typing import Any, get_type_hints, TYPE_CHECKING

from flowforge.errors import OrderConflictError, IOBindingError, BranchOutputMismatchError

if TYPE_CHECKING:
    from flowforge.annotations.metadata import TaskMeta, StepMeta


def validate_order_uniqueness(task_meta: TaskMeta) -> None:
    """Ensure at most one ``unique=True`` step exists per order group.

    Same-order steps are *allowed* — they will execute **in parallel** at
    runtime.  The only error condition is when two or more steps in the same
    order group both declare ``unique=True``, which would be ambiguous
    (only one can be the exclusive runner).

    This validator is a no-op for container tasks and branch tasks because
    neither kind owns ``@step`` nodes directly.

    Parameters
    ----------
    task_meta:
        The ``TaskMeta`` produced by the ``@task`` decorator.

    Raises
    ------
    OrderConflictError
        When two or more steps inside the same order group both have
        ``unique=True``.
    """
    if not task_meta.is_leaf:
        # Container tasks and branch tasks do not have a step chain to
        # validate — skip silently.
        return

    # Group steps by order value.
    order_groups: dict[int, list[StepMeta]] = {}
    for node in task_meta.steps:
        order_groups.setdefault(node.order, []).append(node)

    # Within each group, at most one step may be marked unique=True.
    for order, group in order_groups.items():
        unique_count = sum(1 for s in group if s.unique)
        if unique_count > 1:
            raise OrderConflictError(task_meta.name, order)


def validate_io_chain(task_meta: TaskMeta) -> None:
    """Validate I/O schema compatibility between consecutive order groups.

    Steps are sorted by ``order`` and validated **between groups** only —
    steps that share the same ``order`` receive the same input (parallel
    execution) and therefore are not validated against each other.

    For each pair of adjacent groups ``(group_N, group_N+1)`` the check is:

        last_step_of_group_N.output_schema == first_step_of_group_N+1.input_schema

    The check is *skipped* when either schema is ``None`` — ``None`` means
    "pass through without validation" (auto-binding).

    Parameters
    ----------
    task_meta:
        The ``TaskMeta`` produced by the ``@task`` decorator.

    Raises
    ------
    IOBindingError
        When both schemas are explicitly declared **and** they differ at a
        between-group boundary.
    """
    if not task_meta.is_leaf or len(task_meta.steps) < 2:
        # Nothing to chain-validate for non-leaf tasks or single-step tasks.
        return

    sorted_steps = sorted(task_meta.steps, key=lambda s: s.order)

    for current, next_node in zip(sorted_steps, sorted_steps[1:]):
        # Skip within-group pairs: steps sharing the same order run in
        # parallel and do not chain to each other.
        if current.order == next_node.order:
            continue

        out_schema = current.output_schema
        in_schema  = next_node.input_schema

        # Skip the check when either side uses auto-binding (schema is None).
        if out_schema is None or in_schema is None:
            continue

        if out_schema is not in_schema:
            raise IOBindingError(
                _step_name(current),
                _step_name(next_node),
                f"output={out_schema.__name__} != input={in_schema.__name__}",
            )


def validate_step_branch_handlers(step_meta: StepMeta) -> None:
    """Validate that all handlers in a branching ``@step`` share the same return type.

    When a ``@step`` is a branch dispatcher (``step_meta.is_branch == True``)
    every handler in ``step_meta.branches`` should return the same Pydantic
    model so that the next step in the chain always receives a consistent
    type.

    The check is lenient: handlers *without* a return-type annotation are
    skipped.  Only when **two or more** annotated handlers carry **different**
    return types is an error raised.

    Parameters
    ----------
    step_meta:
        A ``StepMeta`` instance with ``is_branch == True``.

    Raises
    ------
    BranchOutputMismatchError
        When two or more handlers have different explicit return-type
        annotations.
    """
    if not step_meta.branches:
        # No handlers to compare — nothing to validate.
        return

    # Collect return-type annotations from all handlers that declare one.
    # Handlers without annotations are assumed compatible and are skipped.
    annotated: dict[str, type] = {}
    for key, handler in step_meta.branches.items():
        hints: dict[str, Any] = {}
        try:
            hints = get_type_hints(handler)
        except Exception:
            # get_type_hints() can fail for lambdas or compiled code — treat
            # as "no annotation" and skip.
            pass
        ret = hints.get("return")
        if ret is not None:
            annotated[key] = ret

    unique_types = set(annotated.values())
    if len(unique_types) > 1:
        # Build a human-readable detail string like "csv: CsvRow, json: JsonRow"
        detail = ", ".join(f"{k}: {v.__name__}" for k, v in annotated.items())
        # Use the function name as the branch identifier in the error message.
        branch_name = step_meta.func.__name__
        raise BranchOutputMismatchError(branch_name, detail)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _step_name(node: StepMeta) -> str:
    """Return a human-readable name for a ``StepMeta`` node.

    Used in error messages to identify which step caused a validation error.
    """
    func = getattr(node, "func", None)
    if func is not None and callable(func):
        return func.__name__
    return repr(node)
