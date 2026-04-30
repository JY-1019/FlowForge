"""Schema contract helpers for dynamic flow chaining."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flowforge.annotations.metadata import FlowMeta, TaskMeta

# ---------------------------------------------------------------------------
# Schema introspection (used to bridge dynamic flow ↔ existing flow schemas)
# ---------------------------------------------------------------------------

def _entry_input_schema(flow_meta: "FlowMeta") -> type | None:
    """Return the first Pydantic ``input_schema`` along *flow_meta*'s entry path.

    Walks down from the flow → first ordered child flow / task → first step
    and stops at the first non-``None`` ``input_schema``.  Returns ``None``
    when nothing along the path declares one (auto-binding) or when the
    entry path traverses a branch dispatcher (unpredictable).
    """
    if flow_meta.input_schema is not None:
        return flow_meta.input_schema
    if getattr(flow_meta, "is_branch", False):
        return None

    def _task_key(t: "TaskMeta", fallback: int) -> tuple[int, int]:
        return (t.order if t.order is not None else fallback, fallback)

    def _flow_key(f: "FlowMeta", fallback: int) -> tuple[int, int]:
        return (f.order if f.order is not None else fallback, fallback)

    children: list[tuple[tuple[int, int], str, object]] = []
    for idx, cf in enumerate(flow_meta.child_flows):
        children.append((_flow_key(cf, idx), "flow", cf))
    offset = len(flow_meta.child_flows)
    for idx, t in enumerate(flow_meta.tasks):
        children.append((_task_key(t, offset + idx), "task", t))
    if not children:
        return None
    children.sort(key=lambda item: item[0])

    _, kind, first = children[0]
    if kind == "flow":
        return _entry_input_schema(first)  # type: ignore[arg-type]
    return _task_entry_input_schema(first)  # type: ignore[arg-type]


def _task_entry_input_schema(task_meta: "TaskMeta") -> type | None:
    if task_meta.input_schema is not None:
        return task_meta.input_schema
    if task_meta.is_branch:
        return None
    if task_meta.is_leaf:
        if not task_meta.steps:
            return None
        first_step = min(task_meta.steps, key=lambda s: s.order)
        return first_step.input_schema
    if not task_meta.child_tasks:
        return None
    first_child = min(
        task_meta.child_tasks,
        key=lambda t: (t.order if t.order is not None else 0),
    )
    return _task_entry_input_schema(first_child)


def _exit_output_schema(flow_meta: "FlowMeta") -> type | None:
    """Return the declared ``output_schema`` at *flow_meta*'s exit path.

    Walks down to the last ordered child flow / task / step and returns its
    ``output_schema`` (or the flow's own ``output_schema`` when declared).
    Returns ``None`` when nothing is declared along the exit path.
    """
    if flow_meta.output_schema is not None:
        return flow_meta.output_schema
    if getattr(flow_meta, "is_branch", False):
        return None

    children: list[tuple[tuple[int, int], str, object]] = []
    for idx, cf in enumerate(flow_meta.child_flows):
        key = (cf.order if cf.order is not None else idx, idx)
        children.append((key, "flow", cf))
    offset = len(flow_meta.child_flows)
    for idx, t in enumerate(flow_meta.tasks):
        key = (t.order if t.order is not None else offset + idx, offset + idx)
        children.append((key, "task", t))
    if not children:
        return None
    children.sort(key=lambda item: item[0])

    _, kind, last = children[-1]
    if kind == "flow":
        return _exit_output_schema(last)  # type: ignore[arg-type]
    return _task_exit_output_schema(last)  # type: ignore[arg-type]


def _task_exit_output_schema(task_meta: "TaskMeta") -> type | None:
    if task_meta.output_schema is not None:
        return task_meta.output_schema
    if task_meta.is_branch:
        return None
    if task_meta.is_leaf:
        if not task_meta.steps:
            return None
        last_step = max(task_meta.steps, key=lambda s: s.order)
        return last_step.output_schema
    if not task_meta.child_tasks:
        return None
    last_child = max(
        task_meta.child_tasks,
        key=lambda t: (t.order if t.order is not None else 0),
    )
    return _task_exit_output_schema(last_child)


def _schema_to_contract(model_cls: type | None) -> dict[str, Any] | None:
    """Return ``model_cls.model_json_schema()`` or ``None`` when not a model."""
    if model_cls is None:
        return None
    try:
        schema = model_cls.model_json_schema()
    except Exception:
        return None
    return schema


def _summarise_schema_mismatch(
    produced: dict[str, Any] | None,
    expected: dict[str, Any],
) -> str:
    """Return a short, human-readable mismatch description for self-correction."""
    if produced is None:
        return (
            "the generated flow did not declare any output_schema and no explicit "
            "output contract was produced"
        )

    expected_required = set(expected.get("required", []) or [])
    produced_props = set((produced.get("properties") or {}).keys())
    missing = sorted(expected_required - produced_props)
    extra = sorted(produced_props - set((expected.get("properties") or {}).keys()))

    parts: list[str] = []
    if missing:
        parts.append(f"missing required top-level keys: {missing}")
    if extra:
        parts.append(f"unexpected top-level keys: {extra}")
    if not parts:
        parts.append("top-level key names match but value types differ")
    return "; ".join(parts)
