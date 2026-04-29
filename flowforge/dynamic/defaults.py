"""Defaults applied to generated dynamic FlowForge metadata."""
from __future__ import annotations

from typing import Any

from flowforge.types import DynamicRunOptions


def apply_dynamic_defaults(flow_meta: Any, options: DynamicRunOptions | None) -> None:
    """Apply runtime defaults to a generated ``FlowMeta`` tree in-place."""
    if options is None:
        return

    timeout = int(getattr(options, "generated_step_timeout_seconds", 0) or 0)
    if timeout <= 0:
        return

    seen_flows: set[int] = set()
    seen_tasks: set[int] = set()

    def visit_flow(meta: Any) -> None:
        marker = id(meta)
        if marker in seen_flows:
            return
        seen_flows.add(marker)

        for task_meta in getattr(meta, "tasks", []) or []:
            visit_task(task_meta)
        for child_flow in getattr(meta, "child_flows", []) or []:
            visit_flow(child_flow)
        for branch_flow in (getattr(meta, "branches", {}) or {}).values():
            visit_flow(branch_flow)
        fallback = getattr(meta, "fallback", None)
        if fallback is not None:
            visit_flow(fallback)

    def visit_task(meta: Any) -> None:
        marker = id(meta)
        if marker in seen_tasks:
            return
        seen_tasks.add(marker)

        for step_meta in getattr(meta, "steps", []) or []:
            current = int(getattr(step_meta, "timeout_seconds", 0) or 0)
            if current < timeout:
                step_meta.timeout_seconds = timeout
        for child_task in getattr(meta, "child_tasks", []) or []:
            visit_task(child_task)
        for branch_task in (getattr(meta, "branches", {}) or {}).values():
            visit_task(branch_task)
        fallback = getattr(meta, "fallback", None)
        if fallback is not None:
            visit_task(fallback)

    visit_flow(flow_meta)
