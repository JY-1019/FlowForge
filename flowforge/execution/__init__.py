"""FlowForge execution package.

Re-exports the context classes, runners, and engine used during agent
execution.

Note: ``BranchContext`` and ``BranchRunner`` have been removed.  Branch
dispatching is handled inline by ``StepRunner``, ``TaskRunner``, and
``FlowRunner``.  Use ``StepContext.selected_branch`` and
``StepContext.condition_value`` to inspect branch selection within a handler.
"""
from flowforge.execution.context import (
    GlobalContext,
    FlowContext,
    TaskContext,
    StepContext,
)
from flowforge.execution.engine import ExecutionEngine
from flowforge.execution.runner import FlowRunner, TaskRunner, StepRunner

__all__ = [
    "GlobalContext",
    "FlowContext",
    "TaskContext",
    "StepContext",
    "ExecutionEngine",
    "FlowRunner",
    "TaskRunner",
    "StepRunner",
]
