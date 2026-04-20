"""FlowForge annotations package.

Re-exports all public decorator functions and metadata classes so that users
can import from either ``flowforge`` or ``flowforge.annotations``.

Note: ``branch`` is not exported — branch dispatching is now a parameter of
``@step``, ``@task``, and ``@flow``.  See the decorator docstrings for usage.
"""
from flowforge.annotations.decorators import step, task, flow, global_config
from flowforge.annotations.metadata import (
    StepMeta,
    TaskMeta,
    FlowMeta,
    GlobalMeta,
)

__all__ = [
    "step",
    "task",
    "flow",
    "global_config",
    "StepMeta",
    "TaskMeta",
    "FlowMeta",
    "GlobalMeta",
]
