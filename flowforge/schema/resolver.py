"""Dependency resolution, topological sort, and cycle detection."""
from __future__ import annotations

from flowforge.schema.dag import FlowForgeDAG, DAGNode, NodeType
from flowforge.errors import CycleDetectedError


def resolve_execution_order(dag: FlowForgeDAG) -> list[DAGNode]:
    """Return all nodes in topological order for execution."""
    cycles = dag.detect_cycles()
    if cycles:
        raise CycleDetectedError(cycles[0])
    return dag.topological_order()


def get_flow_execution_order(dag: FlowForgeDAG) -> list[DAGNode]:
    """Return only FLOW nodes in execution order (respecting depends_on)."""
    all_ordered = dag.topological_order()
    return [n for n in all_ordered if n.type == NodeType.FLOW]


def get_task_execution_order(dag: FlowForgeDAG, flow_node_id: str) -> list[DAGNode]:
    """Return TASK nodes that are direct children of a flow in order."""
    children = dag.get_children(flow_node_id)
    return [c for c in children if c.type == NodeType.TASK]


def get_step_execution_order(dag: FlowForgeDAG, task_node_id: str) -> list[DAGNode]:
    """Return STEP/BRANCH nodes inside a task, sorted by order."""
    children = dag.get_children(task_node_id)
    step_branch = [c for c in children if c.type in (NodeType.STEP, NodeType.BRANCH)]

    def _order(node: DAGNode) -> int:
        return node.meta.order  # type: ignore[union-attr]

    return sorted(step_branch, key=_order)
