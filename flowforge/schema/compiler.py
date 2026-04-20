"""Compile a ``GlobalMeta`` annotation tree into a ``FlowForgeDAG``.

The compiler performs a single depth-first traversal of the metadata tree
produced by the FlowForge decorators and creates one ``DAGNode`` per
annotated entity.  Edges are added to model two kinds of relationships:

``parent_child``
    Structural containment: global → flow → task → step, with recursion for
    nested flows/tasks.  Branch target tasks/flows are also added as children
    of their dispatcher node under this edge type.

``depends_on``
    Execution dependency declared via ``@flow(depends_on=[...])``.  These
    edges are added in a second pass after all nodes exist.

Branch dispatchers in the DAG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When a flow, task, or step is a branch dispatcher (``meta.is_branch``), its
branch target nodes are added to the DAG as children.  This makes *all
possible execution paths* visible in the static graph — even though only one
path is taken at runtime.

Example::

    # @task with branching
    @task(
        name="search",
        condition=BranchCondition(field="source", enum=["web", "db"]),
        branches={"web": WebSearchTask, "db": DBSearchTask},
    )
    class SearchTask: ...

    # Resulting DAG fragment:
    #   global.pipeline.search          (TASK, is_branch=True)
    #   ├─ global.pipeline.search.web_search  (TASK)
    #   └─ global.pipeline.search.db_search   (TASK)
"""
from __future__ import annotations

import logging

from flowforge.annotations.metadata import (
    GlobalMeta,
    FlowMeta,
    TaskMeta,
    StepMeta,
)
from flowforge.schema.dag import DAGNode, DAGEdge, FlowForgeDAG, NodeType

logger = logging.getLogger(__name__)


def compile_dag(global_meta: GlobalMeta) -> FlowForgeDAG:
    """Traverse the annotation metadata tree and build a ``FlowForgeDAG``.

    Parameters
    ----------
    global_meta:
        The ``GlobalMeta`` produced by the ``@global_config`` decorator.

    Returns
    -------
    FlowForgeDAG
        A fully connected DAG with all ``parent_child`` and ``depends_on``
        edges populated.
    """
    dag = FlowForgeDAG()

    # Root node — every agent has exactly one GLOBAL node.
    dag.add_node(DAGNode(
        id="global",
        type=NodeType.GLOBAL,
        name="global",
        meta=global_meta,
        parent_id=None,
    ))

    for flow_meta in global_meta.flows:
        _add_flow(dag, flow_meta, parent_id="global")

    # Second pass: add cross-flow dependency edges declared via depends_on.
    _add_depends_on_edges(dag)

    return dag


# ---------------------------------------------------------------------------
# Internal: Flow
# ---------------------------------------------------------------------------

def _add_flow(dag: FlowForgeDAG, flow_meta: FlowMeta, parent_id: str) -> str:
    """Add a flow node (and all its descendants) to *dag*.

    Parameters
    ----------
    dag:
        The DAG being constructed.
    flow_meta:
        Metadata for the flow to add.
    parent_id:
        DAG node ID of the parent (``"global"`` or another flow's ID).

    Returns
    -------
    str
        The DAG node ID of the newly created flow node.
    """
    flow_id = f"{parent_id}.{flow_meta.name}"

    dag.add_node(DAGNode(
        id=flow_id,
        type=NodeType.FLOW,
        name=flow_meta.name,
        meta=flow_meta,
        parent_id=parent_id,
    ))
    dag.add_edge(DAGEdge(
        source_id=parent_id,
        target_id=flow_id,
        edge_type="parent_child",
    ))

    if flow_meta.is_branch:
        # Branch dispatcher: add each branch target flow as a child so that
        # the full set of possible execution paths is visible in the DAG.
        for _key, branch_flow_meta in flow_meta.branches.items():
            _add_flow(dag, branch_flow_meta, parent_id=flow_id)
        if flow_meta.fallback:
            _add_flow(dag, flow_meta.fallback, parent_id=flow_id)
    else:
        # Normal flow: add child flows first, then the flow's own tasks.
        for child_flow in flow_meta.child_flows:
            _add_flow(dag, child_flow, parent_id=flow_id)
        for task_meta in flow_meta.tasks:
            _add_task(dag, task_meta, parent_id=flow_id)

    return flow_id


# ---------------------------------------------------------------------------
# Internal: Task
# ---------------------------------------------------------------------------

def _add_task(dag: FlowForgeDAG, task_meta: TaskMeta, parent_id: str) -> str:
    """Add a task node (and all its descendants) to *dag*.

    Three cases are handled:

    1. **Branch task** — ``task_meta.is_branch == True``: adds each branch
       target task as a child of this task.
    2. **Leaf task** — ``task_meta.is_leaf == True``: adds each ``@step``
       as a child of this task, sorted by ``order``.
    3. **Container task** — otherwise: adds each child task recursively.

    Parameters
    ----------
    dag:
        The DAG being constructed.
    task_meta:
        Metadata for the task to add.
    parent_id:
        DAG node ID of the parent (a flow or another task).

    Returns
    -------
    str
        The DAG node ID of the newly created task node.
    """
    task_id = f"{parent_id}.{task_meta.name}"

    dag.add_node(DAGNode(
        id=task_id,
        type=NodeType.TASK,
        name=task_meta.name,
        meta=task_meta,
        parent_id=parent_id,
    ))
    dag.add_edge(DAGEdge(
        source_id=parent_id,
        target_id=task_id,
        edge_type="parent_child",
    ))

    if task_meta.is_branch:
        # Branch dispatcher: add each branch target task as a child so that
        # all possible paths are visible in the static DAG.
        for _key, branch_task_meta in task_meta.branches.items():
            _add_task(dag, branch_task_meta, parent_id=task_id)
        if task_meta.fallback:
            _add_task(dag, task_meta.fallback, parent_id=task_id)
    elif task_meta.is_leaf:
        # Leaf task: add @step nodes sorted by order number.
        for node_meta in sorted(task_meta.steps, key=lambda s: s.order):
            _add_step(dag, node_meta, parent_id=task_id)
    else:
        # Container task: recurse into child tasks.
        for child_task in task_meta.child_tasks:
            _add_task(dag, child_task, parent_id=task_id)

    return task_id


# ---------------------------------------------------------------------------
# Internal: Step
# ---------------------------------------------------------------------------

def _add_step(dag: FlowForgeDAG, meta: StepMeta, parent_id: str) -> str:
    """Add a single step node to *dag*.

    The node ID includes the ``order`` number in square brackets so that two
    steps with the same function name but in different tasks remain unique::

        "global.my_flow.my_task.validate[1]"

    A step with ``meta.is_branch == True`` is still a ``NodeType.STEP``
    node — branch dispatching is a runtime behaviour, not a structural type.

    Parameters
    ----------
    dag:
        The DAG being constructed.
    meta:
        The ``StepMeta`` for this step.
    parent_id:
        DAG node ID of the parent leaf task.

    Returns
    -------
    str
        The DAG node ID of the newly created step node.
    """
    name    = meta.func.__name__
    node_id = f"{parent_id}.{name}[{meta.order}]"

    dag.add_node(DAGNode(
        id=node_id,
        type=NodeType.STEP,
        name=name,
        meta=meta,
        parent_id=parent_id,
    ))
    dag.add_edge(DAGEdge(
        source_id=parent_id,
        target_id=node_id,
        edge_type="parent_child",
    ))

    return node_id


# ---------------------------------------------------------------------------
# Internal: depends_on edges
# ---------------------------------------------------------------------------

def _add_depends_on_edges(dag: FlowForgeDAG) -> None:
    """Add cross-flow ``depends_on`` edges declared via ``@flow(depends_on=[...])``.

    This pass runs *after* all nodes have been added so that both the source
    and the target of each dependency edge are guaranteed to exist.

    Name resolution
    ~~~~~~~~~~~~~~~
    A ``depends_on`` entry may be either:

    * A **full DAG node ID** (e.g. ``"global.pipeline.search"``), which is
      looked up directly.
    * A **short name** (e.g. ``"search"``), which is resolved by scanning all
      FLOW nodes for a matching ``name`` field.

    When a short name is ambiguous (multiple flows share that name), a warning
    is logged — prefer full IDs in that case.
    """
    # Build lookup tables for both full-ID and short-name resolution.
    flow_by_id:   dict[str, DAGNode] = {n.id: n for n in dag.nodes_by_type(NodeType.FLOW)}
    flow_by_name: dict[str, DAGNode] = {}

    for node in dag.nodes_by_type(NodeType.FLOW):
        if node.name in flow_by_name:
            logger.warning(
                "depends_on name '%s' is ambiguous (multiple flows share this "
                "name).  Use the full dotted ID instead.",
                node.name,
            )
        flow_by_name[node.name] = node

    for node in dag.nodes_by_type(NodeType.FLOW):
        flow_meta: FlowMeta = node.meta
        for dep_ref in flow_meta.depends_on:
            # Accept both full IDs (preferred) and short names.
            dep_node = flow_by_id.get(dep_ref) or flow_by_name.get(dep_ref)
            if dep_node is None:
                logger.warning(
                    "flow '%s' has depends_on='%s' but no matching flow was "
                    "found — skipping edge.",
                    node.id,
                    dep_ref,
                )
                continue
            dag.add_edge(DAGEdge(
                source_id=dep_node.id,
                target_id=node.id,
                edge_type="depends_on",
            ))
